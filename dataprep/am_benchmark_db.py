"""
Baza SQLite dla benchmarku AM (AMU-CAI llmzszl)
=================================================

Wczytuje ``am_benchmark.csv`` i zapisuje w osobnej bazie
``am_benchmark.db`` z tabelami ``claims`` i ``claims_evidence``.

Użycie
------
    # CLI
    python am_benchmark_db.py --db data/am_benchmark.db \
        --input data/am_benchmark.csv

    # Moduł
    from am_benchmark_db import init_db, ingest_am_benchmark

    conn = init_db("data/am_benchmark.db")
    ingest_am_benchmark("data/am_benchmark.csv", conn)
    conn.close()

Wymaga
------
    Python 3.10+
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Konfiguracja logowania
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Monitoring integration (lazy — no side-effects on import)
# ---------------------------------------------------------------------------

_monitoring = None  # type: ignore[assignment]


def _get_monitoring():
    global _monitoring
    if _monitoring is None:
        try:
            from monitoring.monitor import MonitoringAgent
            _monitoring = MonitoringAgent()
        except Exception:
            class _NoOp:
                def start(self): return self
                def stop(self): pass
                def report_crash(self, *_, **__): pass
                def report_done(self, *_, **__): pass
            _monitoring = _NoOp()
    return _monitoring


# ---------------------------------------------------------------------------
# Mapowanie kolumn CSV → kolumny tabeli claims
# ---------------------------------------------------------------------------
# Klucz = nazwa kolumny w tabeli `claims`
# Wartość = nazwa kolumny w pliku CSV (None = brak odpowiednika)

AM_FIELD_MAP: dict[str, str | None] = {
    "external_id":    None,           # brak stałego ID — numerujemy wiersze
    "claim_text":     "question",     # pytanie egzaminacyjne → claim_text
    "speaker":        None,
    "speaker_role":   None,
    "claim_date":     "year",         # rok egzaminu
    "label_original": "correct_answer_index",
    "topic":          "name",         # np. „Przyroda"
    "url":            None,
}


# ---------------------------------------------------------------------------
# Normalizacja etykiet
# ---------------------------------------------------------------------------
#
# Benchmark AM zawiera pytania wielokrotnego wyboru z indeksem poprawnej
# odpowiedzi.  Każde pytanie + poprawna odpowiedź tworzy prawdziwe
# stwierdzenie, więc domyślnie mapujemy na SUPPORTS.
#
# TODO: Jeśli chcesz traktować niepoprawne odpowiedzi jako osobne wiersze
#       z etykietą REFUTES, rozbuduj logikę w `ingest_am_benchmark()`.

_AM_LABEL_MAP: dict[str, str] = {
    # Domyślnie puste — wszystkie indeksy → SUPPORTS (fallback poniżej).
    # Dodaj tu wpisy, jeśli pojawią się inne typy etykiet.
}


def normalize_label(original: str) -> str:
    """Mapuje oryginalną etykietę AM benchmark na znormalizowaną.

    Parametry
    ---------
    original : str
        Oryginalna etykieta (indeks poprawnej odpowiedzi).

    Zwraca
    ------
    str
        Jedna z: ``SUPPORTS``, ``REFUTES``, ``PARTIALLY_TRUE``,
        ``NOT_ENOUGH_INFO``.
    """
    key = str(original).strip().lower()
    label = _AM_LABEL_MAP.get(key)

    # Domyślnie: pytanie + poprawna odpowiedź → SUPPORTS
    if label is None:
        return "SUPPORTS"

    return label


# ---------------------------------------------------------------------------
# Schemat bazy danych
# ---------------------------------------------------------------------------

_CREATE_CLAIMS_SQL = """\
CREATE TABLE IF NOT EXISTS claims (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL DEFAULT 'am_benchmark',
    external_id     TEXT,
    claim_text      TEXT    NOT NULL,
    speaker         TEXT,
    speaker_role    TEXT,
    claim_date      TEXT,
    label           TEXT    NOT NULL,
    label_original  TEXT,
    topic           TEXT,
    url             TEXT,
    metadata        TEXT,   -- JSON

    UNIQUE (source, external_id)
);
"""

_CREATE_IDX_CLAIMS_SOURCE = """\
CREATE INDEX IF NOT EXISTS idx_claims_source_external_id
    ON claims (source, external_id);
"""

_CREATE_IDX_CLAIMS_LABEL = """\
CREATE INDEX IF NOT EXISTS idx_claims_label
    ON claims (label);
"""

_CREATE_CLAIMS_EVIDENCE_SQL = """\
CREATE TABLE IF NOT EXISTS claims_evidence (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id    INTEGER NOT NULL,   -- FK → claims.id
    chunk_id    TEXT    NOT NULL,    -- FK → wiki_chunks.chunk_id
    rank        INTEGER NOT NULL,
    score       REAL,
    is_gold     INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_IDX_EVIDENCE_CLAIM = """\
CREATE INDEX IF NOT EXISTS idx_claims_evidence_claim_id
    ON claims_evidence (claim_id);
"""


# ---------------------------------------------------------------------------
# Inicjalizacja bazy danych
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    """Tworzy (lub otwiera) bazę SQLite i zakłada tabele.

    Parametry
    ---------
    db_path : str
        Ścieżka do pliku ``.db`` (zostanie utworzony, jeśli nie istnieje).

    Zwraca
    ------
    sqlite3.Connection
        Otwarte połączenie.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    cur = conn.cursor()
    cur.execute(_CREATE_CLAIMS_SQL)
    cur.execute(_CREATE_IDX_CLAIMS_SOURCE)
    cur.execute(_CREATE_IDX_CLAIMS_LABEL)
    cur.execute(_CREATE_CLAIMS_EVIDENCE_SQL)
    cur.execute(_CREATE_IDX_EVIDENCE_CLAIM)
    conn.commit()

    log.info("Baza gotowa: %s (tabele: claims, claims_evidence)", db_path)
    return conn


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def _extract_mapped_fields(
    record: dict[str, Any],
    field_map: dict[str, str | None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Wyciąga pola z rekordu zgodnie z mapowaniem.

    Zwraca
    ------
    (mapped_fields, remaining_fields)
        mapped_fields — słownik z kluczami jak w field_map.
        remaining_fields — pola, które nie zostały zmapowane (→ metadata).
    """
    used_source_keys: set[str] = set()
    mapped: dict[str, Any] = {}

    for db_col, src_key in field_map.items():
        if src_key is not None and src_key in record:
            mapped[db_col] = record[src_key]
            used_source_keys.add(src_key)
        else:
            mapped[db_col] = None

    remaining = {k: v for k, v in record.items() if k not in used_source_keys}
    return mapped, remaining


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_am_benchmark(csv_path: str, conn: sqlite3.Connection) -> None:
    """Wczytuje plik AM benchmark CSV i wstawia wiersze do ``claims``.

    Każde pytanie egzaminacyjne staje się jednym wierszem. ``claim_text``
    zawiera pytanie, a ``metadata`` przechowuje listę odpowiedzi
    i indeks poprawnej.

    Operacja jest idempotentna: istniejące wiersze z
    ``source='am_benchmark'`` są usuwane.

    Parametry
    ---------
    csv_path : str
        Ścieżka do ``am_benchmark.csv``.
    conn : sqlite3.Connection
        Otwarte połączenie z bazą.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {csv_path}")

    records: list[dict[str, str]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))

    log.info("AM Benchmark: wczytano %d rekordów z %s", len(records), csv_path)

    rows: list[tuple[Any, ...]] = []
    skipped = 0

    for idx, record in enumerate(records):
        mapped, extra = _extract_mapped_fields(record, AM_FIELD_MAP)

        claim_text = mapped.get("claim_text") or ""
        if not claim_text.strip():
            skipped += 1
            continue

        # Data: rok egzaminu
        claim_date = str(mapped.get("claim_date") or "").strip() or None

        # Etykieta
        label_original = str(mapped.get("label_original") or "").strip()
        label = normalize_label(label_original)

        # Metadata: odpowiedzi, indeks poprawnej, typ egzaminu
        meta: dict[str, Any] = {}
        if "answers" in record:
            meta["answers"] = record["answers"]
        if "correct_answer_index" in record:
            meta["correct_answer_index"] = record["correct_answer_index"]
        if "type" in record:
            meta["exam_type"] = record["type"]
        for k, v in extra.items():
            if v is not None and v != "" and k not in meta:
                meta[k] = v

        # external_id: indeks wiersza (brak naturalnego ID)
        external_id = str(idx)

        rows.append((
            "am_benchmark",
            external_id,
            claim_text,
            mapped.get("speaker"),
            mapped.get("speaker_role"),
            claim_date,
            label,
            label_original,
            mapped.get("topic"),
            mapped.get("url"),
            json.dumps(meta, ensure_ascii=False) if meta else None,
        ))

    if skipped:
        log.warning("AM Benchmark: pominięto %d rekordów bez claim_text", skipped)

    # Idempotentne wstawianie
    cur = conn.cursor()
    cur.execute("DELETE FROM claims WHERE source = 'am_benchmark'")
    deleted = cur.rowcount
    if deleted:
        log.info("AM Benchmark: usunięto %d istniejących wierszy", deleted)

    cur.executemany(
        """
        INSERT INTO claims
            (source, external_id, claim_text, speaker, speaker_role,
             claim_date, label, label_original, topic, url, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    log.info("AM Benchmark: wstawiono %d wierszy do tabeli claims", len(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Punkt wejścia CLI — parsuje argumenty i uruchamia ingest."""
    parser = argparse.ArgumentParser(
        description="Ingest AM benchmark data into a SQLite database.",
    )
    parser.add_argument(
        "--db",
        default="data/am_benchmark.db",
        help="Ścieżka do pliku bazy SQLite (domyślnie: data/am_benchmark.db).",
    )
    parser.add_argument(
        "--input",
        default="data/am_benchmark.csv",
        help="Ścieżka do pliku CSV z benchmarkiem AM.",
    )
    args = parser.parse_args()

    mon = _get_monitoring()
    mon.start()
    conn = init_db(args.db)

    try:
        ingest_am_benchmark(args.input, conn)

        # Podsumowanie
        total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        label_dist = conn.execute(
            "SELECT label, COUNT(*) AS cnt FROM claims GROUP BY label ORDER BY cnt DESC"
        ).fetchall()
        topic_dist = conn.execute(
            "SELECT topic, COUNT(*) AS cnt FROM claims GROUP BY topic ORDER BY cnt DESC LIMIT 10"
        ).fetchall()

        print(f"\n{'='*60}")
        print(f"  Baza: {args.db}")
        print(f"  Łącznie rekordów: {total}")
        print(f"{'='*60}")
        print(f"  {'Etykieta':<20s} {'Liczba':>8s}")
        print(f"  {'-'*20} {'-'*8}")
        for row in label_dist:
            print(f"  {row['label']:<20s} {row['cnt']:>8d}")

        print(f"\n  {'Temat (top 10)':<30s} {'Liczba':>8s}")
        print(f"  {'-'*30} {'-'*8}")
        for row in topic_dist:
            print(f"  {str(row['topic']):<30s} {row['cnt']:>8d}")
        print()

        mon.report_done(
            context="am_benchmark_db",
            lines=[
                f"{total:,} claims ingested → {args.db}",
                "  |  ".join(f"{r['label']}: {r['cnt']}" for r in label_dist),
            ],
        )

    except Exception as exc:
        mon.report_crash(exc, context="am_benchmark_db/main")
        raise
    finally:
        conn.close()
        mon.stop()
        log.info("Połączenie zamknięte.")


if __name__ == "__main__":
    main()
