"""
Baza SQLite dla fact-checków z Demagog.pl
==========================================

Wczytuje ``demagog_wypowiedzi_detailed.json`` i zapisuje
w osobnej bazie ``demagog.db`` z tabelami ``claims`` i ``claims_evidence``.

Użycie
------
    # CLI
    python demagog_db.py --db data/demagog.db \
        --input data/demagog_wypowiedzi_detailed.json

    # Moduł
    from demagog_db import init_db, ingest_demagog

    conn = init_db("data/demagog.db")
    ingest_demagog("data/demagog_wypowiedzi_detailed.json", conn)
    conn.close()

Wymaga
------
    Python 3.10+
"""

from __future__ import annotations

import argparse
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
            from claims_processing.monitoring.monitor import MonitoringAgent
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
# Mapowanie pól JSON → kolumny tabeli claims
# ---------------------------------------------------------------------------
# Klucz = nazwa kolumny w tabeli `claims`
# Wartość = nazwa pola w obiekcie JSON

DEMAGOG_FIELD_MAP: dict[str, str | None] = {
    "external_id":    "id",
    "claim_text":     "statement",
    "speaker":        "person_name",
    "speaker_role":   "person_function",
    "claim_date":     "publication_date",
    "label_original": "rating",
    "topic":          None,           # pobierane z `tags` — obsługiwane osobno
    "url":            "detail_url",
}


# ---------------------------------------------------------------------------
# Normalizacja etykiet
# ---------------------------------------------------------------------------
#
# Docelowe etykiety:
#   SUPPORTS          – twierdzenie prawdziwe / zweryfikowane
#   REFUTES           – twierdzenie fałszywe
#   PARTIALLY_TRUE    – częściowo prawdziwe / manipulacja
#   NOT_ENOUGH_INFO   – nieweryfikowalne / brak danych

_DEMAGOG_LABEL_MAP: dict[str, str] = {
    "prawda":            "SUPPORTS",
    "fałsz":             "REFUTES",
    "częściowa prawda":  "PARTIALLY_TRUE",
    "częściowo prawda":  "PARTIALLY_TRUE",
    "manipulacja":       "PARTIALLY_TRUE",
    "nieweryfikowalne":  "NOT_ENOUGH_INFO",
    "brak danych":       "NOT_ENOUGH_INFO",
}


def normalize_label(original: str) -> str:
    """Mapuje oryginalną etykietę Demagog na znormalizowaną.

    Parametry
    ---------
    original : str
        Oryginalna etykieta (np. ``'Fałsz'``, ``'Prawda'``).

    Zwraca
    ------
    str
        Jedna z: ``SUPPORTS``, ``REFUTES``, ``PARTIALLY_TRUE``,
        ``NOT_ENOUGH_INFO``.
    """
    key = str(original).strip().lower()
    label = _DEMAGOG_LABEL_MAP.get(key)

    if label is None:
        log.warning("Nieznana etykieta '%s' → NOT_ENOUGH_INFO", original)
        return "NOT_ENOUGH_INFO"

    return label


# ---------------------------------------------------------------------------
# Schemat bazy danych
# ---------------------------------------------------------------------------

_CREATE_CLAIMS_SQL = """\
CREATE TABLE IF NOT EXISTS claims (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL DEFAULT 'demagog',
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

def _parse_demagog_date(raw: str | None) -> str | None:
    """Próbuje wyciągnąć datę ISO z formatu Demagog, np. '25.02.2026 godz.16:28'."""
    if not raw:
        return None
    date_part = raw.split("godz.")[0].strip().split(" ")[0].strip()
    parts = date_part.split(".")
    if len(parts) == 3:
        day, month, year = parts
        return f"{year}-{month}-{day}"
    return raw


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

def ingest_demagog(json_path: str, conn: sqlite3.Connection) -> None:
    """Wczytuje plik Demagog JSON i wstawia wiersze do ``claims``.

    Operacja jest idempotentna: istniejące wiersze z ``source='demagog'``
    są usuwane przed wstawieniem nowych.

    Parametry
    ---------
    json_path : str
        Ścieżka do ``demagog_wypowiedzi_detailed.json``.
    conn : sqlite3.Connection
        Otwarte połączenie z bazą.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {json_path}")

    with open(path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    log.info("Demagog: wczytano %d rekordów z %s", len(data), json_path)

    rows: list[tuple[Any, ...]] = []
    skipped = 0

    for entry in data:
        mapped, extra = _extract_mapped_fields(entry, DEMAGOG_FIELD_MAP)

        claim_text = mapped.get("claim_text") or ""
        if not claim_text.strip():
            skipped += 1
            continue

        # Topic: z `tags` (lista) → przecinki
        tags = entry.get("tags")
        topic = ", ".join(tags) if isinstance(tags, list) and tags else None

        # Data ISO
        claim_date = _parse_demagog_date(mapped.get("claim_date"))

        # Etykieta
        label_original = str(mapped.get("label_original") or "").strip()
        label = normalize_label(label_original)

        # Metadata: reszta pól
        meta = {k: v for k, v in extra.items() if v is not None and v != ""}

        rows.append((
            "demagog",
            str(mapped.get("external_id")) if mapped.get("external_id") is not None else None,
            claim_text,
            mapped.get("speaker"),
            mapped.get("speaker_role"),
            claim_date,
            label,
            label_original,
            topic,
            mapped.get("url"),
            json.dumps(meta, ensure_ascii=False) if meta else None,
        ))

    if skipped:
        log.warning("Demagog: pominięto %d rekordów bez claim_text", skipped)

    # Idempotentne wstawianie
    cur = conn.cursor()
    cur.execute("DELETE FROM claims WHERE source = 'demagog'")
    deleted = cur.rowcount
    if deleted:
        log.info("Demagog: usunięto %d istniejących wierszy", deleted)

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
    log.info("Demagog: wstawiono %d wierszy do tabeli claims", len(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Punkt wejścia CLI — parsuje argumenty i uruchamia ingest."""
    parser = argparse.ArgumentParser(
        description="Ingest Demagog fact-checks into a SQLite database.",
    )
    parser.add_argument(
        "--db",
        default="data/demagog.db",
        help="Ścieżka do pliku bazy SQLite (domyślnie: data/demagog.db).",
    )
    parser.add_argument(
        "--input",
        default="data/demagog_wypowiedzi_detailed.json",
        help="Ścieżka do pliku JSON z Demagog.",
    )
    args = parser.parse_args()

    mon = _get_monitoring()
    mon.start()
    conn = init_db(args.db)

    try:
        ingest_demagog(args.input, conn)

        # Podsumowanie
        total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        label_dist = conn.execute(
            "SELECT label, COUNT(*) AS cnt FROM claims GROUP BY label ORDER BY cnt DESC"
        ).fetchall()

        print(f"\n{'='*60}")
        print(f"  Baza: {args.db}")
        print(f"  Łącznie rekordów: {total}")
        print(f"{'='*60}")
        print(f"  {'Etykieta':<20s} {'Liczba':>8s}")
        print(f"  {'-'*20} {'-'*8}")
        for row in label_dist:
            print(f"  {row['label']:<20s} {row['cnt']:>8d}")
        print()

        mon.report_done(
            context="demagog_db",
            lines=[
                f"{total:,} claims ingested → {args.db}",
                "  |  ".join(f"{r['label']}: {r['cnt']}" for r in label_dist),
            ],
        )

    except Exception as exc:
        mon.report_crash(exc, context="demagog_db/main")
        raise
    finally:
        conn.close()
        mon.stop()
        log.info("Połączenie zamknięte.")


if __name__ == "__main__":
    main()
