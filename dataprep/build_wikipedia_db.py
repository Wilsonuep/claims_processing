"""
build_wikipedia_db.py
=====================

Skrypt pobiera JSONL wygenerowany przez scrapper Wikipedii, mapuje płaski tekst (extract)
na ustrukturyzowane sekcje i akapity, a następnie korzysta z `wikipeda_chunking.py` oraz 
`wikipedia_db.py` w celu chunkowania, zwektoryzowania i zapisania do bazy danych wektorowej (SQLite + vec0).
"""

import argparse
import json
import logging
import os
import re
import sys
import time

from pathlib import Path

# Dodanie ścieżki do lokalnych modułów (katalogu wyżej)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprep.wikipeda_chunking import build_wiki_chunks, Article, Section
from dataprep.wikipedia_db import init_db, insert_chunks_with_embeddings
from dataprep.wikipedia_embedding import load_model

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
                def update(self, **_): pass
            _monitoring = _NoOp()
    return _monitoring


def parse_article_text(text: str) -> list[Section]:
    """
    Mapuje płaski tekst artykułu na listę obiektów Section (zgodnych z Article TypedDict).

    Scrapper Wikipedii używa następującej konwencji znaków nowej linii:
        - ``\\n\\n\\n`` (potrójny) — granica sekcji (nagłówek sekcji jest blokiem
          pomiędzy dwoma potrójnymi znakami nowej linii)
        - ``\\n\\n``  (podwójny) — granica akapitu wewnątrz tej samej sekcji
        - ``\\n``    (pojedynczy) — łamanie wierszy w akapicie (np. listy wypunktowane)

    Algorytm:
        1. Dzielimy tekst na sekcje po ``\\n\\n\\n``.
        2. Pierwsza sekcja = „Wprowadzenie" (brak nagłówka).
        3. Kolejne sekcje: pierwsza linia po splicie to nagłówek,
           reszta to akapity oddzielone ``\\n\\n``.
    """
    sections: list[Section] = []

    # Podział na sekcje (separowane co najmniej potrójnym \n)
    raw_sections = re.split(r'\n{3,}', text)

    for idx, raw_section in enumerate(raw_sections):
        raw_section = raw_section.strip()
        if not raw_section:
            continue

        if idx == 0:
            # Pierwszy blok to zawsze "Wprowadzenie" — nie ma nagłówka
            section_title = "Wprowadzenie"
            body = raw_section
        else:
            # W kolejnych blokach pierwszy wiersz to tytuł sekcji
            lines = raw_section.split('\n', 1)
            section_title = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""

        if not body:
            # Sekcja bez treści (np. „Przypisy", „Uwagi") — pomijamy
            # lub dodajemy jako pustą, żeby nie zgubić metadanych
            continue

        # Akapity wewnątrz sekcji rozdzielone podwójnym \n
        paragraphs = [p.strip() for p in body.split('\n\n') if p.strip()]

        if paragraphs:
            sections.append({
                "section_title": section_title,
                "paragraphs": paragraphs,
            })

    return sections


def main():
    parser = argparse.ArgumentParser(description="Zbuduj bazę wektorową Wikipedii z pliku JSONL.")
    parser.add_argument("--input", default="polish_wikipedia_articles.jsonl", help="Plik wejściowy JSONL ze zrzutu Wikipedii")
    parser.add_argument("--db", default="data/wiki.db", help="Ścieżka do docelowej bazy danych sqlite (vector info)")
    parser.add_argument("--limit", type=int, default=None, help="Liczba artykułów do przetworzenia w ramach bazy (do celów testowych)")
    parser.add_argument("--embed-model", default="sdadas/mmlw-retrieval-roberta-large-v2", help="Rodzaj modelu wczytywanego z HF (sentence-transformers)")
    parser.add_argument("--batch-size", type=int, default=500, help="Rozmiar batcha do zapisu bazy danych i wektoryzacji")
    
    args = parser.parse_args()

    mon = _get_monitoring()
    mon.start()
    try:
        _run(args, mon)
    finally:
        mon.stop()


def _phase(n: int, total: int, label: str) -> float:
    """Log a phase start banner and return the start timestamp."""
    log.info("── Phase %d/%d: %s ...", n, total, label)
    return time.perf_counter()


def _phase_done(n: int, total: int, label: str, t0: float, detail: str = "") -> None:
    """Log a phase completion banner with elapsed time."""
    elapsed = time.perf_counter() - t0
    suffix = f"  —  {detail}" if detail else ""
    log.info("── Phase %d/%d: done in %.1fs%s", n, total, elapsed, suffix)


def _run(args, mon) -> None:
    """Core pipeline logic, separated from main() for clean monitoring wrap."""
    TOTAL_PHASES = 6

    # ── Phase 1: Load JSONL ──────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        log.error("!!! Błąd: Nie znaleziono pliku '%s'.", args.input)
        log.error("Zanim uruchomisz ten skrypt, odpal datascrap/polish_wikipedia_webscrapper.py")
        sys.exit(1)

    t0 = _phase(1, TOTAL_PHASES, f"Loading JSONL  ({args.input})")
    mon.update(mode="build_db", phase="loading JSONL", done=0, total=0)
    raw_articles: list[dict] = []
    invalid_lines_count = 0
    with open(input_path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_articles.append(json.loads(line))
            except json.JSONDecodeError as e:
                invalid_lines_count += 1
                if invalid_lines_count <= 10:
                    log.warning(
                        "Pomijam błędną linię nr %d (JSONDecodeError): %s — pierwsze 80 znaków: %s",
                        line_no, e, line[:80],
                    )
                elif invalid_lines_count == 11:
                    log.warning("Zbyt wiele błędnych linii — wstrzymuję logowanie kolejnych błędów JSON.")
                continue
            if args.limit and len(raw_articles) >= args.limit:
                break

    if invalid_lines_count > 0:
        log.warning("Zignorowano łącznie %d błędnych linii.", invalid_lines_count)
    if not raw_articles:
        log.error("Nie wczytano żadnych artykułów — sprawdź plik wejściowy.")
        sys.exit(1)
    if args.limit:
        log.info("Limit: %d artykułów.", args.limit)
    _phase_done(1, TOTAL_PHASES, "Loading JSONL", t0, f"{len(raw_articles):,} articles loaded")

    # ── Phase 2: Deduplication ───────────────────────────────────────────────
    t0 = _phase(2, TOTAL_PHASES, "Deduplication")
    mon.update(mode="build_db", phase="deduplication", done=0, total=len(raw_articles))
    seen_pageids: set[int] = set()
    deduped_articles: list[dict] = []
    for raw in raw_articles:
        pid = raw.get("pageid", raw.get("id", 0))
        if pid in seen_pageids:
            log.warning("Duplikat pageid %s ('%s') — pomijam.", pid, raw.get("title", ""))
            continue
        seen_pageids.add(pid)
        deduped_articles.append(raw)
    dupes = len(raw_articles) - len(deduped_articles)
    raw_articles = deduped_articles
    _phase_done(2, TOTAL_PHASES, "Deduplication", t0,
                f"{len(raw_articles):,} unique articles" + (f"  ({dupes} dupes removed)" if dupes else ""))

    # ── Phase 3: Parse articles ──────────────────────────────────────────────
    t0 = _phase(3, TOTAL_PHASES, f"Parsing {len(raw_articles):,} articles")
    mon.update(mode="build_db", phase="parsing articles", done=0, total=len(raw_articles))
    structured_articles: list[Article] = []
    total_to_parse = len(raw_articles)
    for idx, raw in enumerate(raw_articles, 1):
        text = raw.get("text", "")
        if not text or not text.strip():
            continue
        parsed_sections = parse_article_text(text)
        structured_articles.append({
            "page_id": raw.get("pageid", raw.get("id", 0)),
            "title": raw.get("title", "Brak Tytułu"),
            "sections": parsed_sections,
        })
        if idx % 500 == 0 or idx == total_to_parse:
            log.info("  [%*d/%d]  %5.1f%%",
                     len(str(total_to_parse)), idx, total_to_parse, idx / total_to_parse * 100)
            mon.update(mode="build_db", phase="parsing articles", done=idx, total=total_to_parse)
    skipped = total_to_parse - len(structured_articles)
    _phase_done(3, TOTAL_PHASES, "Parsing", t0,
                f"{len(structured_articles):,} parsed" + (f"  ({skipped} empty skipped)" if skipped else ""))

    # ── Phase 4: Chunking ────────────────────────────────────────────────────
    t0 = _phase(4, TOTAL_PHASES, f"Chunking {len(structured_articles):,} articles")
    mon.update(mode="build_db", phase="chunking", done=0, total=len(structured_articles))
    try:
        chunks = build_wiki_chunks(structured_articles)
    except Exception as e:
        mon.report_crash(e, context="build_wikipedia_db/build_wiki_chunks")
        log.error("Błąd podczas cięcia artykułów (chunking): %s", e)
        sys.exit(1)
    if not chunks:
        log.warning("Brak chunków do zwektoryzowania — prawdopodobnie zbiór był pusty.")
        sys.exit(0)
    _phase_done(4, TOTAL_PHASES, "Chunking", t0, f"{len(chunks):,} chunks generated")

    # ── Phase 5: Load embedding model ────────────────────────────────────────
    t0 = _phase(5, TOTAL_PHASES, f"Loading model  '{args.embed_model}'")
    mon.update(mode="build_db", phase="loading embedding model", done=0, total=0)
    try:
        embed_model = load_model(args.embed_model)
    except Exception as e:
        mon.report_crash(e, context="build_wikipedia_db/load_model")
        log.error("Nie udało się załadować modelu %s: %s", args.embed_model, e)
        sys.exit(1)
    test_vec = embed_model.encode("test").tolist()
    embed_dim = len(test_vec)
    _phase_done(5, TOTAL_PHASES, "Model load", t0, f"embed_dim={embed_dim}")

    def embed_fn(text: str) -> list[float]:
        return embed_model.encode(text, normalize_embeddings=True).tolist()

    # ── Phase 6: Init DB + insert chunks ─────────────────────────────────────
    db_path_obj = Path(args.db)
    if not db_path_obj.parent.exists():
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)

    t0 = _phase(6, TOTAL_PHASES, f"Init DB + insert  ({args.db},  batch={args.batch_size})")
    mon.update(mode="build_db", phase="init DB", done=0, total=len(chunks))
    try:
        conn = init_db(str(args.db), embedding_dim=embed_dim)
    except Exception as e:
        mon.report_crash(e, context="build_wikipedia_db/init_db")
        log.error("Nie udało się zainicjalizować bazy SQLite-vec: %s", e)
        sys.exit(1)

    try:
        insert_chunks_with_embeddings(conn, chunks, embed_fn, batch_size=args.batch_size)
    except Exception as e:
        mon.report_crash(e, context="build_wikipedia_db/insert_chunks_with_embeddings")
        log.error("Nie powiodła się próba zapisu bazy danych: %s", e)
        conn.close()
        sys.exit(1)

    conn.close()
    _phase_done(6, TOTAL_PHASES, "DB insert", t0, f"{len(chunks):,} chunks stored")

    log.info("=" * 60)
    log.info("GOTOWE!  Baza wiedzy zasilona: %s", args.db)
    log.info("  Artykuły:  %d", len(structured_articles))
    log.info("  Chunki:    %d", len(chunks))
    log.info("  Embed dim: %d", embed_dim)
    log.info("=" * 60)

    mon.report_done(
        context="build_wikipedia_db",
        lines=[
            f"{len(chunks):,} chunks from {len(structured_articles):,} articles",
            f"DB: {args.db}  |  embed_dim={embed_dim}",
        ],
    )

if __name__ == "__main__":
    main()