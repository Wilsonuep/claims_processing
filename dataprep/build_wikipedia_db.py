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

from dataprep.wikipeda_chunking import build_article_chunks, Article, Section
from dataprep.wikipedia_db import init_db, insert_chunk_batch
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
    parser.add_argument("--batch-size", type=int, default=500, help="Number of chunks to accumulate before embedding + inserting")
    parser.add_argument("--embed-batch-size", type=int, default=64, help="Batch size passed to model.encode() (affects GPU memory)")

    args = parser.parse_args()

    mon = _get_monitoring()
    mon.start()
    try:
        _run(args, mon)
    finally:
        mon.stop()


def _fmt_dur(seconds: float) -> str:
    """Format seconds → '1h 23m', '45m 10s', or '8s'."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, s = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {s:02d}s"
    hours, m = divmod(minutes, 60)
    return f"{hours}h {m:02d}m"


def _count_lines(path: str) -> int:
    """Fast line count via binary read (no full decode needed)."""
    count = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            count += chunk.count(b"\n")
    return count


def _embed_and_insert(conn, chunk_buffer: list, embed_model, embed_batch_size: int, mon) -> int:
    """Batch-embed chunk_buffer and insert into DB. Returns number of chunks inserted."""
    if not chunk_buffer:
        return 0
    texts = [c.text for c in chunk_buffer]
    try:
        embeddings = embed_model.encode(
            texts,
            batch_size=embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except Exception as exc:
        mon.report_crash(exc, context="build_wikipedia_db/embed_batch")
        raise
    pairs = [(c, emb.tolist()) for c, emb in zip(chunk_buffer, embeddings)]
    try:
        insert_chunk_batch(conn, pairs)
    except Exception as exc:
        mon.report_crash(exc, context="build_wikipedia_db/insert_chunk_batch")
        raise
    return len(chunk_buffer)


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
    """Streaming pipeline: article-by-article to avoid loading all chunks into RAM.

    Flow:
        Phase 1 — Load embedding model (needed to know embed_dim for DB init)
        Phase 2 — Init DB
        Phase 3 — Load resume state (page_ids already in DB)
        Phase 4 — Stream JSONL → parse → chunk → batch-embed → insert
    """
    TOTAL_PHASES = 4

    # ── Phase 1: Load embedding model ────────────────────────────────────────
    t0 = _phase(1, TOTAL_PHASES, f"Loading model  '{args.embed_model}'")
    mon.update(mode="build_db", phase="loading embedding model", done=0, total=0)
    try:
        embed_model = load_model(args.embed_model)
    except Exception as e:
        mon.report_crash(e, context="build_wikipedia_db/load_model")
        log.error("Could not load model %s: %s", args.embed_model, e)
        sys.exit(1)
    for _noisy in ("sentence_transformers", "transformers", "tokenizers", "huggingface_hub", "filelock"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    embed_dim = len(embed_model.encode("test").tolist())
    _phase_done(1, TOTAL_PHASES, "Model load", t0, f"embed_dim={embed_dim}")

    # ── Phase 2: Init DB ──────────────────────────────────────────────────────
    db_path_obj = Path(args.db)
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    t0 = _phase(2, TOTAL_PHASES, f"Init DB  ({args.db})")
    mon.update(mode="build_db", phase="init DB")
    try:
        conn = init_db(str(args.db), embedding_dim=embed_dim)
    except Exception as e:
        mon.report_crash(e, context="build_wikipedia_db/init_db")
        log.error("Failed to init DB: %s", e)
        sys.exit(1)
    _phase_done(2, TOTAL_PHASES, "Init DB", t0)

    # ── Phase 3: Resume state ─────────────────────────────────────────────────
    t0 = _phase(3, TOTAL_PHASES, "Loading resume state")
    mon.update(mode="build_db", phase="resume state", done=0, total=0)
    done_page_ids: set[int] = {
        row[0] for row in conn.execute("SELECT DISTINCT page_id FROM wiki_chunks")
    }
    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)
    total_lines = _count_lines(str(args.input))
    log.info("  JSONL total lines: %d  |  Already in DB: %d articles",
             total_lines, len(done_page_ids))
    if done_page_ids:
        log.info("  Resuming — will skip %d already-processed articles.", len(done_page_ids))
    _phase_done(3, TOTAL_PHASES, "Resume state", t0)

    # ── Phase 4: Stream → parse → chunk → batch-embed → insert ──────────────
    t0 = _phase(4, TOTAL_PHASES,
                f"Stream + embed + insert  (chunk_buf={args.batch_size}, embed_batch={args.embed_batch_size})")
    mon.update(mode="build_db", phase="streaming", done=0, total=total_lines)

    seen_pageids: set[int] = set()   # dedup within this run
    chunk_buffer: list = []
    articles_new = 0
    articles_skipped = 0
    chunks_inserted = 0
    t_start = time.perf_counter()
    last_log_time = t_start
    # ETA is based solely on new-article throughput to avoid skip-phase contamination.
    # t_new_start is set on the first new article; remaining_new estimates articles left.
    t_new_start: float | None = None
    remaining_new = max(total_lines - len(done_page_ids), 1)

    with open(input_path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Bad JSON at line %d: %s", line_no, exc)
                continue

            if args.limit and (articles_new + articles_skipped) >= args.limit:
                break

            pid = raw.get("pageid", raw.get("id", 0))
            if pid in seen_pageids:
                continue
            seen_pageids.add(pid)

            # Resume: skip articles already committed to DB
            if pid in done_page_ids:
                articles_skipped += 1
                continue

            text = raw.get("text", "")
            if not text or not text.strip():
                continue

            sections = parse_article_text(text)
            article_chunks = build_article_chunks({
                "page_id": pid,
                "title": raw.get("title", ""),
                "sections": sections,
            })
            chunk_buffer.extend(article_chunks)
            if t_new_start is None:
                t_new_start = time.perf_counter()
            articles_new += 1

            if len(chunk_buffer) >= args.batch_size:
                chunks_inserted += _embed_and_insert(
                    conn, chunk_buffer, embed_model, args.embed_batch_size, mon
                )
                chunk_buffer.clear()

                now = time.perf_counter()
                if now - last_log_time >= 30:
                    elapsed_new = now - t_new_start
                    rate_new = articles_new / max(elapsed_new, 0.1)
                    eta = (remaining_new - articles_new) / max(rate_new, 0.001)
                    log.info(
                        "[line %d/%d  %.1f%%]  new=%d/%d  skipped=%d  chunks=%d  "
                        "%.1f art/s  ETA %s",
                        line_no, total_lines, line_no / total_lines * 100,
                        articles_new, remaining_new, articles_skipped, chunks_inserted,
                        rate_new, _fmt_dur(max(eta, 0)),
                    )
                    mon.update(
                        mode="build_db", phase="streaming",
                        done=line_no, total=total_lines,
                        elapsed_sec=now - t_start,
                    )
                    last_log_time = now

    # Flush remaining buffer
    if chunk_buffer:
        chunks_inserted += _embed_and_insert(
            conn, chunk_buffer, embed_model, args.embed_batch_size, mon
        )
        chunk_buffer.clear()

    conn.close()
    elapsed_total = time.perf_counter() - t_start
    _phase_done(4, TOTAL_PHASES, "Stream+embed+insert", t0,
                f"{chunks_inserted:,} new chunks from {articles_new:,} new articles "
                f"(+{articles_skipped:,} skipped)")

    log.info("=" * 60)
    log.info("DONE!  DB: %s", args.db)
    log.info("  New articles:     %d", articles_new)
    log.info("  Skipped (resume): %d", articles_skipped)
    log.info("  New chunks:       %d", chunks_inserted)
    log.info("  Embed dim:        %d", embed_dim)
    log.info("=" * 60)

    mon.report_done(
        context="build_wikipedia_db",
        lines=[
            f"{chunks_inserted:,} new chunks from {articles_new:,} articles",
            f"(+{articles_skipped:,} skipped — already in DB)",
            f"DB: {args.db}  |  embed_dim={embed_dim}",
        ],
    )

if __name__ == "__main__":
    main()