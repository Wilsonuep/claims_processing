"""
Moduł embeddingu chunków Wikipedii
====================================

Oblicza wektory embeddingowe dla chunków wygenerowanych przez
``wikipeda_chunking.py`` i zapisuje wynik do JSONL lub SQLite,
gotowego do załadowania do bazy wektorowej.

Rekomendowany model (self-hosted, polski)
------------------------------------------
    sdadas/mmlw-retrieval-roberta-large-v2

    - 1024-wymiarowe embeddingi
    - okno kontekstowe 512 tokenów (nasze chunki ≤256 → OK)
    - NDCG@10 = 60.71 na polskim benchmarku retrieval (PIRB)
    - ~355M parametrów, ~1.4 GB VRAM (fp32) / ~700 MB (fp16)

Alternatywy:
    - sdadas/stella-pl-retrieval-8k   (PIRB avg 62.69, 8k kontekst, większy)
    - intfloat/multilingual-e5-large  (multilingual, 1024-dim, ~560M param)
    - BAAI/bge-m3                     (multilingual, 1024-dim, 8k kontekst)

Użycie
------
    python wikipedia_embedding.py \\
        --input  chunks.jsonl \\
        --output chunks_embedded.jsonl \\
        --model  sdadas/mmlw-retrieval-roberta-large-v2 \\
        --batch-size 64 \\
        --device mps          # lub cuda / cpu

Wymagane pakiety
-----------------
    pip install sentence-transformers tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Monitoring integration (lazy — only instantiated when the module is run
# as a script, never when imported by other modules)
# ---------------------------------------------------------------------------

_monitoring = None  # type: ignore[assignment]


def _get_monitoring():
    """Return the module-level MonitoringAgent singleton (lazy init)."""
    global _monitoring
    if _monitoring is None:
        try:
            from claims_processing.monitoring.monitor import MonitoringAgent
            _monitoring = MonitoringAgent()
        except Exception:
            # Monitoring import failed — return a no-op stub
            class _NoOp:
                def start(self): return self
                def stop(self): pass
                def update(self, **_): pass
                def report_crash(self, *_, **__): pass
            _monitoring = _NoOp()
    return _monitoring

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
# Domyślne parametry
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "sdadas/mmlw-retrieval-roberta-large-v2"
DEFAULT_BATCH_SIZE = 64
DEFAULT_DEVICE: str | None = None  # auto-detect: cuda > mps > cpu


# ---------------------------------------------------------------------------
# Ładowanie modelu
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple, object] = {}


def load_model(
    model_name: str = DEFAULT_MODEL,
    device: str | None = DEFAULT_DEVICE,
    trust_remote_code: bool = True,
):
    """Ładuje model embeddingowy przez sentence-transformers.

    Parametry
    ---------
    model_name : str
        Nazwa modelu z HuggingFace Hub lub ścieżka lokalna.
    device : str | None
        Urządzenie obliczeniowe: 'cuda', 'mps', 'cpu'.
        Jeśli None — automatyczny wybór (cuda > mps > cpu).
    trust_remote_code : bool
        Czy zaufać zdalnemu kodowi modelu (wymagane przez niektóre modele).

    Zwraca
    ------
    SentenceTransformer
        Załadowany model gotowy do .encode().
    """
    import torch

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    cache_key = (model_name, device)
    if cache_key in _MODEL_CACHE:
        log.info("Embedding model already loaded, reusing: %s on %s", model_name, device)
        return _MODEL_CACHE[cache_key]

    from sentence_transformers import SentenceTransformer

    log.info("Ładowanie modelu: %s  →  urządzenie: %s", model_name, device)
    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    dim = model.get_sentence_embedding_dimension()
    log.info("Model załadowany. Wymiar embeddingu: %d", dim)
    _MODEL_CACHE[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# Odczyt / zapis chunków (JSONL)
# ---------------------------------------------------------------------------

def read_chunks_jsonl(path: str | Path) -> list[dict]:
    """Wczytuje chunki z pliku JSONL (jeden JSON-obiekt na linię)."""
    chunks: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    log.info("Wczytano %d chunków z %s", len(chunks), path)
    return chunks


def write_chunks_jsonl(chunks: list[dict], path: str | Path) -> None:
    """Zapisuje chunki (ze zdaniem embedding) do pliku JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            # Konwersja numpy array → lista (JSON-serializable)
            row = dict(chunk)
            if isinstance(row.get("embedding"), np.ndarray):
                row["embedding"] = row["embedding"].tolist()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Zapisano %d chunków do %s", len(chunks), path)


# ---------------------------------------------------------------------------
# Odczyt / zapis chunków (SQLite)
# ---------------------------------------------------------------------------

# Schemat zgodny z wikipedia_db.py (id AUTOINCREMENT + osobna kolumna embedding BLOB)
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS wiki_chunks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id          TEXT    NOT NULL UNIQUE,
    page_id           INTEGER NOT NULL,
    title             TEXT    NOT NULL,
    section_title     TEXT,
    paragraph_index   INTEGER NOT NULL,
    sentence_indices  TEXT    NOT NULL,   -- JSON array
    text              TEXT    NOT NULL,
    num_tokens        INTEGER NOT NULL,
    embedding         BLOB               -- float32 numpy bytes (only used by this module)
);
"""


def write_chunks_sqlite(
    chunks: list[dict],
    db_path: str | Path,
    table: str = "wiki_chunks",
) -> None:
    """Zapisuje chunki z embeddingami do bazy SQLite.

    embedding jest przechowywany jako BLOB (surowe bajty float32 numpy),
    co jest wydajne i łatwe do odczytu przez np.frombuffer().
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(_CREATE_TABLE_SQL)

    rows = []
    for c in chunks:
        emb = c.get("embedding")
        if isinstance(emb, np.ndarray):
            emb_blob = emb.astype(np.float32).tobytes()
        elif isinstance(emb, list):
            emb_blob = np.array(emb, dtype=np.float32).tobytes()
        else:
            emb_blob = None

        sent_indices = c.get("sentence_indices", [])
        if isinstance(sent_indices, list):
            sent_indices = json.dumps(sent_indices)

        rows.append((
            c["chunk_id"],
            c["page_id"],
            c["title"],
            c["section_title"],
            c["paragraph_index"],
            sent_indices,
            c["text"],
            c["num_tokens"],
            emb_blob,
        ))

    cur.executemany(
        f"""INSERT OR REPLACE INTO {table}
            (chunk_id, page_id, title, section_title,
             paragraph_index, sentence_indices, text, num_tokens, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    con.commit()
    con.close()
    log.info("Zapisano %d chunków do SQLite: %s", len(chunks), db_path)


# ---------------------------------------------------------------------------
# Główna logika embeddingu
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: list[dict],
    model=None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = DEFAULT_DEVICE,
    show_progress: bool = True,
    normalize: bool = True,
) -> list[dict]:
    """Oblicza embeddingi dla listy chunków i dodaje pole 'embedding'.

    Parametry
    ---------
    chunks : list[dict]
        Lista słowników chunków (muszą zawierać klucz 'text').
    model : SentenceTransformer | None
        Jeśli podany, używa tego modelu. W przeciwnym razie ładuje
        model z ``model_name``.
    model_name : str
        Nazwa modelu HuggingFace (używana gdy model=None).
    batch_size : int
        Rozmiar batcha do encode().
    device : str | None
        Urządzenie obliczeniowe (cuda / mps / cpu / None=auto).
    show_progress : bool
        Czy wyświetlać pasek postępu.
    normalize : bool
        Czy normalizować embeddingi do wektorów jednostkowych
        (wymagane dla cosine similarity).

    Zwraca
    ------
    list[dict]
        Ta sama lista chunków z dodanym polem 'embedding' (np.ndarray).
    """
    if model is None:
        model = load_model(model_name, device=device)

    texts = [c["text"] for c in chunks]
    total = len(texts)
    mon = _get_monitoring()

    log.info(
        "Rozpoczynam embedding %d chunków  (batch=%d, normalize=%s)",
        total, batch_size, normalize,
    )

    # Notify monitoring: embedding is starting
    mon.update(
        agent_name="wikipedia_embedding",
        benchmark="embedding",
        done=0,
        total=total,
        correct=0,
        errors=0,
        tokens=0,
        elapsed_sec=0.0,
    )

    t0 = time.perf_counter()
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
    except Exception as exc:
        mon.report_crash(exc, context="wikipedia_embedding/embed_chunks/model.encode")
        raise
    elapsed = time.perf_counter() - t0

    speed = total / elapsed if elapsed > 0 else float("inf")
    log.info(
        "Embedding zakończony: %.1f s  (%.0f chunków/s, "
        "wymiar=%d, dtype=%s)",
        elapsed, speed, embeddings.shape[1], embeddings.dtype,
    )

    # Notify monitoring: embedding complete
    mon.update(
        agent_name="wikipedia_embedding",
        benchmark="embedding",
        done=total,
        total=total,
        correct=total,
        errors=0,
        tokens=0,
        elapsed_sec=elapsed,
    )

    # Dopisz embedding do każdego chunka.
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    return chunks


# ---------------------------------------------------------------------------
# Estymacja czasu embeddingu
# ---------------------------------------------------------------------------

def estimate_embedding_time(
    num_chunks: int,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
) -> dict:
    """Szacuje czas embeddingu na podstawie typowych prędkości.

    Zwraca słownik z szacunkami dla różnych urządzeń.
    Wartości bazowe oszacowane dla modelu RoBERTa-large (~355M param).
    """
    # Typowe prędkości (chunki/s) dla modelu ~350M parametrów
    speeds = {
        "cpu":  10,     # ~10 chunków/s (wielowątkowy CPU)
        "mps":  120,    # ~120 chunków/s (Apple Silicon M1/M2/M3)
        "cuda": 450,    # ~450 chunków/s (RTX 3090/4090, batch=64)
    }

    estimates = {}
    for dev, cps in speeds.items():
        seconds = num_chunks / cps
        estimates[dev] = {
            "urządzenie": dev,
            "chunków_na_sekundę": cps,
            "czas_sekundy": round(seconds, 1),
            "czas_minuty": round(seconds / 60, 1),
            "czas_godziny": round(seconds / 3600, 2),
        }

    return estimates


def print_time_estimates(num_chunks: int, num_articles: int | None = None) -> None:
    """Wyświetla czytelną tabelę szacunków czasu embeddingu."""
    estimates = estimate_embedding_time(num_chunks)

    print()
    print("=" * 65)
    header = f"  Szacowany czas embeddingu dla {num_chunks:,} chunków"
    if num_articles:
        header += f"  (~{num_articles:,} artykułów)"
    print(header)
    print(f"  Model: {DEFAULT_MODEL}")
    print("=" * 65)
    print(f"  {'Urządzenie':<12} {'Chunków/s':>12} {'Czas':>20}")
    print("-" * 65)
    for dev, est in estimates.items():
        secs = est["czas_sekundy"]
        if secs < 60:
            time_str = f"{secs:.0f} s"
        elif secs < 3600:
            time_str = f"{est['czas_minuty']:.1f} min"
        elif secs < 86400:
            time_str = f"{est['czas_godziny']:.1f} godz."
        else:
            days = secs / 86400
            time_str = f"{days:.1f} dni"
        label = {"cpu": "CPU", "mps": "MPS (Apple)", "cuda": "CUDA (GPU)"}.get(dev, dev)
        print(f"  {label:<12} {est['chunków_na_sekundę']:>12,} {time_str:>20}")
    print("=" * 65)
    print()


# ---------------------------------------------------------------------------
# Pipeline: chunki z pliku → embedding → zapis
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str,
    output_path: str,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = DEFAULT_DEVICE,
    output_format: str = "jsonl",    # "jsonl" lub "sqlite"
    normalize: bool = True,
) -> None:
    """Pełny pipeline: wczytaj chunki → oblicz embeddingi → zapisz.

    Parametry
    ---------
    input_path : str
        Ścieżka do pliku JSONL z chunkami (wejście).
    output_path : str
        Ścieżka do pliku wyjściowego (JSONL lub SQLite).
    model_name : str
        Nazwa modelu embeddingowego.
    batch_size : int
        Rozmiar batcha.
    device : str | None
        Urządzenie (cuda / mps / cpu).
    output_format : str
        Format wyjściowy: 'jsonl' lub 'sqlite'.
    normalize : bool
        Czy normalizować wektory.
    """
    mon = _get_monitoring()

    # 1. Wczytaj chunki
    chunks = read_chunks_jsonl(input_path)
    if not chunks:
        log.warning("Brak chunków do przetworzenia.")
        return

    # 2. Pokaż estymację czasu
    print_time_estimates(len(chunks))

    # 3. Załaduj model i oblicz embeddingi
    try:
        model = load_model(model_name, device=device)
    except Exception as exc:
        mon.report_crash(exc, context="wikipedia_embedding/run_pipeline/load_model")
        raise

    try:
        chunks = embed_chunks(
            chunks,
            model=model,
            batch_size=batch_size,
            normalize=normalize,
        )
    except Exception as exc:
        mon.report_crash(exc, context="wikipedia_embedding/run_pipeline/embed_chunks")
        raise

    # 4. Zapisz wynik
    try:
        if output_format == "sqlite":
            write_chunks_sqlite(chunks, output_path)
        else:
            write_chunks_jsonl(chunks, output_path)
    except Exception as exc:
        mon.report_crash(exc, context="wikipedia_embedding/run_pipeline/write_output")
        raise

    log.info("Pipeline zakończony pomyślnie.")


# ---------------------------------------------------------------------------
# Interfejs wiersza poleceń (CLI)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Oblicza embeddingi dla chunków polskiej Wikipedii.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  # Embedding z JSONL do JSONL (auto-detect GPU)
  python wikipedia_embedding.py -i chunks.jsonl -o chunks_emb.jsonl

  # Embedding do SQLite na Apple Silicon
  python wikipedia_embedding.py -i chunks.jsonl -o wiki.db -f sqlite -d mps

  # Tylko estymacja czasu (bez embeddingu)
  python wikipedia_embedding.py --estimate 500000
        """,
    )
    p.add_argument("-i", "--input", type=str, help="Plik JSONL z chunkami (wejście)")
    p.add_argument("-o", "--output", type=str, help="Plik wyjściowy (JSONL lub SQLite)")
    p.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL,
                   help=f"Nazwa modelu HuggingFace (domyślnie: {DEFAULT_MODEL})")
    p.add_argument("-b", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Rozmiar batcha (domyślnie: {DEFAULT_BATCH_SIZE})")
    p.add_argument("-d", "--device", type=str, default=None,
                   choices=["cuda", "mps", "cpu"],
                   help="Urządzenie obliczeniowe (domyślnie: auto-detect)")
    p.add_argument("-f", "--format", type=str, default="jsonl",
                   choices=["jsonl", "sqlite"],
                   help="Format wyjściowy (domyślnie: jsonl)")
    p.add_argument("--no-normalize", action="store_true",
                   help="Nie normalizuj embeddingów")
    p.add_argument("--estimate", type=int, metavar="N",
                   help="Tylko pokaż estymację czasu dla N chunków (bez embeddingu)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Tryb estymacji — nie wymaga modelu ani pliku wejściowego
    if args.estimate:
        print_time_estimates(args.estimate)
        return

    if not args.input or not args.output:
        print("Błąd: wymagane argumenty --input i --output", file=sys.stderr)
        print("Użyj --help aby zobaczyć dostępne opcje.", file=sys.stderr)
        sys.exit(1)

    mon = _get_monitoring()
    mon.start()
    try:
        run_pipeline(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
            output_format=args.format,
            normalize=not args.no_normalize,
        )
    except Exception as exc:
        mon.report_crash(exc, context="wikipedia_embedding/main")
        raise
    finally:
        mon.stop()


if __name__ == "__main__":
    main()
