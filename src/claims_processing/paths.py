"""
Centralna konfiguracja ścieżek (paths)
=======================================

Jedyne źródło prawdy dla lokalizacji danych, baz wynikowych i katalogu
Wikipedii. Każdy moduł, który wcześniej wyliczał ``PROJECT_ROOT`` przez
``Path(__file__).parent.parent`` lub ``os.path.dirname(...)`` powinien
importować stałe stąd. Dzięki temu:

* głębokość zagnieżdżenia modułu w pakiecie nie ma znaczenia,
* przeniesienie danych do innej struktury katalogów to zmiana w jednym
  pliku,
* ścieżki można nadpisać zmiennymi środowiskowymi (przydatne, gdy
  ``wiki.db`` leży na innym dysku niż repozytorium).

Override przez zmienne środowiskowe
------------------------------------
    CLAIMS_DATA_DIR      — katalog z danymi (domyślnie ``<root>/data``)
    CLAIMS_RESULTS_DIR   — katalog z wynikami (domyślnie ``<root>/results``)
    BM25_WIKI_DB         — pełna ścieżka do wiki.db dla agentów BM25
    RAG_WIKI_DB          — pełna ścieżka do wiki.db dla agentów RAG

Kolejność rozwiązywania ``PROJECT_ROOT``:
    1. zmienna środowiskowa ``CLAIMS_PROJECT_ROOT`` (jeśli ustawiona),
    2. korzeń repozytorium wyliczony względem tego pliku
       (``src/claims_processing/paths.py`` → ``parents[2]``).
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_project_root() -> Path:
    env = os.getenv("CLAIMS_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    # paths.py -> claims_processing -> src -> <repo root>
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Katalogi główne
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = _resolve_project_root()

DATA_DIR: Path = Path(os.getenv("CLAIMS_DATA_DIR", str(PROJECT_ROOT / "data")))
RESULTS_DIR: Path = Path(os.getenv("CLAIMS_RESULTS_DIR", str(PROJECT_ROOT / "results")))

RAW_DIR: Path = DATA_DIR / "raw"
BENCHMARKS_DIR: Path = DATA_DIR / "benchmarks"
WIKI_DIR: Path = DATA_DIR / "wiki"

# ---------------------------------------------------------------------------
# Konkretne pliki
# ---------------------------------------------------------------------------

AM_BENCHMARK_CSV: Path = BENCHMARKS_DIR / "am_benchmark.csv"
AM_BENCHMARK_DB: Path = BENCHMARKS_DIR / "am_benchmark.db"
AM_BENCHMARK_4K_DB: Path = BENCHMARKS_DIR / "am_benchmark_4k.db"

WIKI_DB: Path = WIKI_DIR / "wiki.db"

RESULTS_AM_DB: Path = RESULTS_DIR / "results_am_benchmark.db"
RESULTS_AM_SUBSAMPLE_DB: Path = RESULTS_DIR / "results_am_subsample.db"

# Demagog (zarchiwizowane w extras/ — ścieżki utrzymane dla kompletności)
DEMAGOG_DB: Path = BENCHMARKS_DIR / "demagog.db"
RESULTS_DEMAGOG_DB: Path = RESULTS_DIR / "results_demagog.db"


# ---------------------------------------------------------------------------
# Wiki DB z możliwością nadpisania per-agent (BM25 / RAG)
# ---------------------------------------------------------------------------

def bm25_wiki_db() -> str:
    """Ścieżka do wiki.db dla agentów BM25 (``BM25_WIKI_DB`` ma pierwszeństwo)."""
    return os.getenv("BM25_WIKI_DB", str(WIKI_DB))


def rag_wiki_db() -> str:
    """Ścieżka do wiki.db dla agentów RAG (``RAG_WIKI_DB`` ma pierwszeństwo)."""
    return os.getenv("RAG_WIKI_DB", str(WIKI_DB))


def ensure_dirs() -> None:
    """Tworzy katalogi danych/wyników, jeśli nie istnieją."""
    for d in (DATA_DIR, RAW_DIR, BENCHMARKS_DIR, WIKI_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RESULTS_DIR",
    "RAW_DIR",
    "BENCHMARKS_DIR",
    "WIKI_DIR",
    "AM_BENCHMARK_CSV",
    "AM_BENCHMARK_DB",
    "AM_BENCHMARK_4K_DB",
    "WIKI_DB",
    "RESULTS_AM_DB",
    "RESULTS_AM_SUBSAMPLE_DB",
    "DEMAGOG_DB",
    "RESULTS_DEMAGOG_DB",
    "bm25_wiki_db",
    "rag_wiki_db",
    "ensure_dirs",
]
