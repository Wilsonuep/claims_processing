"""
Ewaluacja agentów na benchmarku AM (AMU-CAI)
=============================================

Wrapper uruchamiający generyczną pętlę ewaluacyjną
wyłącznie na bazie ``am_benchmark.db``.

Użycie
------
    python -m scripts.run_eval_am_benchmark
    python -m scripts.run_eval_am_benchmark --limit 10
    python -m scripts.run_eval_am_benchmark --agents uam_ga1,uam_ga2
    python -m scripts.run_eval_am_benchmark --clear

Wymaga
------
    Python 3.10+
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_NAME = "am_benchmark"
INPUT_DB_PATH = str(PROJECT_ROOT / "data" / "am_benchmark.db")
RESULTS_DB_PATH = str(PROJECT_ROOT / "results" / "results_am_benchmark.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Reconfigure stdout for UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Rejestracja agentów
# ---------------------------------------------------------------------------
# Tutaj importuj i rejestruj agentów, którzy mają być ewaluowani
# na benchmarku AM.
#
# Przykład:
#
#   from eval.eval_loop import register_agent
#   from agents_uam.my_agent import MyUamAgent
#   register_agent(MyUamAgent())
#
# Na razie rejestr jest pusty — dodaj agentów poniżej,
# gdy będą gotowi do ewaluacji.


def _get_completed_pairs() -> set[tuple[str, str]]:
    """Returns (base_agent_name, model_name) pairs that have a complete set of results.

    base_agent_name is derived by splitting agent_name at the first '__'
    (e.g. 'uam_ga1__bielik' → 'uam_ga1').  This lets us match a run stored
    under 'uam_ga1__bielik' against a freshly registered 'uam_ga1' that has
    the same model_name, whether or not --models was passed this time.
    """
    if not os.path.exists(RESULTS_DB_PATH) or not os.path.exists(INPUT_DB_PATH):
        return set()
    try:
        total = sqlite3.connect(INPUT_DB_PATH).execute(
            "SELECT COUNT(*) FROM claims"
        ).fetchone()[0]
        rows = sqlite3.connect(RESULTS_DB_PATH).execute(
            "SELECT agent_name, model_name, COUNT(*) FROM agent_results "
            "WHERE benchmark_name=? GROUP BY agent_name, model_name",
            (BENCHMARK_NAME,),
        ).fetchall()
        completed: set[tuple[str, str]] = set()
        for agent_name, model_name, n in rows:
            if n >= total:
                base = agent_name.split("__")[0]
                completed.add((base, model_name or ""))
        return completed
    except Exception:
        return set()


def _register_default_agents(models: list[str] | None = None) -> None:
    from eval.eval_loop import register_agent
    from agents_uam.single_web import SingleWebAgent
    from agents_uam.rag_claim_decomp import ClaimDecompRAGAgent

    model_list = models or [None]  # None = use global MODEL from .env
    for model_override in model_list:
        register_agent(SingleWebAgent(model_override=model_override))         # uam_ga2  tier 1
        register_agent(ClaimDecompRAGAgent(model_override=model_override))    # uam_ga4  tier 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Punkt wejścia — ewaluacja na benchmarku AM."""
    parser = argparse.ArgumentParser(
        description="Ewaluacja agentów na benchmarku AM (am_benchmark.db).",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help=(
            "Nazwy agentów oddzielone przecinkami (np. uam_ga1,uam_ga2). "
            "Domyślnie: wszyscy zarejestrowani."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maksymalna liczba twierdzeń do ewaluacji (debugging).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Wyczyść poprzednie wyniki przed uruchomieniem.",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "cloud", "sequential"],
        default="local",
        help="Tryb ewaluacji: local (tiered), cloud (parallel), sequential (default).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Liczba równoległych workerów (tylko --mode cloud).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Nazwy modeli oddzielone przecinkami. Tworzy wariant każdego agenta "
            "dla każdego modelu. Np. --models bielik-11b,llama3.1:8b. "
            "Domyślnie: globalny LLM_MODEL z .env."
        ),
    )
    parser.add_argument(
        "--tier2-limit",
        type=int,
        default=20000,
        help="Maks. liczba claimów dla agentów tier 2 (tylko --mode local).",
    )
    parser.add_argument(
        "--tier3-limit",
        type=int,
        default=20000,
        help="Maks. liczba claimów dla agentów tier 3 (tylko --mode local).",
    )
    args = parser.parse_args()

    from eval.eval_loop import (
        eval_benchmark,
        eval_benchmark_cloud,
        eval_benchmark_local,
        get_registered_agents,
        monitoring,
    )

    # Rejestracja agentów
    models_list = [m.strip() for m in args.models.split(",")] if args.models else None
    _register_default_agents(models=models_list)

    agents = get_registered_agents()

    if args.agents:
        selected_names = {n.strip() for n in args.agents.split(",")}
        agents = [a for a in agents if a.name in selected_names]

    # Skip agents that are already 100% complete (unless --clear wipes results).
    # Matches by (base_name, model_name) so runs stored with --models suffix
    # are still recognised when re-running without --models (and vice-versa).
    if not args.clear:
        completed_pairs = _get_completed_pairs()
        if completed_pairs:
            def _is_done(a) -> bool:
                base = a.name.split("__")[0]
                return (base, a.model_name or "") in completed_pairs
            skipped = [a.name for a in agents if _is_done(a)]
            agents = [a for a in agents if not _is_done(a)]
            if skipped:
                log.info("Pominięto ukończonych agentów (%d): %s", len(skipped), ", ".join(sorted(skipped)))

    if not agents:
        log.warning(
            "Brak zarejestrowanych agentów dla benchmarku AM.\n"
            "  Dodaj agentów w scripts/run_eval_am_benchmark.py → _register_default_agents()."
        )
        return

    log.info("Benchmark: %s", BENCHMARK_NAME)
    log.info("Input DB:  %s", INPUT_DB_PATH)
    log.info("Output DB: %s", RESULTS_DB_PATH)
    log.info("Agenci:    %s", ", ".join(a.name for a in agents))

    monitoring.start()
    try:
        if args.mode == "local":
            eval_benchmark_local(
                benchmark_name=BENCHMARK_NAME,
                input_db_path=INPUT_DB_PATH,
                results_db_path=RESULTS_DB_PATH,
                agents=agents,
                limit=args.limit,
                clear=args.clear,
                tier2_limit=args.tier2_limit,
                tier3_limit=args.tier3_limit,
            )
        elif args.mode == "cloud":
            eval_benchmark_cloud(
                benchmark_name=BENCHMARK_NAME,
                input_db_path=INPUT_DB_PATH,
                results_db_path=RESULTS_DB_PATH,
                agents=agents,
                limit=args.limit,
                clear=args.clear,
                workers=args.workers,
            )
        else:
            eval_benchmark(
                benchmark_name=BENCHMARK_NAME,
                input_db_path=INPUT_DB_PATH,
                results_db_path=RESULTS_DB_PATH,
                agents=agents,
                limit=args.limit,
                clear=args.clear,
            )
    except Exception as exc:
        monitoring.report_crash(exc, context="run_eval_am_benchmark/main")
        raise
    finally:
        monitoring.stop()

    log.info("Ewaluacja AM Benchmark zakończona.")


if __name__ == "__main__":
    main()
