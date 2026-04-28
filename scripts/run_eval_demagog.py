"""
Ewaluacja agentów na benchmarku Demagog
========================================

Wrapper uruchamiający generyczną pętlę ewaluacyjną
wyłącznie na bazie ``demagog.db``.

Użycie
------
    python -m scripts.run_eval_demagog
    python -m scripts.run_eval_demagog --limit 10
    python -m scripts.run_eval_demagog --agents uam_ga1
    python -m scripts.run_eval_demagog --clear

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
BENCHMARK_NAME = "demagog"
INPUT_DB_PATH = str(PROJECT_ROOT / "dataprep" / "demagog.db")
RESULTS_DB_PATH = str(PROJECT_ROOT / "results" / "results_demagog.db")

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
# na benchmarku Demagog.
#
# Przykład:
#
#   from eval.eval_loop import register_agent
#   from agents_dem.my_agent import MyDemagogAgent
#   register_agent(MyDemagogAgent())
#
# Na razie rejestr jest pusty — dodaj agentów poniżej,
# gdy będą gotowi do ewaluacji.


def _get_completed_pairs() -> set[tuple[str, str]]:
    """Returns (base_agent_name, model_name) pairs that have a complete set of results."""
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
    from agents_dem.bm25_claim_decomp import ClaimDecompBM25Agent
    from agents_dem.fewshot_cot_rag import FewShotCoTAgent
    from agents_dem.fewshot_cot_debate_rag import DebateCoTAgent

    model_list = models or [None]  # None = use global MODEL from .env
    for model_override in model_list:
        register_agent(ClaimDecompBM25Agent(model_override=model_override))   # dem_ga5  tier 2
        register_agent(FewShotCoTAgent(model_override=model_override))        # dem_ga6  tier 3
        register_agent(DebateCoTAgent(model_override=model_override))         # dem_ga7  tier 3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Punkt wejścia — ewaluacja na benchmarku Demagog."""
    parser = argparse.ArgumentParser(
        description="Ewaluacja agentów na benchmarku Demagog (demagog.db).",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help=(
            "Nazwy agentów oddzielone przecinkami (np. agent1,agent2). "
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
            "Brak zarejestrowanych agentów dla benchmarku Demagog.\n"
            "  Dodaj agentów w scripts/run_eval_demagog.py → _register_default_agents()."
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
        monitoring.report_crash(exc, context="run_eval_demagog/main")
        raise
    finally:
        monitoring.stop()

    log.info("Ewaluacja Demagog zakończona.")


if __name__ == "__main__":
    main()
