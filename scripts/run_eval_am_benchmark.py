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
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_NAME = "am_benchmark"
INPUT_DB_PATH = str(PROJECT_ROOT / "dataprep" / "am_benchmark.db")
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


def _register_default_agents() -> None:
    """Rejestruje domyślnych agentów dla benchmarku AM.

    Dodaj tutaj importy i rejestracje agentów dedykowanych
    dla AM benchmark, np.:

        from eval.eval_loop import register_agent
        from agents_uam.some_agent import SomeAgent
        register_agent(SomeAgent())
    """
    # TODO: Zarejestruj agentów, gdy będą gotowi.
    pass


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
    args = parser.parse_args()

    from eval.eval_loop import eval_benchmark, get_registered_agents

    # Rejestracja agentów
    _register_default_agents()

    agents = get_registered_agents()

    if args.agents:
        selected_names = {n.strip() for n in args.agents.split(",")}
        agents = [a for a in agents if a.name in selected_names]

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

    eval_benchmark(
        benchmark_name=BENCHMARK_NAME,
        input_db_path=INPUT_DB_PATH,
        results_db_path=RESULTS_DB_PATH,
        agents=agents,
        limit=args.limit,
        clear=args.clear,
    )

    log.info("Ewaluacja AM Benchmark zakończona.")


if __name__ == "__main__":
    main()
