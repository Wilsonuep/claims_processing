"""
Generyczna pętla ewaluacyjna (eval_loop)
=========================================

Ewaluuje dowolne agenty implementujące ``BaseAgent`` na benchmarkach
ładowanych z baz SQLite w ``dataprep/``.  Wyniki zapisuje do osobnych baz
w katalogu ``results/``.

Użycie
------
    # Ewaluacja wszystkich agentów na wszystkich benchmarkach
    python -m eval.eval_loop

    # Tylko wybrany benchmark
    python -m eval.eval_loop --benchmarks demagog

    # Tylko wybrani agenci
    python -m eval.eval_loop --agents uam_ga1

    # Limit rekordów (debugging)
    python -m eval.eval_loop --limit 10

    # Wyczyść poprzednie wyniki przed uruchomieniem
    python -m eval.eval_loop --clear

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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.base_agent import BaseAgent, validate_result

# ---------------------------------------------------------------------------
# Konfiguracja logowania
# ---------------------------------------------------------------------------

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
# Konfiguracja benchmarków
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARKS: list[dict[str, str]] = [
    {"name": "demagog", "input_db": "dataprep/demagog.db"},
    {"name": "am_benchmark", "input_db": "dataprep/am_benchmark.db"},
]

RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# Rejestr agentów
# ---------------------------------------------------------------------------

_AGENT_REGISTRY: list[BaseAgent] = []


def register_agent(agent: BaseAgent) -> None:
    """Rejestruje agenta w globalnym rejestrze.

    Parametry
    ---------
    agent : BaseAgent
        Instancja agenta do zarejestrowania.
    """
    if not isinstance(agent, BaseAgent):
        raise TypeError(
            f"Agent musi dziedziczyć po BaseAgent, otrzymano: {type(agent).__name__}"
        )
    if not hasattr(agent, "name") or not agent.name:
        raise ValueError("Agent musi mieć ustawiony atrybut 'name'.")
    _AGENT_REGISTRY.append(agent)
    log.info("Zarejestrowano agenta: %s", agent.name)


def get_registered_agents() -> list[BaseAgent]:
    """Zwraca listę wszystkich zarejestrowanych agentów."""
    return list(_AGENT_REGISTRY)


def clear_registry() -> None:
    """Czyści rejestr agentów."""
    _AGENT_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Schemat bazy wyników
# ---------------------------------------------------------------------------

_CREATE_AGENT_RESULTS_SQL = """\
CREATE TABLE IF NOT EXISTS agent_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name        TEXT    NOT NULL,
    claim_id          INTEGER NOT NULL,
    benchmark_name    TEXT    NOT NULL,
    original_label    TEXT    NOT NULL,
    model_label       TEXT    NOT NULL,
    is_correct        INTEGER NOT NULL,
    total_tokens      INTEGER NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    time_thought      REAL    NOT NULL,
    raw_output        TEXT,
    created_at        TEXT    NOT NULL
);
"""

_CREATE_IDX_AGENT_NAME = """\
CREATE INDEX IF NOT EXISTS idx_agent_results_agent_name
    ON agent_results(agent_name);
"""

_CREATE_IDX_CLAIM_ID = """\
CREATE INDEX IF NOT EXISTS idx_agent_results_claim_id
    ON agent_results(claim_id);
"""


# ---------------------------------------------------------------------------
# Inicjalizacja bazy wyników
# ---------------------------------------------------------------------------


def init_results_db(db_path: str) -> sqlite3.Connection:
    """Tworzy (lub otwiera) bazę wyników i zakłada tabelę ``agent_results``.

    Parametry
    ---------
    db_path : str
        Ścieżka do pliku ``.db`` (zostanie utworzony, jeśli nie istnieje).

    Zwraca
    ------
    sqlite3.Connection
        Otwarte połączenie.
    """
    # Upewnij się, że katalog istnieje
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    cur = conn.cursor()
    cur.execute(_CREATE_AGENT_RESULTS_SQL)
    cur.execute(_CREATE_IDX_AGENT_NAME)
    cur.execute(_CREATE_IDX_CLAIM_ID)
    conn.commit()

    log.info("Baza wyników gotowa: %s", db_path)
    return conn


# ---------------------------------------------------------------------------
# Pomocnicze — konwersja wierszy DB → dict
# ---------------------------------------------------------------------------


def row_to_claim_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Konwertuje wiersz SQLite na słownik claim.

    Mapuje klucze tak, aby wynikowy słownik zawierał co najmniej:
    ``id``, ``claim_text``, ``label`` (ground truth).

    Parametry
    ---------
    row : sqlite3.Row
        Wiersz z tabeli ``claims``.

    Zwraca
    ------
    dict
        Słownik z danymi twierdzenia.
    """
    return dict(row)


# ---------------------------------------------------------------------------
# Wstawianie wyniku do bazy
# ---------------------------------------------------------------------------


def insert_result(
    conn: sqlite3.Connection,
    agent_name: str,
    claim_id: int,
    benchmark_name: str,
    result: dict[str, Any],
) -> None:
    """Wstawia pojedynczy wynik ewaluacji do tabeli ``agent_results``.

    Parametry
    ---------
    conn : sqlite3.Connection
        Połączenie z bazą wyników.
    agent_name : str
        Nazwa agenta.
    claim_id : int
        Identyfikator twierdzenia z tabeli ``claims``.
    benchmark_name : str
        Nazwa benchmarku (np. ``'demagog'``, ``'am_benchmark'``).
    result : dict
        Wynik zwrócony przez ``agent.eval()``.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT INTO agent_results
            (agent_name, claim_id, benchmark_name, original_label,
             model_label, is_correct, total_tokens, prompt_tokens,
             completion_tokens, time_thought, raw_output, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            agent_name,
            claim_id,
            benchmark_name,
            result["original_label"],
            result["model_label"],
            int(bool(result["is_correct"])),
            int(result["total_tokens"]),
            int(result["prompt_tokens"]),
            int(result["completion_tokens"]),
            float(result["time_thought"]),
            result.get("raw_output", ""),
            now_iso,
        ),
    )


# ---------------------------------------------------------------------------
# Ewaluacja pojedynczego agenta na jednym twierdzeniu
# ---------------------------------------------------------------------------


def eval_single(
    agent: BaseAgent,
    claim: dict[str, Any],
) -> dict[str, Any]:
    """Wywołuje ``agent.eval(claim)`` i waliduje rezultat.

    Parametry
    ---------
    agent : BaseAgent
        Instancja agenta.
    claim : dict
        Słownik twierdzenia.

    Zwraca
    ------
    dict
        Zwalidowany wynik agenta.

    Rzuca
    -----
    ValueError
        Gdy wynik nie zawiera wymaganych kluczy.
    """
    result = agent.eval(claim)
    validate_result(result, agent.name)
    return result


# ---------------------------------------------------------------------------
# Ewaluacja jednego benchmarku
# ---------------------------------------------------------------------------


def eval_benchmark(
    benchmark_name: str,
    input_db_path: str,
    results_db_path: str,
    agents: list[BaseAgent],
    *,
    limit: int | None = None,
    clear: bool = False,
) -> None:
    """Ewaluuje wszystkich agentów na wszystkich twierdzeniach z benchmarku.

    Parametry
    ---------
    benchmark_name : str
        Nazwa benchmarku (np. ``'demagog'``, ``'am_benchmark'``).
    input_db_path : str
        Ścieżka do bazy wejściowej z tabelą ``claims``.
    results_db_path : str
        Ścieżka do bazy wynikowej.
    agents : list[BaseAgent]
        Lista agentów do ewaluacji.
    limit : int | None
        Opcjonalny limit rekordów (do debugowania).
    clear : bool
        Czy wyczyścić poprzednie wyniki przed uruchomieniem.
    """
    if not agents:
        log.warning("Brak agentów do ewaluacji — pomijam benchmark '%s'.", benchmark_name)
        return

    if not os.path.exists(input_db_path):
        log.error("Baza wejściowa nie istnieje: %s", input_db_path)
        return

    log.info(
        "═" * 60
        + "\n  Benchmark: %s\n  Input DB:  %s\n  Output DB: %s\n  Agenci:    %s\n"
        + "═" * 60,
        benchmark_name,
        input_db_path,
        results_db_path,
        ", ".join(a.name for a in agents),
    )

    # --- Połączenia ---
    input_conn = sqlite3.connect(input_db_path)
    input_conn.row_factory = sqlite3.Row

    results_conn = init_results_db(results_db_path)

    if clear:
        results_conn.execute("DELETE FROM agent_results")
        results_conn.commit()
        log.info("Wyczyszczono poprzednie wyniki w %s", results_db_path)

    # --- Pobranie twierdzeń ---
    query = "SELECT * FROM claims"
    if limit:
        query += f" LIMIT {limit}"

    rows = input_conn.execute(query).fetchall()
    total_claims = len(rows)
    log.info("Załadowano %d twierdzeń z tabeli 'claims'.", total_claims)

    # --- Pętla ewaluacyjna ---
    for agent in agents:
        log.info("─" * 40)
        log.info("Agent: %s", agent.name)
        log.info("─" * 40)

        correct_count = 0
        error_count = 0
        total_tokens_sum = 0
        total_time_sum = 0.0

        for idx, row in enumerate(rows, start=1):
            claim = row_to_claim_dict(row)
            claim_id = claim["id"]
            claim_text_preview = (claim.get("claim_text") or "")[:80]

            try:
                result = eval_single(agent, claim)
            except Exception as exc:
                error_count += 1
                log.error(
                    "[%d/%d] BŁĄD — claim_id=%s agent=%s: %s",
                    idx,
                    total_claims,
                    claim_id,
                    agent.name,
                    exc,
                )
                # Wstawiamy wiersz z informacją o błędzie
                error_result: dict[str, Any] = {
                    "model_label": "ERROR",
                    "original_label": claim.get("label", ""),
                    "is_correct": False,
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "time_thought": 0.0,
                    "raw_output": f"ERROR: {exc}",
                }
                insert_result(
                    results_conn, agent.name, claim_id, benchmark_name, error_result
                )
                results_conn.commit()
                continue

            # Wstawienie wyniku
            insert_result(
                results_conn, agent.name, claim_id, benchmark_name, result
            )
            results_conn.commit()

            # Aktualizacja statystyk
            if result["is_correct"]:
                correct_count += 1
            total_tokens_sum += int(result["total_tokens"])
            total_time_sum += float(result["time_thought"])

            # Postęp
            accuracy_pct = correct_count / idx * 100
            log.info(
                "[%d/%d] claim_id=%-6s | poprawna=%-5s | "
                "tokeny=%d | czas=%.2fs | trafność=%.1f%% | %s…",
                idx,
                total_claims,
                claim_id,
                "TAK" if result["is_correct"] else "NIE",
                int(result["total_tokens"]),
                float(result["time_thought"]),
                accuracy_pct,
                claim_text_preview,
            )

        # --- Podsumowanie agenta ---
        log.info("═" * 60)
        log.info("Agent: %s — podsumowanie", agent.name)
        log.info("  Twierdzenia:  %d", total_claims)
        log.info("  Poprawne:     %d (%.1f%%)", correct_count,
                 correct_count / max(total_claims, 1) * 100)
        log.info("  Błędy:        %d", error_count)
        log.info("  Tokeny łącz.: %d", total_tokens_sum)
        log.info("  Czas łącz.:   %.1f s", total_time_sum)
        log.info("═" * 60)

    # --- Zamknięcie połączeń ---
    input_conn.close()
    results_conn.close()
    log.info("Ewaluacja benchmarku '%s' zakończona.", benchmark_name)


# ---------------------------------------------------------------------------
# CLI — uruchomienie wszystkich benchmarków
# ---------------------------------------------------------------------------


def _resolve_db_paths(
    benchmark_name: str,
) -> tuple[str, str]:
    """Zwraca ścieżki input_db i results_db dla danego benchmarku.

    Parametry
    ---------
    benchmark_name : str
        Nazwa benchmarku (np. ``'demagog'``).

    Zwraca
    ------
    (input_db_path, results_db_path)
    """
    # Szukamy w konfiguracji
    for b in BENCHMARKS:
        if b["name"] == benchmark_name:
            input_db = str(PROJECT_ROOT / b["input_db"])
            results_db = str(
                PROJECT_ROOT / RESULTS_DIR / f"results_{benchmark_name}.db"
            )
            return input_db, results_db

    # Fallback
    input_db = str(PROJECT_ROOT / "dataprep" / f"{benchmark_name}.db")
    results_db = str(PROJECT_ROOT / RESULTS_DIR / f"results_{benchmark_name}.db")
    return input_db, results_db


def main() -> None:
    """Punkt wejścia CLI — ewaluuje agentów na benchmarkach."""
    parser = argparse.ArgumentParser(
        description="Generyczna pętla ewaluacyjna — ewaluuje agentów na benchmarkach fact-checking.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help=(
            "Lista benchmarków do ewaluacji (np. demagog am_benchmark). "
            "Domyślnie: wszystkie zarejestrowane."
        ),
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

    # Wybór benchmarków
    benchmark_names = args.benchmarks or [b["name"] for b in BENCHMARKS]

    # Wybór agentów
    agents = get_registered_agents()

    if not agents:
        log.warning(
            "Brak zarejestrowanych agentów. Zarejestruj agentów "
            "za pomocą register_agent() przed uruchomieniem."
        )
        log.info(
            "Wskazówka: Zaimportuj moduły agentów lub użyj skryptów "
            "run_eval_demagog.py / run_eval_am_benchmark.py."
        )
        return

    if args.agents:
        selected_names = {n.strip() for n in args.agents.split(",")}
        agents = [a for a in agents if a.name in selected_names]
        if not agents:
            log.error(
                "Żaden z podanych agentów nie został znaleziony: %s",
                args.agents,
            )
            return

    log.info("Agenci do ewaluacji: %s", ", ".join(a.name for a in agents))
    log.info("Benchmarki: %s", ", ".join(benchmark_names))

    # Ewaluacja
    for bname in benchmark_names:
        input_db, results_db = _resolve_db_paths(bname)
        eval_benchmark(
            benchmark_name=bname,
            input_db_path=input_db,
            results_db_path=results_db,
            agents=agents,
            limit=args.limit,
            clear=args.clear,
        )

    log.info("Ewaluacja wszystkich benchmarków zakończona.")


if __name__ == "__main__":
    main()
