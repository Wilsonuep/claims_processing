"""
Generyczna pętla ewaluacyjna (eval_loop)
=========================================

Ewaluuje dowolne agenty implementujące ``BaseAgent`` na benchmarkach
ładowanych z baz SQLite w ``dataprep/``.  Wyniki zapisuje do osobnych baz
w katalogu ``results/``.

Tryby wykonania
----------------
    --mode cloud    Równoległa ewaluacja (ThreadPoolExecutor, --workers N).
                    Idealna dla Together.ai / vLLM z rate-limitami.

    --mode local    Tiered scheduling — grupuje agentów wg kosztu
                    i automatycznie limituje liczbę claimów dla
                    drogich agentów (--tier2-limit, --tier3-limit).

Użycie
------
    # Cloud: wszystkie agenty, 10 równoległych workers
    python -m eval.eval_loop --mode cloud --workers 10

    # Local: tiered scheduling (fast agents full, expensive sampled)
    python -m eval.eval_loop --mode local --tier2-limit 2000 --tier3-limit 500

    # Wybrane agenty + benchmark + limit
    python -m eval.eval_loop --agents uam_ga1,uam_ga6 --benchmarks demagog --limit 100

    # Wyczyść poprzednie wyniki
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gen_agent.base_agent import BaseAgent, validate_result

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
# Tier classification — LLM calls per claim determine scheduling tier
# ---------------------------------------------------------------------------

# Agents with ≤1 LLM calls per claim — run on full dataset
TIER1_AGENTS: frozenset[str] = frozenset({
    "uam_ga1",   # single (1 call)
    "uam_ga2",   # single_web (1 call)
    "uam_ga3",   # single_bm25 (1 call)
})

# Agents with 2 LLM calls per claim — moderate cost
TIER2_AGENTS: frozenset[str] = frozenset({
    "uam_ga4",   # rag_claim_decomp (2 calls)
    "uam_ga5",   # bm25_claim_decomp (2 calls)
})

# Agents with 4+ LLM calls per claim — expensive
TIER3_AGENTS: frozenset[str] = frozenset({
    "uam_ga6",         # fewshot_cot_rag (4-5 calls)
    "uam_ga_debate",   # debate pipeline (7-8 calls)
})

# Default limits for local mode
DEFAULT_TIER2_LIMIT: int = 2000
DEFAULT_TIER3_LIMIT: int = 500


def _get_agent_tier(agent_name: str) -> int:
    """Returns the cost tier (1, 2, or 3) for an agent."""
    if agent_name in TIER1_AGENTS:
        return 1
    if agent_name in TIER2_AGENTS:
        return 2
    if agent_name in TIER3_AGENTS:
        return 3
    # Unknown agents default to tier 2 (moderate)
    return 2


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

    conn = sqlite3.connect(db_path, check_same_thread=False)
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


# ═══════════════════════════════════════════════════════════════════════════
# SEQUENTIAL EVALUATION (default, compatible with original)
# ═══════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════
# CLOUD MODE — parallel evaluation with ThreadPoolExecutor
# ═══════════════════════════════════════════════════════════════════════════


def _eval_claim_thread(
    agent: BaseAgent,
    claim: dict[str, Any],
    claim_idx: int,
    total_claims: int,
) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
    """Thread-safe evaluation of a single claim.

    Returns
    -------
    (claim_idx, claim, result_or_none)
    """
    try:
        result = eval_single(agent, claim)
        return claim_idx, claim, result
    except Exception as exc:
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
        return claim_idx, claim, error_result


def eval_benchmark_cloud(
    benchmark_name: str,
    input_db_path: str,
    results_db_path: str,
    agents: list[BaseAgent],
    *,
    limit: int | None = None,
    clear: bool = False,
    workers: int = 5,
) -> None:
    """Cloud mode: parallel evaluation with ThreadPoolExecutor.

    Sends multiple claims in parallel to a cloud API (Together.ai, vLLM)
    for maximum throughput. Each agent is run sequentially, but claims
    within each agent are parallelized.

    Parameters
    ----------
    workers : int
        Number of parallel threads per agent (default: 5).
        Together.ai standard tier supports ~600 req/min ≈ 10 req/sec.
    """
    if not agents:
        log.warning("Brak agentów — pomijam benchmark '%s'.", benchmark_name)
        return

    if not os.path.exists(input_db_path):
        log.error("Baza wejściowa nie istnieje: %s", input_db_path)
        return

    log.info(
        "═" * 60
        + "\n  [CLOUD MODE] workers=%d"
        + "\n  Benchmark: %s\n  Input DB:  %s\n  Output DB: %s\n  Agenci:    %s\n"
        + "═" * 60,
        workers,
        benchmark_name,
        input_db_path,
        results_db_path,
        ", ".join(a.name for a in agents),
    )

    input_conn = sqlite3.connect(input_db_path)
    input_conn.row_factory = sqlite3.Row

    results_conn = init_results_db(results_db_path)

    if clear:
        results_conn.execute("DELETE FROM agent_results")
        results_conn.commit()
        log.info("Wyczyszczono poprzednie wyniki w %s", results_db_path)

    query = "SELECT * FROM claims"
    if limit:
        query += f" LIMIT {limit}"

    rows = input_conn.execute(query).fetchall()
    claims = [row_to_claim_dict(row) for row in rows]
    total_claims = len(claims)
    log.info("Załadowano %d twierdzeń.", total_claims)

    import threading
    db_lock = threading.Lock()

    for agent in agents:
        log.info("─" * 40)
        log.info("Agent: %s (cloud, %d workers)", agent.name, workers)
        log.info("─" * 40)

        correct_count = 0
        error_count = 0
        total_tokens_sum = 0
        total_time_sum = 0.0
        processed = 0
        t_agent_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _eval_claim_thread, agent, claim, idx, total_claims,
                ): idx
                for idx, claim in enumerate(claims, start=1)
            }

            for future in as_completed(futures):
                idx, claim, result = future.result()
                processed += 1
                claim_id = claim["id"]

                with db_lock:
                    insert_result(
                        results_conn, agent.name, claim_id,
                        benchmark_name, result,
                    )
                    if processed % 50 == 0 or processed == total_claims:
                        results_conn.commit()

                if result["model_label"] == "ERROR":
                    error_count += 1
                else:
                    if result["is_correct"]:
                        correct_count += 1
                    total_tokens_sum += int(result["total_tokens"])
                    total_time_sum += float(result["time_thought"])

                # Log progress every 50 claims
                if processed % 50 == 0 or processed == total_claims:
                    accuracy_pct = correct_count / max(processed - error_count, 1) * 100
                    elapsed = time.perf_counter() - t_agent_start
                    claims_per_sec = processed / max(elapsed, 0.1)
                    eta_sec = (total_claims - processed) / max(claims_per_sec, 0.01)
                    log.info(
                        "[%d/%d] agent=%s | trafność=%.1f%% | "
                        "%.1f claims/s | ETA=%.0fs",
                        processed, total_claims, agent.name,
                        accuracy_pct, claims_per_sec, eta_sec,
                    )

        # Final commit
        results_conn.commit()

        agent_elapsed = time.perf_counter() - t_agent_start
        log.info("═" * 60)
        log.info("Agent: %s — podsumowanie (cloud)", agent.name)
        log.info("  Twierdzenia:  %d", total_claims)
        log.info("  Poprawne:     %d (%.1f%%)", correct_count,
                 correct_count / max(total_claims, 1) * 100)
        log.info("  Błędy:        %d", error_count)
        log.info("  Tokeny łącz.: %d", total_tokens_sum)
        log.info("  Wall-clock:   %.1f s (%.1f claims/s)",
                 agent_elapsed, total_claims / max(agent_elapsed, 0.1))
        log.info("═" * 60)

    input_conn.close()
    results_conn.close()
    log.info("Ewaluacja benchmarku '%s' zakończona (cloud mode).", benchmark_name)


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL MODE — tiered scheduling
# ═══════════════════════════════════════════════════════════════════════════


def eval_benchmark_local(
    benchmark_name: str,
    input_db_path: str,
    results_db_path: str,
    agents: list[BaseAgent],
    *,
    limit: int | None = None,
    clear: bool = False,
    tier2_limit: int = DEFAULT_TIER2_LIMIT,
    tier3_limit: int = DEFAULT_TIER3_LIMIT,
) -> None:
    """Local mode: tiered scheduling for self-hosted models.

    Groups agents by cost tier and applies per-tier claim limits:
    - Tier 1 (1 LLM call): full dataset
    - Tier 2 (2 LLM calls): tier2_limit claims
    - Tier 3 (4+ LLM calls): tier3_limit claims

    This prevents prohibitively long runtimes when using slow local
    inference (e.g. 30-42 tokens/sec on consumer GPUs).

    Parameters
    ----------
    tier2_limit : int
        Max claims for tier 2 agents (default: 2000).
    tier3_limit : int
        Max claims for tier 3 agents (default: 500).
    """
    if not agents:
        log.warning("Brak agentów — pomijam benchmark '%s'.", benchmark_name)
        return

    if not os.path.exists(input_db_path):
        log.error("Baza wejściowa nie istnieje: %s", input_db_path)
        return

    # Group agents by tier
    tier_groups: dict[int, list[BaseAgent]] = {1: [], 2: [], 3: []}
    for agent in agents:
        tier = _get_agent_tier(agent.name)
        tier_groups[tier].append(agent)

    tier_limits = {
        1: limit,           # full dataset (or --limit override)
        2: min(tier2_limit, limit) if limit else tier2_limit,
        3: min(tier3_limit, limit) if limit else tier3_limit,
    }

    log.info(
        "═" * 60
        + "\n  [LOCAL MODE] Tiered scheduling"
        + "\n  Benchmark: %s"
        + "\n  Tier 1 (fast):      %s → %s claims"
        + "\n  Tier 2 (moderate):  %s → %d claims"
        + "\n  Tier 3 (expensive): %s → %d claims\n"
        + "═" * 60,
        benchmark_name,
        [a.name for a in tier_groups[1]] or "(none)",
        "ALL" if tier_limits[1] is None else tier_limits[1],
        [a.name for a in tier_groups[2]] or "(none)",
        tier_limits[2],
        [a.name for a in tier_groups[3]] or "(none)",
        tier_limits[3],
    )

    # Load all claims once
    input_conn = sqlite3.connect(input_db_path)
    input_conn.row_factory = sqlite3.Row

    all_rows = input_conn.execute("SELECT * FROM claims").fetchall()
    total_available = len(all_rows)
    log.info("Załadowano %d twierdzeń z bazy.", total_available)

    results_conn = init_results_db(results_db_path)

    if clear:
        results_conn.execute("DELETE FROM agent_results")
        results_conn.commit()
        log.info("Wyczyszczono poprzednie wyniki.")

    # Execute tiers in order (fast first → expensive last)
    for tier_num in (1, 2, 3):
        tier_agents = tier_groups[tier_num]
        if not tier_agents:
            continue

        tier_lim = tier_limits[tier_num]
        if tier_lim is not None:
            rows = all_rows[:tier_lim]
        else:
            rows = all_rows

        total_claims = len(rows)

        log.info(
            "━" * 50
            + "\n  TIER %d: %d agents × %d claims"
            + "\n━" * 50,
            tier_num,
            len(tier_agents),
            total_claims,
        )

        for agent in tier_agents:
            log.info("─" * 40)
            log.info(
                "Agent: %s (tier %d, %d claims)", agent.name, tier_num, total_claims,
            )
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
                        idx, total_claims, claim_id, agent.name, exc,
                    )
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
                        results_conn, agent.name, claim_id,
                        benchmark_name, error_result,
                    )
                    results_conn.commit()
                    continue

                insert_result(
                    results_conn, agent.name, claim_id, benchmark_name, result,
                )
                results_conn.commit()

                if result["is_correct"]:
                    correct_count += 1
                total_tokens_sum += int(result["total_tokens"])
                total_time_sum += float(result["time_thought"])

                accuracy_pct = correct_count / idx * 100
                log.info(
                    "[%d/%d] claim_id=%-6s | poprawna=%-5s | "
                    "tokeny=%d | czas=%.2fs | trafność=%.1f%% | %s…",
                    idx, total_claims, claim_id,
                    "TAK" if result["is_correct"] else "NIE",
                    int(result["total_tokens"]),
                    float(result["time_thought"]),
                    accuracy_pct,
                    claim_text_preview,
                )

            # Agent summary
            log.info("═" * 60)
            log.info("Agent: %s — podsumowanie (tier %d)", agent.name, tier_num)
            log.info("  Twierdzenia:  %d / %d", total_claims, total_available)
            log.info("  Poprawne:     %d (%.1f%%)", correct_count,
                     correct_count / max(total_claims, 1) * 100)
            log.info("  Błędy:        %d", error_count)
            log.info("  Tokeny łącz.: %d", total_tokens_sum)
            log.info("  Czas łącz.:   %.1f s", total_time_sum)
            log.info("═" * 60)

    input_conn.close()
    results_conn.close()
    log.info("Ewaluacja benchmarku '%s' zakończona (local mode).", benchmark_name)


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
        description=(
            "Generyczna pętla ewaluacyjna — ewaluuje agentów "
            "na benchmarkach fact-checking."
        ),
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

    # --- Mode selection ---
    parser.add_argument(
        "--mode",
        choices=["sequential", "cloud", "local"],
        default="sequential",
        help=(
            "Tryb ewaluacji: "
            "'sequential' — domyślny, po kolei; "
            "'cloud' — równoległa ewaluacja (ThreadPoolExecutor); "
            "'local' — tiered scheduling dla self-hosted modeli."
        ),
    )

    # --- Cloud mode options ---
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Liczba równoległych workerów (tryb cloud). Default: 5.",
    )

    # --- Local mode options ---
    parser.add_argument(
        "--tier2-limit",
        type=int,
        default=DEFAULT_TIER2_LIMIT,
        help=(
            f"Limit claimów dla agentów tier 2 w trybie local. "
            f"Default: {DEFAULT_TIER2_LIMIT}."
        ),
    )
    parser.add_argument(
        "--tier3-limit",
        type=int,
        default=DEFAULT_TIER3_LIMIT,
        help=(
            f"Limit claimów dla agentów tier 3 w trybie local. "
            f"Default: {DEFAULT_TIER3_LIMIT}."
        ),
    )

    args = parser.parse_args()

    # --- LLM backend info ---
    try:
        from gen_agent.llm_client import BACKEND_INFO
        log.info(
            "LLM Config: backend=%s, model=%s, is_local=%s",
            BACKEND_INFO["backend"],
            BACKEND_INFO["model"],
            BACKEND_INFO["is_local"],
        )
    except ImportError:
        pass

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
    log.info("Tryb: %s", args.mode)

    # Ewaluacja
    for bname in benchmark_names:
        input_db, results_db = _resolve_db_paths(bname)

        if args.mode == "cloud":
            eval_benchmark_cloud(
                benchmark_name=bname,
                input_db_path=input_db,
                results_db_path=results_db,
                agents=agents,
                limit=args.limit,
                clear=args.clear,
                workers=args.workers,
            )
        elif args.mode == "local":
            eval_benchmark_local(
                benchmark_name=bname,
                input_db_path=input_db,
                results_db_path=results_db,
                agents=agents,
                limit=args.limit,
                clear=args.clear,
                tier2_limit=args.tier2_limit,
                tier3_limit=args.tier3_limit,
            )
        else:  # sequential (default, backward compatible)
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
