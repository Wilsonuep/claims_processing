#!/usr/bin/env python
"""Build a combined 4k-subset results DB for fair 1:1 cross-model comparison.

Four models were evaluated on the AM benchmark (agents ga1-ga7):

    Bielik-11B          full 18,820 claims
    llama3.1:8b         full 18,820 claims
    qwen2.5:7b          4,000-claim subset (seed 42)
    Llama-PLLuM-8B      4,000-claim subset (seed 42)

The two subset models share the *same* 4,000 claim ids. To compare all four
models apples-to-apples we restrict every model to the common claim ids that
every (agent x model) pair was actually evaluated on, and copy those rows into a
fresh DB together with the matching benchmark claims (topic / year / metadata).

Source DBs are opened read-only. All writes go to results/results_am_subsample.db.
Re-runnable: the output file is recreated from scratch on every run.

    python scripts/build_subsample_results_db.py
"""
from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path

from claims_processing import paths

SOURCE_RESULTS_DB = paths.RESULTS_AM_DB
BENCHMARK_DB = paths.AM_BENCHMARK_DB
SUBSET_DB = paths.AM_BENCHMARK_4K_DB
OUTPUT_DB = paths.RESULTS_AM_SUBSAMPLE_DB

BENCHMARK_NAME = "am_benchmark"
BASE_AGENTS = [f"uam_ga{i}" for i in range(1, 8)]

# model_name -> short label (mirrors _short_model() used in the notebooks)
MODELS = {
    "hf.co/speakleash/Bielik-11B-v2.3-Instruct-GGUF:Q4_K_M": "Bielik-11B",
    "llama3.1:8b": "llama3.1:8b",
    "qwen2.5:7b": "qwen2.5:7b",
    "hf.co/mradermacher/Llama-PLLuM-8B-instruct-GGUF:Q4_K_M": "PLLuM-8B",
}

# Canonical schema, copied verbatim from eval/eval_loop.py (incl. model_name).
# Do NOT use the older schema in scripts/merge_results.py - it lacks model_name.
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
    model_name        TEXT    NOT NULL DEFAULT '',
    created_at        TEXT    NOT NULL
);
"""
_CREATE_IDX_AGENT_NAME = (
    "CREATE INDEX IF NOT EXISTS idx_agent_results_agent_name "
    "ON agent_results(agent_name);"
)
_CREATE_IDX_CLAIM_ID = (
    "CREATE INDEX IF NOT EXISTS idx_agent_results_claim_id "
    "ON agent_results(claim_id);"
)
_CREATE_UNIQUE_IDX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_results_unique_run "
    "ON agent_results(agent_name, claim_id, benchmark_name, model_name);"
)
_CREATE_CLAIMS_SQL = """\
CREATE TABLE IF NOT EXISTS claims (
    id              INTEGER PRIMARY KEY,
    claim_text      TEXT,
    topic           TEXT,
    claim_date      TEXT,
    label_original  TEXT,
    exam_type       TEXT,
    metadata        TEXT
);
"""

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_subsample")


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _claim_ids(conn: sqlite3.Connection, base_agent: str, model_name: str) -> set[int]:
    rows = conn.execute(
        "SELECT DISTINCT claim_id FROM agent_results "
        "WHERE agent_name LIKE ? AND model_name = ?",
        (base_agent + "__%", model_name),
    ).fetchall()
    return {r[0] for r in rows}


def main() -> int:
    for p in (SOURCE_RESULTS_DB, BENCHMARK_DB, SUBSET_DB):
        if not p.exists():
            log.error("Missing required DB: %s", p)
            return 1

    src = _ro(SOURCE_RESULTS_DB)

    # --- 1. Build per-(agent, model) id sets and the global intersection ----
    log.info("Coverage (distinct claim_id per agent x model):")
    header = "  agent     " + "".join(f"{lbl:>14}" for lbl in MODELS.values())
    log.info(header)

    id_sets: dict[tuple[str, str], set[int]] = {}
    for ga in BASE_AGENTS:
        counts = []
        for mn in MODELS:
            s = _claim_ids(src, ga, mn)
            id_sets[(ga, mn)] = s
            counts.append(len(s))
        log.info("  %-8s" + "".join(f"{c:>14,}" for c in counts), ga)

    common: set[int] = set.intersection(*id_sets.values())
    log.info("\nGlobal intersection across all agent x model pairs: %d ids", len(common))

    subset_ids = {r[0] for r in _ro(SUBSET_DB).execute("SELECT id FROM claims")}
    log.info("Canonical 4k-subset DB (am_benchmark_4k.db) ids: %d", len(subset_ids))
    if not common <= subset_ids:
        log.warning(
            "%d common ids are NOT in am_benchmark_4k.db (unexpected)",
            len(common - subset_ids),
        )
    if not common:
        log.error("Empty intersection - cannot build a 1:1 comparison DB.")
        return 1

    # --- 2. Crosscheck: every (agent, model) must cover all common ids ------
    ok = True
    for (ga, mn), s in id_sets.items():
        missing = common - s
        if missing:
            ok = False
            log.error("  %s / %s missing %d common ids", ga, MODELS[mn], len(missing))
    if not ok:
        log.error("Crosscheck failed - some pairs do not cover the common set.")
        return 1
    log.info("Crosscheck OK: all %d pairs cover the %d common ids.",
             len(id_sets), len(common))

    # --- 3. Create the output DB from scratch ------------------------------
    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()
    out = sqlite3.connect(OUTPUT_DB)
    out.executescript(
        _CREATE_AGENT_RESULTS_SQL
        + _CREATE_IDX_AGENT_NAME
        + _CREATE_IDX_CLAIM_ID
        + _CREATE_UNIQUE_IDX
        + _CREATE_CLAIMS_SQL
    )

    # --- 4. Copy result rows (INSERT OR IGNORE; common ids; known models) --
    common_list = sorted(common)
    id_ph = ",".join("?" * len(common_list))
    mn_ph = ",".join("?" * len(MODELS))
    cols = (
        "agent_name, claim_id, benchmark_name, original_label, model_label, "
        "is_correct, total_tokens, prompt_tokens, completion_tokens, "
        "time_thought, raw_output, model_name, created_at"
    )
    rows = src.execute(
        f"SELECT {cols} FROM agent_results "
        f"WHERE benchmark_name = ? AND claim_id IN ({id_ph}) "
        f"AND model_name IN ({mn_ph})",
        [BENCHMARK_NAME, *common_list, *MODELS.keys()],
    ).fetchall()
    out.executemany(
        f"INSERT OR IGNORE INTO agent_results ({cols}) "
        f"VALUES ({','.join('?' * 13)})",
        rows,
    )
    out.commit()
    log.info("\nCopied %d result rows.", len(rows))

    # --- 5. Copy matching benchmark claims (self-contained notebook) -------
    bench = _ro(BENCHMARK_DB)
    claim_rows = bench.execute(
        f"SELECT id, claim_text, topic, claim_date, label_original, metadata "
        f"FROM claims WHERE id IN ({id_ph})",
        common_list,
    ).fetchall()
    enriched = []
    for cid, text, topic, date, lbl, meta in claim_rows:
        exam_type = None
        if meta:
            try:
                exam_type = json.loads(meta).get("exam_type")
            except (ValueError, TypeError):
                pass
        enriched.append((cid, text, topic, date, lbl, exam_type, meta))
    out.executemany(
        "INSERT OR IGNORE INTO claims "
        "(id, claim_text, topic, claim_date, label_original, exam_type, metadata) "
        "VALUES (?,?,?,?,?,?,?)",
        enriched,
    )
    out.commit()
    log.info("Copied %d claim rows.", len(enriched))

    # --- 6. Summary + assertions -------------------------------------------
    log.info("\nFinal row counts per (agent, model):")
    bad = False
    for (an, mn), n, dc in [
        ((r[0], r[1]), r[2], r[3])
        for r in out.execute(
            "SELECT agent_name, model_name, COUNT(*), COUNT(DISTINCT claim_id) "
            "FROM agent_results GROUP BY agent_name, model_name "
            "ORDER BY agent_name, model_name"
        )
    ]:
        flag = "" if dc == len(common) else "  <-- MISMATCH"
        if flag:
            bad = True
        log.info("  %-70s %-12s  rows=%-6d claims=%-6d%s",
                 an, MODELS.get(mn, mn), n, dc, flag)

    n_claims = out.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    log.info("\nclaims table: %d rows (expected %d)", n_claims, len(common))
    out.close()

    if bad or n_claims != len(common):
        log.error("Validation failed - see mismatches above.")
        return 1
    log.info("\nOK -> %s", OUTPUT_DB)
    return 0


if __name__ == "__main__":
    sys.exit(main())
