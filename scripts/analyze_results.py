"""
Corruption analysis for results_am_benchmark.db.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --db results/results_am_benchmark.db
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

# Ensure Polish characters print correctly on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VALID_UAM = {"0", "1", "2", "3"}
VALID_DEM = {"PRAWDA", "CZĘŚCIOWA_PRAWDA", "FAŁSZ", "MANIPULACJA", "NIEWERYFIKOWALNE"}
# '4' is also used by some DEM agent variants for NIEWERYFIKOWALNE
VALID_DEM_EXTENDED = VALID_DEM | {"4"}
ERROR_LABELS = {"ERROR", "ERROR_MAX_STEPS"}


def valid_set_for_agent(agent_name: str) -> set[str]:
    if "dem" in agent_name.lower():
        return VALID_DEM_EXTENDED | VALID_UAM
    return VALID_UAM


def classify_label(label: str, agent_name: str) -> str:
    valid = valid_set_for_agent(agent_name)
    if label in valid:
        return "valid"
    if label == "ERROR_MAX_STEPS":
        return "error_max_steps"
    if label == "ERROR":
        return "error_generic"
    # Float-like: "1.0", "2.5"
    try:
        f = float(label)
        rounded = str(round(f))
        if rounded in VALID_UAM:
            return "fixable_float"
        return "corrupt_float_oob"
    except ValueError:
        pass
    # "Output: N" or similar text prefix containing a digit
    if re.search(r"\b[0-3]\b", label):
        return "fixable_prefix"
    # Looks like a JSON array
    if label.startswith("["):
        return "fixable_array"
    return "corrupt_other"


def try_mine_label_from_trajectory(raw_output: str) -> str | None:
    try:
        traj = json.loads(raw_output)
    except Exception:
        return None
    for turn in reversed(traj):
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content", "")
        # Try full ReAct JSON parse
        try:
            jm = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
            js = jm.group(1).strip() if jm else None
            if not js:
                start, end = content.find("{"), content.rfind("}")
                if start != -1 and end != -1:
                    js = content[start : end + 1]
            if js:
                parsed = json.loads(js)
                if parsed.get("action") == "final_answer":
                    lbl = parsed.get("action_input", {}).get("label")
                    if lbl is not None:
                        return str(lbl).strip()
        except Exception:
            pass
        # Fallback: first "label": "..." occurrence
        m = re.search(r'"label"\s*:\s*"([^"]+)"', content)
        if m:
            return m.group(1).strip()
    return None


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def main(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # ── 1. Overview ────────────────────────────────────────────
    print_section("DATABASE OVERVIEW")
    total = conn.execute("SELECT COUNT(*) FROM agent_results").fetchone()[0]
    print(f"Total records : {total:,}")

    rows = conn.execute(
        "SELECT agent_name, COUNT(*) AS cnt FROM agent_results GROUP BY agent_name ORDER BY agent_name"
    ).fetchall()
    print(f"\n{'Agent':<40}  {'Records':>8}")
    print("-" * 52)
    for r in rows:
        print(f"{r['agent_name']:<40}  {r['cnt']:>8,}")

    # ── 2. Label distribution ──────────────────────────────────
    print_section("LABEL DISTRIBUTION (non-standard only)")
    rows = conn.execute(
        """
        SELECT model_label, COUNT(*) AS cnt,
               GROUP_CONCAT(DISTINCT agent_name) AS agents
        FROM agent_results
        WHERE model_label NOT IN ('0','1','2','3',
            'PRAWDA','CZĘŚCIOWA_PRAWDA','FAŁSZ','MANIPULACJA','NIEWERYFIKOWALNE','4')
        GROUP BY model_label
        ORDER BY cnt DESC
        """
    ).fetchall()
    print(f"\n{'Label':<30}  {'Count':>7}  Agents")
    print("-" * 80)
    for r in rows:
        print(f"{r['model_label']:<30}  {r['cnt']:>7,}  {r['agents']}")

    # ── 3. Corruption classification ──────────────────────────
    print_section("CORRUPTION CLASSIFICATION")
    all_rows = conn.execute(
        "SELECT id, agent_name, model_label, original_label, raw_output FROM agent_results"
    ).fetchall()

    buckets: dict[str, int] = {}
    for r in all_rows:
        cls = classify_label(r["model_label"], r["agent_name"])
        buckets[cls] = buckets.get(cls, 0) + 1

    print(f"\n{'Category':<28}  {'Count':>7}  {'%':>6}")
    print("-" * 46)
    for cls in [
        "valid",
        "error_max_steps",
        "error_generic",
        "fixable_float",
        "fixable_prefix",
        "fixable_array",
        "corrupt_float_oob",
        "corrupt_other",
    ]:
        n = buckets.get(cls, 0)
        print(f"{cls:<28}  {n:>7,}  {n/total*100:>5.1f}%")

    total_corrupt = sum(
        buckets.get(c, 0)
        for c in ["error_max_steps", "error_generic", "fixable_float",
                  "fixable_prefix", "fixable_array", "corrupt_float_oob", "corrupt_other"]
    )
    total_fixable = sum(
        buckets.get(c, 0)
        for c in ["fixable_float", "fixable_prefix", "fixable_array"]
    )
    print(f"\nTotal non-standard records  : {total_corrupt:,}")
    print(f"Programmatically fixable    : {total_fixable:,}  (float/prefix/array)")

    # ── 4. ERROR_MAX_STEPS trajectory mining ──────────────────
    print_section("ERROR_MAX_STEPS — TRAJECTORY RECOVERY ESTIMATE")
    max_rows = conn.execute(
        "SELECT id, agent_name, original_label, raw_output FROM agent_results WHERE model_label='ERROR_MAX_STEPS'"
    ).fetchall()

    if not max_rows:
        print("No ERROR_MAX_STEPS records found.")
    else:
        recoverable = 0
        for r in max_rows:
            mined = try_mine_label_from_trajectory(r["raw_output"] or "")
            if mined and mined in valid_set_for_agent(r["agent_name"]):
                recoverable += 1
        total_max = len(max_rows)
        print(f"\nERROR_MAX_STEPS total       : {total_max:,}")
        print(f"Recoverable from trajectory : {recoverable:,}  ({recoverable/total_max*100:.1f}%)")
        print(f"Need re-run                 : {total_max - recoverable:,}")

    # ── 5. ERROR records (auto-deleted on next run) ────────────
    print_section("ERROR (GENERIC) RECORDS")
    err_rows = conn.execute(
        """
        SELECT agent_name, COUNT(*) AS cnt,
               SUBSTR(raw_output, 1, 120) AS sample
        FROM agent_results
        WHERE model_label = 'ERROR'
        GROUP BY agent_name
        ORDER BY cnt DESC
        """
    ).fetchall()
    if err_rows:
        print("\nThese are auto-deleted and retried when the eval loop restarts.")
        print(f"\n{'Agent':<40}  {'Count':>6}")
        print("-" * 50)
        for r in err_rows:
            print(f"{r['agent_name']:<40}  {r['cnt']:>6,}")
        sample = conn.execute(
            "SELECT raw_output FROM agent_results WHERE model_label='ERROR' LIMIT 1"
        ).fetchone()
        if sample:
            print(f"\nSample error message: {sample[0][:200]}")
    else:
        print("No ERROR records found.")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze corrupted results in the benchmark DB.")
    parser.add_argument(
        "--db",
        default="results/results_am_benchmark.db",
        help="Path to SQLite results database",
    )
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"ERROR: database not found at {db}", file=sys.stderr)
        sys.exit(1)

    main(str(db))
