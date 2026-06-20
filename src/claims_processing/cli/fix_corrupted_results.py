"""
Repair corrupted model_label values in results_am_benchmark.db.

Three fix passes (in order):
  1. Programmatic fixes — float labels, "Output: N" prefix, JSON arrays
  2. Trajectory mining — extract final_answer from ERROR_MAX_STEPS raw_output
  3. Delete unfixable ERROR / None / N/A records so eval_loop retries them

Usage:
    python scripts/fix_corrupted_results.py --dry-run   # preview changes
    python scripts/fix_corrupted_results.py             # apply changes
    python scripts/fix_corrupted_results.py --skip-mining  # skip slow trajectory pass
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

VALID_UAM = {"0", "1", "2", "3"}
VALID_DEM = {"PRAWDA", "CZĘŚCIOWA_PRAWDA", "FAŁSZ", "MANIPULACJA", "NIEWERYFIKOWALNE", "4"}


def valid_set_for_agent(agent_name: str) -> set[str]:
    if "dem" in agent_name.lower():
        return VALID_DEM | VALID_UAM
    return VALID_UAM


def fix_float_label(label: str) -> str | None:
    """'1.0' → '1', '2.5' → None (ambiguous, don't fix)."""
    try:
        f = float(label)
        rounded = str(round(f))
        if rounded in VALID_UAM and abs(f - round(f)) < 0.01:
            return rounded
    except ValueError:
        pass
    return None


def fix_prefix_label(label: str) -> str | None:
    """'Output: 2' → '2', 'Answer:1' → '1'."""
    m = re.search(r"\b([0-3])\b", label)
    return m.group(1) if m else None


def fix_array_label(label: str) -> str | None:
    """\"['1', '3']\" → '1' (take first element)."""
    try:
        parsed = json.loads(label.replace("'", '"'))
        if isinstance(parsed, list) and parsed:
            candidate = str(parsed[0]).strip()
            if candidate in VALID_UAM:
                return candidate
    except Exception:
        pass
    return None


def try_mine_label(raw_output: str) -> str | None:
    try:
        traj = json.loads(raw_output)
    except Exception:
        return None
    for turn in reversed(traj):
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content", "")
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
        m = re.search(r'"label"\s*:\s*"([^"]+)"', content)
        if m:
            return m.group(1).strip()
    return None


def main(db_path: str, dry_run: bool, skip_mining: bool) -> None:
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"\n{'='*60}")
    print(f"  fix_corrupted_results.py  [{mode}]")
    print(f"  DB: {db_path}")
    print("=" * 60)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    fixed_total = 0
    deleted_total = 0

    # ── Pass 1: Programmatic fixes ─────────────────────────────
    print("\n[Pass 1] Programmatic label fixes (float / prefix / array)")

    rows = conn.execute(
        """
        SELECT id, agent_name, model_label, original_label
        FROM agent_results
        WHERE model_label NOT IN ('0','1','2','3',
            'PRAWDA','CZĘŚCIOWA_PRAWDA','FAŁSZ','MANIPULACJA','NIEWERYFIKOWALNE','4',
            'ERROR','ERROR_MAX_STEPS')
        """
    ).fetchall()

    pass1_fixes: list[tuple[str, bool, int]] = []
    for r in rows:
        label = r["model_label"]
        new_label = (
            fix_float_label(label)
            or fix_prefix_label(label)
            or fix_array_label(label)
        )
        if new_label and new_label in valid_set_for_agent(r["agent_name"]):
            is_correct = new_label.strip() == str(r["original_label"]).strip()
            pass1_fixes.append((new_label, is_correct, r["id"]))

    print(f"  Found {len(rows)} non-standard labels, {len(pass1_fixes)} fixable")
    for new_label, is_correct, row_id in pass1_fixes[:10]:
        old = conn.execute(
            "SELECT model_label FROM agent_results WHERE id=?", (row_id,)
        ).fetchone()["model_label"]
        print(f"    id={row_id}: '{old}' → '{new_label}'  correct={is_correct}")
    if len(pass1_fixes) > 10:
        print(f"    ... and {len(pass1_fixes)-10} more")

    if not dry_run and pass1_fixes:
        conn.executemany(
            "UPDATE agent_results SET model_label=?, is_correct=? WHERE id=?",
            pass1_fixes,
        )
        conn.commit()
        print(f"  Applied {len(pass1_fixes)} fixes.")
    fixed_total += len(pass1_fixes)

    # ── Pass 2: Trajectory mining for ERROR_MAX_STEPS ─────────
    if not skip_mining:
        print("\n[Pass 2] Trajectory mining for ERROR_MAX_STEPS")
        max_rows = conn.execute(
            "SELECT id, agent_name, original_label, raw_output FROM agent_results WHERE model_label='ERROR_MAX_STEPS'"
        ).fetchall()
        print(f"  Scanning {len(max_rows):,} ERROR_MAX_STEPS records…")

        pass2_fixes: list[tuple[str, bool, int]] = []
        for r in max_rows:
            mined = try_mine_label(r["raw_output"] or "")
            if mined and mined in valid_set_for_agent(r["agent_name"]):
                is_correct = mined.strip() == str(r["original_label"]).strip()
                pass2_fixes.append((mined, is_correct, r["id"]))

        print(f"  Recoverable: {len(pass2_fixes):,} / {len(max_rows):,}")
        for new_label, is_correct, row_id in pass2_fixes[:5]:
            print(f"    id={row_id}: ERROR_MAX_STEPS → '{new_label}'  correct={is_correct}")
        if len(pass2_fixes) > 5:
            print(f"    ... and {len(pass2_fixes)-5} more")

        if not dry_run and pass2_fixes:
            conn.executemany(
                "UPDATE agent_results SET model_label=?, is_correct=? WHERE id=?",
                pass2_fixes,
            )
            conn.commit()
            print(f"  Applied {len(pass2_fixes)} trajectory fixes.")
        fixed_total += len(pass2_fixes)
    else:
        print("\n[Pass 2] Skipped (--skip-mining)")

    # ── Pass 3: Delete unfixable records for retry ────────────
    print("\n[Pass 3] Delete unfixable records so eval_loop retries them")

    unfixable_labels = ("ERROR", "None", "N/A", "-1", "")
    placeholders = ",".join("?" * len(unfixable_labels))
    unfixable_rows = conn.execute(
        f"SELECT id, agent_name, model_label FROM agent_results WHERE model_label IN ({placeholders})",
        unfixable_labels,
    ).fetchall()

    # Also flag any remaining non-standard labels that weren't fixed
    remaining_corrupt = conn.execute(
        """
        SELECT id, agent_name, model_label FROM agent_results
        WHERE model_label NOT IN ('0','1','2','3',
            'PRAWDA','CZĘŚCIOWA_PRAWDA','FAŁSZ','MANIPULACJA','NIEWERYFIKOWALNE','4',
            'ERROR','ERROR_MAX_STEPS')
        """
    ).fetchall()

    to_delete = list(unfixable_rows) + list(remaining_corrupt)
    if to_delete:
        print(f"  {len(to_delete)} records will be deleted for re-run:")
        from collections import Counter
        counts = Counter(r["model_label"] for r in to_delete)
        for lbl, cnt in counts.most_common():
            print(f"    '{lbl}': {cnt}")
        if not dry_run:
            ids = [(r["id"],) for r in to_delete]
            conn.executemany("DELETE FROM agent_results WHERE id=?", ids)
            conn.commit()
            print(f"  Deleted {len(to_delete)} records.")
    else:
        print("  Nothing to delete.")
    deleted_total += len(to_delete)

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY  [{mode}]")
    print(f"  Records fixed   : {fixed_total:,}")
    print(f"  Records deleted : {deleted_total:,}  (will be retried by eval_loop)")
    if not skip_mining:
        remaining_max = conn.execute(
            "SELECT COUNT(*) FROM agent_results WHERE model_label='ERROR_MAX_STEPS'"
        ).fetchone()[0]
        print(f"  ERROR_MAX_STEPS still remaining: {remaining_max:,}  (need re-run with max_steps=8)")
    if dry_run:
        print("\n  Re-run without --dry-run to apply changes.")
    print("=" * 60)

    conn.close()


if __name__ == "__main__":
    from claims_processing import paths

    parser = argparse.ArgumentParser(description="Fix corrupted model_label values in benchmark DB.")
    parser.add_argument("--db", default=str(paths.RESULTS_AM_DB))
    parser.add_argument("--dry-run", action="store_true", help="Preview without modifying DB")
    parser.add_argument("--skip-mining", action="store_true", help="Skip slow trajectory-mining pass")
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"ERROR: database not found at {db}", file=sys.stderr)
        sys.exit(1)

    main(str(db), dry_run=args.dry_run, skip_mining=args.skip_mining)
