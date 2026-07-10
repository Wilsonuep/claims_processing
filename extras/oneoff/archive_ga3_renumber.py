"""
archive_ga3_renumber.py — one-off results-DB migration (2026-07-10)
===================================================================

Archives uam_ga3 (rag_claim_decomp) and renumbers the remaining agents down,
mirroring the earlier web-agent archival (commit b180911) — whose row-rename
was run ad-hoc and never version-controlled; this one is committed for the
record.

Rationale: uam_ga3 ran RAGRetriever with RAG_MODE at its default ("bm25") for
every stored result, making it functionally redundant with uam_ga4
(bm25_claim_decomp) — the two share identical prompts and differ only in
retrieval code path.

Per DB (results_am_benchmark.db, results_am_subsample.db), in one
transaction, collision-safe ascending order:

    uam_ga3__*  ->  uam_ga_rag_decomp_arch__*   (rename & retain, not deleted)
    uam_ga4__*  ->  uam_ga3__*
    uam_ga5__*  ->  uam_ga4__*
    uam_ga6__*  ->  uam_ga5__*

Model suffixes (__llama3.1-8b, __hf.co-speakleash-..., ...) are untouched.

Before touching anything, both DBs are copied to
results/_backup_pre_ga3_archive_<YYYYMMDD_HHMMSS>/ via the SQLite
online-backup API (same pattern as tools/backup_dbs.py).

Usage
-----
    python extras/oneoff/archive_ga3_renumber.py --dry-run   # report only
    python extras/oneoff/archive_ga3_renumber.py             # migrate
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from claims_processing import paths

DBS = [paths.RESULTS_AM_DB, paths.RESULTS_AM_SUBSAMPLE_DB]

# (old_prefix, new_prefix) — order matters: each target prefix is vacated
# before it is reused, so the UNIQUE index on (agent_name, claim_id,
# benchmark_name, model_name) can never collide.
RENAMES = [
    ("uam_ga3", "uam_ga_rag_decomp_arch"),
    ("uam_ga4", "uam_ga3"),
    ("uam_ga5", "uam_ga4"),
    ("uam_ga6", "uam_ga5"),
]


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT agent_name, COUNT(*) FROM agent_results GROUP BY agent_name"
    ).fetchall()
    return dict(rows)


def _prefix_total(counts: dict[str, int], prefix: str) -> int:
    return sum(n for name, n in counts.items() if name.startswith(prefix + "__"))


def backup(db: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(str(db))
    dst_conn = sqlite3.connect(str(dest_dir / db.name))
    try:
        src_conn.backup(dst_conn, pages=100, sleep=0.005)
    finally:
        dst_conn.close()
        src_conn.close()
    print(f"  backed up {db.name} -> {dest_dir / db.name}")


def migrate(db: Path, dry_run: bool) -> None:
    print(f"\n=== {db} ===")
    conn = sqlite3.connect(str(db))
    try:
        before = _counts(conn)
        total_before = sum(before.values())
        print(f"  rows before: {total_before}")

        # Preconditions: source prefixes present, archive name absent.
        if _prefix_total(before, "uam_ga_rag_decomp_arch"):
            print("  SKIP: uam_ga_rag_decomp_arch already present (migrated?)")
            return
        for old, _ in RENAMES:
            if _prefix_total(before, old) == 0:
                print(f"  ABORT: no rows for prefix {old}__* — unexpected state")
                sys.exit(1)

        if dry_run:
            for old, new in RENAMES:
                n = _prefix_total(before, old)
                print(f"  would rename {old}__* -> {new}__*  ({n} rows)")
            return

        conn.execute("BEGIN")
        for old, new in RENAMES:
            cur = conn.execute(
                "UPDATE agent_results "
                "SET agent_name = ? || substr(agent_name, ?) "
                "WHERE agent_name LIKE ? ESCAPE '\\'",
                (new, len(old) + 1, old + r"\_\_%"),
            )
            print(f"  {old}__* -> {new}__*  ({cur.rowcount} rows)")

        after = _counts(conn)
        total_after = sum(after.values())
        expected = {}
        for name, n in before.items():
            new_name = name
            for old, new in RENAMES:
                if name.startswith(old + "__"):
                    new_name = new + name[len(old):]
                    break
            expected[new_name] = n
        if total_after != total_before or after != expected:
            conn.execute("ROLLBACK")
            print("  ABORT: post-migration counts do not match — rolled back")
            sys.exit(1)

        conn.execute("COMMIT")
        print(f"  rows after: {total_after} (unchanged) — committed")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dry-run", action="store_true", help="report only, no writes")
    args = parser.parse_args()

    for db in DBS:
        if not db.exists():
            print(f"ABORT: missing DB {db}")
            sys.exit(1)

    if not args.dry_run:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = paths.RESULTS_DIR / f"_backup_pre_ga3_archive_{stamp}"
        print(f"Backing up to {dest} ...")
        for db in DBS:
            backup(db, dest)

    for db in DBS:
        migrate(db, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
