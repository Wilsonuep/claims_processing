"""
backup_dbs.py — safe SQLite backup to external drive
=====================================================

Copies all result and data DBs to D:\\claims_backup\\ using the SQLite
online-backup API (sqlite3.connect().backup()). This never locks the source
DB for more than a few milliseconds per page, so running eval is not
interrupted and no watchdog/WAL inconsistency issues occur.

Usage
-----
    python -m scripts.backup_dbs           # backs up results/ and data/ DBs
    python -m scripts.backup_dbs --dest E:\\claims_backup

Schedule (Windows Task Scheduler — run once to register, then every 12 h):
    schtasks /create /tn "ClaimsDBBackup" /tr "\"C:\\Users\\piotr\\claims_processing\\.venv\\Scripts\\python.exe\" -m scripts.backup_dbs" /sc daily /mo 1 /st 00:00 /ri 720 /du 9999:59 /f

    /ri 720  = repeat every 720 minutes (12 h)
    /du 9999:59 = repeat indefinitely within the day window
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# DBs to back up — relative to PROJECT_ROOT
_DB_GLOBS = [
    "results/*.db",
    "data/*.db",
]

DEFAULT_DEST = Path("D:/claims_backup")
KEEP_SNAPSHOTS = 5  # how many timestamped backups to retain


def backup_db(src: Path, dest_dir: Path) -> None:
    """Copy *src* into *dest_dir* using sqlite3 online backup API.

    The API copies page-by-page with a tiny sleep between batches so the
    source connection is never held exclusively. Safe to call on a live DB.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name

    src_conn = sqlite3.connect(str(src))
    dst_conn = sqlite3.connect(str(dest))
    try:
        # pages=-1 copies everything in one shot; progress callback logs %.
        def _progress(status, remaining, total):
            if total > 0 and remaining % max(total // 10, 1) == 0:
                pct = (total - remaining) / total * 100
                log.debug("  %s … %.0f%%", src.name, pct)

        src_conn.backup(dst_conn, pages=100, progress=_progress, sleep=0.005)
        size_mb = dest.stat().st_size / 1_048_576
        log.info("OK  %-40s  →  %s  (%.1f MB)", src.name, dest, size_mb)
    finally:
        dst_conn.close()
        src_conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup SQLite DBs to external drive.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args()

    dest_root: Path = args.dest
    if not dest_root.drive:
        log.error("Destination drive not found: %s", dest_root)
        sys.exit(1)

    # Timestamped sub-folder so each run is a distinct snapshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = dest_root / timestamp

    dbs: list[Path] = []
    for pattern in _DB_GLOBS:
        dbs.extend(sorted(PROJECT_ROOT.glob(pattern)))

    # Filter out -shm and -wal sidecar files (not real DBs)
    dbs = [p for p in dbs if p.suffix == ".db"]

    if not dbs:
        log.warning("No .db files found under %s", PROJECT_ROOT)
        return

    log.info("Backing up %d DB(s) → %s", len(dbs), dest_dir)
    errors = 0
    for db in dbs:
        try:
            backup_db(db, dest_dir)
        except Exception as exc:
            log.error("FAIL  %s: %s", db.name, exc)
            errors += 1

    if errors:
        log.warning("Backup completed with %d error(s).", errors)
        sys.exit(1)
    else:
        log.info("Backup complete. All %d DB(s) saved to %s", len(dbs), dest_dir)

    # Remove old snapshots, keep only the most recent KEEP_SNAPSHOTS
    snapshots = sorted(
        [d for d in dest_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    to_delete = snapshots[:-KEEP_SNAPSHOTS]
    for old in to_delete:
        import shutil
        shutil.rmtree(old)
        log.info("Removed old snapshot: %s", old.name)


if __name__ == "__main__":
    main()
