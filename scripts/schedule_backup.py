"""
schedule_backup.py — run backup_dbs every 3 hours in the terminal.

Usage
-----
    python -m scripts.schedule_backup
    python -m scripts.schedule_backup --interval 6      # every 6 hours
    python -m scripts.schedule_backup --dest E:\\backup

Keep this terminal open while evals are running. Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PYTHON = Path(sys.executable)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_backup(dest: Path | None) -> bool:
    cmd = [str(PYTHON), "-m", "scripts.backup_dbs"]
    if dest:
        cmd += ["--dest", str(dest)]
    log.info("Starting backup: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backup_dbs on a fixed interval.")
    parser.add_argument("--interval", type=float, default=3.0, metavar="HOURS",
                        help="Backup interval in hours (default: 3)")
    parser.add_argument("--dest", type=Path, default=None,
                        help="Passed through to backup_dbs --dest")
    parser.add_argument("--no-immediate", action="store_true",
                        help="Skip the initial backup and wait for the first interval")
    args = parser.parse_args()

    interval_secs = args.interval * 3600
    log.info("Scheduler started — backup every %.1f h. Ctrl+C to stop.", args.interval)

    if not args.no_immediate:
        run_backup(args.dest)

    while True:
        next_run = datetime.now() + timedelta(seconds=interval_secs)
        log.info("Next backup at %s", next_run.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            time.sleep(interval_secs)
        except KeyboardInterrupt:
            log.info("Scheduler stopped by user.")
            break
        ok = run_backup(args.dest)
        if not ok:
            log.warning("Backup finished with errors (see above).")


if __name__ == "__main__":
    main()
