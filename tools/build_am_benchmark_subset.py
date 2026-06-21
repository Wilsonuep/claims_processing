"""
Build a reproducible 4000-claim subset of the AM benchmark.

Reads `data/am_benchmark.db`, samples `SAMPLE_SIZE` claim ids using
`random.Random(SEED)`, and writes them to `data/am_benchmark_4k.db` with
the same schema (table + indexes) as the source.

Re-running with the same SEED produces a DB with identical claim ids,
so resume in the eval loop works correctly: the same 4 000 ids are
loaded every run.

    python scripts/build_am_benchmark_subset.py
"""

from __future__ import annotations

import logging
import random
import sqlite3
import sys
from pathlib import Path

from claims_processing import paths

SOURCE_DB = paths.AM_BENCHMARK_DB
OUTPUT_DB = paths.AM_BENCHMARK_4K_DB
SAMPLE_SIZE = 4000
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _read_claims_ddl(src: sqlite3.Connection) -> list[str]:
    """Returns CREATE TABLE / CREATE INDEX statements for the `claims` table."""
    rows = src.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE tbl_name = 'claims' AND sql IS NOT NULL"
    ).fetchall()
    return [r[0] for r in rows]


def main() -> None:
    if not SOURCE_DB.exists():
        log.error("Source DB not found: %s", SOURCE_DB)
        sys.exit(1)

    log.info("Source DB:  %s", SOURCE_DB)
    log.info("Output DB:  %s", OUTPUT_DB)
    log.info("Sample size: %d   Seed: %d", SAMPLE_SIZE, SEED)

    with sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True) as src:
        pool = [row[0] for row in src.execute(
            "SELECT id FROM claims ORDER BY id"
        ).fetchall()]
        ddl = _read_claims_ddl(src)

    if len(pool) < SAMPLE_SIZE:
        log.error("Pool has %d claims, need %d", len(pool), SAMPLE_SIZE)
        sys.exit(1)

    log.info("Pool size: %d", len(pool))
    chosen = sorted(random.Random(SEED).sample(pool, SAMPLE_SIZE))
    log.info("Sampled %d ids; first 5 = %s", len(chosen), chosen[:5])

    if OUTPUT_DB.exists():
        log.info("Removing existing %s", OUTPUT_DB)
        OUTPUT_DB.unlink()

    with sqlite3.connect(OUTPUT_DB) as dst:
        for stmt in ddl:
            dst.execute(stmt)
        dst.execute(f"ATTACH DATABASE '{SOURCE_DB.as_posix()}' AS source")
        placeholders = ",".join("?" * len(chosen))
        dst.execute(
            f"INSERT INTO claims SELECT * FROM source.claims WHERE id IN ({placeholders})",
            chosen,
        )
        dst.commit()
        dst.execute("DETACH DATABASE source")

        n_rows = dst.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        label_dist = dst.execute(
            "SELECT label_original, COUNT(*) FROM claims "
            "GROUP BY label_original ORDER BY label_original"
        ).fetchall()

    log.info("Wrote %d rows to %s", n_rows, OUTPUT_DB)
    log.info("label_original distribution (ground truth): %s", dict(label_dist))


if __name__ == "__main__":
    main()
