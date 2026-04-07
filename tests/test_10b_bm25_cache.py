"""
Test BM25Index process-level cache (OOM regression).

Verifies:
1. Two from_sqlite() calls with the same path return the SAME object (identity).
2. _INDEX_CACHE has exactly one entry after two loads.
3. A call with limit= (debug mode) does NOT go into cache.
4. A call with a different path creates a separate cache entry.
"""
import os
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gen_agent.bm25 import BM25Index


def _make_test_db(path: str, n_rows: int = 20) -> None:
    """Creates a minimal SQLite DB with wiki_chunks table for testing."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE wiki_chunks (chunk_id TEXT PRIMARY KEY, text TEXT, title TEXT)"
    )
    conn.executemany(
        "INSERT INTO wiki_chunks VALUES (?, ?, ?)",
        [(f"c{i}", f"Dokument testowy numer {i} o historii Polski", f"Artykuł {i}")
         for i in range(n_rows)],
    )
    conn.commit()
    conn.close()


def test_bm25_cache() -> tuple[bool, float, str | None]:
    start = time.time()
    try:
        # ── Setup: clear cache and create test DB ─────────────────────────────
        BM25Index._INDEX_CACHE.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_wiki.db")
            _make_test_db(db_path)

            # ── 1. Two loads → same object ────────────────────────────────────
            idx1 = BM25Index.from_sqlite(db_path, table="wiki_chunks", text_column="text")
            idx2 = BM25Index.from_sqlite(db_path, table="wiki_chunks", text_column="text")

            if idx1 is not idx2:
                return False, time.time() - start, (
                    "Two from_sqlite() calls returned different objects — cache not working"
                )

            # ── 2. Exactly one cache entry ────────────────────────────────────
            if len(BM25Index._INDEX_CACHE) != 1:
                return False, time.time() - start, (
                    f"Expected 1 cache entry, got {len(BM25Index._INDEX_CACHE)}"
                )

            # ── 3. limit= debug load does NOT go into cache ───────────────────
            BM25Index._INDEX_CACHE.clear()
            idx_debug = BM25Index.from_sqlite(
                db_path, table="wiki_chunks", text_column="text", limit=5
            )
            if len(BM25Index._INDEX_CACHE) != 0:
                return False, time.time() - start, (
                    "Debug load (limit=5) should NOT be cached, but was"
                )

            # ── 4. Second DB path → separate cache entry ──────────────────────
            db_path2 = os.path.join(tmpdir, "test_wiki2.db")
            _make_test_db(db_path2)

            BM25Index._INDEX_CACHE.clear()
            BM25Index.from_sqlite(db_path, table="wiki_chunks", text_column="text")
            BM25Index.from_sqlite(db_path2, table="wiki_chunks", text_column="text")

            if len(BM25Index._INDEX_CACHE) != 2:
                return False, time.time() - start, (
                    f"Expected 2 cache entries for 2 different DBs, got {len(BM25Index._INDEX_CACHE)}"
                )

        BM25Index._INDEX_CACHE.clear()
        return True, time.time() - start, None

    except Exception as exc:
        BM25Index._INDEX_CACHE.clear()
        return False, time.time() - start, f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    ok, elapsed, err = test_bm25_cache()
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] test_bm25_cache ({elapsed:.2f}s)")
    if err:
        print(f"  Error: {err}")
    sys.exit(0 if ok else 1)
