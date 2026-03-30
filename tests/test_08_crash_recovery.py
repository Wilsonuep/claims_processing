"""
Test eval loop crash recovery and resume behavior.

Scenario:
1. Create an input DB with 10 claims.
2. Run a "crashy" agent that succeeds on claims 1-4, raises on claim 5,
   succeeds on 6-10. After the first pass: 9 rows (4 successes + 1 ERROR + 4 no-retry = only 5 processed).
   Actually: eval_loop catches per-claim exceptions and writes ERROR rows, continues.
   After first pass: 9 successes + 1 ERROR row = 10 rows total.
3. Re-run the agent — ERROR rows are deleted by get_evaluated_claim_ids() and claim 5 is retried.
   After second pass: all 10 rows should have non-ERROR labels.

Also tests:
- Sequential mode crash recovery
- Cloud mode crash recovery
"""
import os
import sys
import time
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gen_agent.base_agent import BaseAgent
from eval.eval_loop import (
    eval_benchmark,
    eval_benchmark_cloud,
    init_results_db,
    get_evaluated_claim_ids,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input_db(path: str, n: int = 10) -> None:
    """Create a test claims DB with n claims."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE claims (id INTEGER PRIMARY KEY, claim_text TEXT, label TEXT)")
    for i in range(1, n + 1):
        conn.execute("INSERT INTO claims VALUES (?, ?, ?)", (i, f"Test claim {i}", "SUPPORTS"))
    conn.commit()
    conn.close()


def _count_results(path: str, agent_name: str, error_only: bool = False) -> int:
    conn = sqlite3.connect(path)
    if error_only:
        count = conn.execute(
            "SELECT COUNT(*) FROM agent_results WHERE agent_name=? AND model_label='ERROR'",
            (agent_name,),
        ).fetchone()[0]
    else:
        count = conn.execute(
            "SELECT COUNT(*) FROM agent_results WHERE agent_name=?",
            (agent_name,),
        ).fetchone()[0]
    conn.close()
    return count


def _cleanup(*paths: str) -> None:
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Crashy agent — fails on the 5th call it processes
# ---------------------------------------------------------------------------

class CrashyAgent(BaseAgent):
    """Agent that raises on every N-th claim (simulates transient failure)."""
    name = "crashy_agent"
    cost_tier = 1

    def __init__(self, crash_on: int = 5):
        self._call_count = 0
        self._crash_on = crash_on

    def eval(self, claim: dict) -> dict:
        self._call_count += 1
        if self._call_count == self._crash_on:
            raise RuntimeError(f"Simulated crash on call #{self._call_count}")
        return {
            "model_label": "SUPPORTS",
            "original_label": claim.get("label", "SUPPORTS"),
            "is_correct": True,
            "total_tokens": 5,
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "time_thought": 0.01,
            "raw_output": "ok",
        }


# ---------------------------------------------------------------------------
# Test: sequential mode crash recovery
# ---------------------------------------------------------------------------

def test_crash_recovery_sequential() -> tuple[bool, float, str | None]:
    start = time.time()
    in_db = "test_crash_in.db"
    out_db = "test_crash_out.db"
    _cleanup(in_db, out_db)

    try:
        _make_input_db(in_db, n=10)

        agent = CrashyAgent(crash_on=5)

        # ── First run: agent crashes on claim 5 ──────────────────────────────
        eval_benchmark(
            benchmark_name="test_crash",
            input_db_path=in_db,
            results_db_path=out_db,
            agents=[agent],
        )

        total_after_1st = _count_results(out_db, "crashy_agent")
        errors_after_1st = _count_results(out_db, "crashy_agent", error_only=True)

        if total_after_1st != 10:
            return False, time.time() - start, (
                f"First run: expected 10 rows, got {total_after_1st}"
            )
        if errors_after_1st != 1:
            return False, time.time() - start, (
                f"First run: expected 1 ERROR row, got {errors_after_1st}"
            )

        # ── Second run: resume — ERROR row is retried ────────────────────────
        agent2 = CrashyAgent(crash_on=999)  # won't crash this time
        eval_benchmark(
            benchmark_name="test_crash",
            input_db_path=in_db,
            results_db_path=out_db,
            agents=[agent2],
        )

        errors_after_2nd = _count_results(out_db, "crashy_agent", error_only=True)
        total_after_2nd = _count_results(out_db, "crashy_agent")

        if errors_after_2nd != 0:
            return False, time.time() - start, (
                f"Second run: expected 0 ERROR rows, got {errors_after_2nd}"
            )
        if total_after_2nd != 10:
            return False, time.time() - start, (
                f"Second run: expected 10 rows, got {total_after_2nd}"
            )

        # ── Third run: nothing to do (all already processed) ─────────────────
        rconn = init_results_db(out_db)
        remaining = get_evaluated_claim_ids(rconn, "crashy_agent", "test_crash")
        rconn.close()

        if len(remaining) != 10:
            return False, time.time() - start, (
                f"Third run check: expected 10 evaluated IDs, got {len(remaining)}"
            )

        return True, time.time() - start, None

    except Exception as e:
        return False, time.time() - start, str(e)
    finally:
        _cleanup(in_db, out_db)


# ---------------------------------------------------------------------------
# Test: cloud mode crash recovery
# ---------------------------------------------------------------------------

def test_crash_recovery_cloud() -> tuple[bool, float, str | None]:
    start = time.time()
    in_db = "test_crash_cloud_in.db"
    out_db = "test_crash_cloud_out.db"
    _cleanup(in_db, out_db)

    try:
        _make_input_db(in_db, n=8)

        agent = CrashyAgent(crash_on=3)

        eval_benchmark_cloud(
            benchmark_name="test_crash_cloud",
            input_db_path=in_db,
            results_db_path=out_db,
            agents=[agent],
            workers=2,
        )

        total = _count_results(out_db, "crashy_agent")
        errors = _count_results(out_db, "crashy_agent", error_only=True)

        if total != 8:
            return False, time.time() - start, f"Cloud: expected 8 rows, got {total}"
        if errors != 1:
            return False, time.time() - start, f"Cloud: expected 1 ERROR, got {errors}"

        # Resume
        agent2 = CrashyAgent(crash_on=999)
        eval_benchmark_cloud(
            benchmark_name="test_crash_cloud",
            input_db_path=in_db,
            results_db_path=out_db,
            agents=[agent2],
            workers=2,
        )

        errors_after = _count_results(out_db, "crashy_agent", error_only=True)
        if errors_after != 0:
            return False, time.time() - start, f"Cloud resume: expected 0 ERRORs, got {errors_after}"

        return True, time.time() - start, None

    except Exception as e:
        return False, time.time() - start, str(e)
    finally:
        _cleanup(in_db, out_db)


# ---------------------------------------------------------------------------
# Combined test entry point (for tester.py)
# ---------------------------------------------------------------------------

def test_crash_recovery() -> tuple[bool, float, str | None]:
    """Run both sequential and cloud crash recovery tests."""
    ok1, t1, err1 = test_crash_recovery_sequential()
    if not ok1:
        return False, t1, f"[sequential] {err1}"

    ok2, t2, err2 = test_crash_recovery_cloud()
    if not ok2:
        return False, t1 + t2, f"[cloud] {err2}"

    return True, t1 + t2, None


if __name__ == "__main__":
    success, elapsed, err = test_crash_recovery()
    if success:
        print(f"PASSED ({elapsed:.3f}s)")
    else:
        print(f"FAILED ({elapsed:.3f}s): {err}")
