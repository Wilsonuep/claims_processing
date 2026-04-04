"""
Readiness check: cloud eval pipeline (parallel, real LLM calls).

Same setup as test_04 but uses eval_benchmark_cloud (ThreadPoolExecutor)
with workers=2 to verify parallel claim processing works end-to-end.

Steps:
  1. LLM connectivity ping
  2. Run agents_dem.single.SingleAgent on 3 claims via eval_benchmark_cloud
  3. Verify all 3 results stored, not all ERROR
  4. Verify ordering/deduplication: re-running skips already-done claims

Passes ↔ LLM backend handles parallel requests, cloud mode is functional.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.eval_loop import eval_benchmark_cloud, get_evaluated_claim_ids, init_results_db


_CLAIMS = [
    (1, "Kraków leży w Polsce.", "PRAWDA"),
    (2, "Słońce krąży wokół Ziemi.", "FAŁSZ"),
    (3, "Wisła wpada do Morza Bałtyckiego.", "PRAWDA"),
]


def _make_input_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE claims "
        "(id INTEGER PRIMARY KEY, claim_text TEXT NOT NULL, label TEXT NOT NULL)"
    )
    conn.executemany("INSERT INTO claims VALUES (?, ?, ?)", _CLAIMS)
    conn.commit()
    conn.close()


def _cleanup(*paths: str) -> None:
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def test_eval_cloud() -> tuple[bool, float, str | None]:
    start = time.time()
    input_db = "test_eval_cloud_in.db"
    output_db = "test_eval_cloud_out.db"
    _cleanup(input_db, output_db)

    try:
        # ── 1. LLM connectivity ping ───────────────────────────────────────────
        try:
            from gen_agent.llm_client import client, MODEL, LLM_BACKEND
            ping = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Odpowiedz jednym słowem: tak."}],
                max_tokens=10,
            )
            reply = (ping.choices[0].message.content or "").strip()
            if not reply:
                return False, time.time() - start, (
                    f"LLM ({LLM_BACKEND}/{MODEL}) returned empty ping response — "
                    f"model may be unavailable or deprecated. "
                    f"Check LLM_MODEL in .env."
                )
        except Exception as e:
            return False, time.time() - start, (
                f"LLM backend not reachable ({LLM_BACKEND}/{MODEL}): {e}\n"
                "Check .env: LLM_BACKEND, LLM_MODEL, LLM_BASE_URL, together_api_key"
            )

        # ── 2. Build input DB ─────────────────────────────────────────────────
        _make_input_db(input_db)

        from agents_dem.single import SingleAgent
        agent = SingleAgent()

        # ── 3. First run: parallel eval ───────────────────────────────────────
        eval_benchmark_cloud(
            benchmark_name="test_dem_cloud",
            input_db_path=input_db,
            results_db_path=output_db,
            agents=[agent],
            workers=2,
            limit=len(_CLAIMS),
        )

        res_conn = sqlite3.connect(output_db)
        res_conn.row_factory = sqlite3.Row

        total = res_conn.execute(
            "SELECT COUNT(*) FROM agent_results WHERE agent_name = ?",
            (agent.name,),
        ).fetchone()[0]
        if total != len(_CLAIMS):
            res_conn.close()
            return False, time.time() - start, (
                f"Expected {len(_CLAIMS)} result rows, got {total}"
            )

        errors = res_conn.execute(
            "SELECT COUNT(*) FROM agent_results "
            "WHERE agent_name = ? AND model_label = 'ERROR'",
            (agent.name,),
        ).fetchone()[0]
        if errors == len(_CLAIMS):
            rows = res_conn.execute(
                "SELECT raw_output FROM agent_results WHERE agent_name = ?",
                (agent.name,),
            ).fetchall()
            raw = "; ".join(dict(r)["raw_output"][:120] for r in rows)
            res_conn.close()
            return False, time.time() - start, (
                f"All {len(_CLAIMS)} claims returned ERROR. Outputs: {raw}"
            )

        res_conn.close()

        # ── 4. Resume run: already-done claims are skipped ────────────────────
        rconn = init_results_db(output_db)
        evaluated_ids = get_evaluated_claim_ids(rconn, agent.name, "test_dem_cloud")
        rconn.close()

        # ERROR rows are deleted by get_evaluated_claim_ids, so we expect
        # (len(_CLAIMS) - errors) IDs to be considered "done"
        expected_done = len(_CLAIMS) - errors
        if len(evaluated_ids) != expected_done:
            return False, time.time() - start, (
                f"Resume check: expected {expected_done} evaluated IDs, "
                f"got {len(evaluated_ids)}"
            )

        return True, time.time() - start, None

    except Exception as e:
        import traceback
        return False, time.time() - start, f"{e}\n{traceback.format_exc()}"
    finally:
        _cleanup(input_db, output_db)


if __name__ == "__main__":
    success, elapsed, err = test_eval_cloud()
    if success:
        print(f"PASSED ({elapsed:.2f}s)")
    else:
        print(f"FAILED ({elapsed:.2f}s): {err}")
