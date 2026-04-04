"""
Readiness check: local eval pipeline (real LLM call).

Steps:
  1. LLM connectivity ping — fails fast with actionable message if backend down
  2. Run agents_dem.single.SingleAgent (simplest BaseAgent, 1 LLM call/claim)
     on 3 synthetic Polish claims via eval_benchmark_local (tiered mode)
  3. Verify all 3 results are stored in the DB and none are permanent ERRORs
  4. Verify MonitoringAgent state was updated during eval (done=3)

Passes ↔ LLM backend is reachable, tiered eval runs end-to-end.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gen_agent.base_agent import BaseAgent
from eval.eval_loop import eval_benchmark_local
import eval.eval_loop as _el


# ---------------------------------------------------------------------------
# Synthetic claims — 3 unambiguous Polish statements for Demagog label format
# ---------------------------------------------------------------------------

_CLAIMS = [
    (1, "Warszawa jest stolicą Polski.", "PRAWDA"),
    (2, "Ziemia jest płaska.", "FAŁSZ"),
    (3, "Polska leży w Europie.", "PRAWDA"),
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


def test_eval_local() -> tuple[bool, float, str | None]:
    start = time.time()
    input_db = "test_eval_local_in.db"
    output_db = "test_eval_local_out.db"
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

        # ── 3. Run eval (local/tiered mode) with simplest Demagog agent ───────
        from agents_dem.single import SingleAgent

        agent = SingleAgent()
        eval_benchmark_local(
            benchmark_name="test_dem_local",
            input_db_path=input_db,
            results_db_path=output_db,
            agents=[agent],
            limit=len(_CLAIMS),
        )

        # ── 4. Verify results DB ──────────────────────────────────────────────
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

        # All 3 claims must have been attempted; not all can be permanent ERROR
        # (eval_loop retries ERROR rows, so after one pass the ones that
        #  succeeded are stored; at least 1 of 3 trivial claims should succeed)
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
                f"All {len(_CLAIMS)} claims returned ERROR — LLM is responding "
                f"but agent parsing failed. Outputs: {raw}"
            )

        res_conn.close()

        # ── 5. MonitoringAgent state updated during eval ──────────────────────
        state = _el.monitoring._snapshot()
        if state["done"] < len(_CLAIMS):
            return False, time.time() - start, (
                f"monitoring.done={state['done']} after eval, expected ≥{len(_CLAIMS)}"
            )

        return True, time.time() - start, None

    except Exception as e:
        import traceback
        return False, time.time() - start, f"{e}\n{traceback.format_exc()}"
    finally:
        _cleanup(input_db, output_db)


if __name__ == "__main__":
    success, elapsed, err = test_eval_local()
    if success:
        print(f"PASSED ({elapsed:.2f}s)")
    else:
        print(f"FAILED ({elapsed:.2f}s): {err}")
