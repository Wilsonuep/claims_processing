"""
Test MonitoringAgent progress accuracy.

Verifies:
1. phase_started_at resets when done→0 or phase changes.
2. _build_progress_payload computes live rate/ETA from wall-clock
   (not from stale elapsed_sec).
3. Progress percentage uses 2 decimal places (meaningful for large datasets).
4. report_done() fires a well-formed payload.
5. End-to-end: simulate insert_chunks_with_embeddings progress updates and
   verify payload content at each checkpoint.
6. (Optional) Live push notification — runs only when BRRR_WEBHOOK_URL is set.
"""
from __future__ import annotations

import os
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claims_processing.monitoring.monitor import MonitoringAgent, _fmt_duration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _CapturingAgent(MonitoringAgent):
    """MonitoringAgent subclass that captures outgoing payloads instead of
    actually sending them over HTTP — lets us verify content in tests."""

    def __init__(self, **kwargs):
        super().__init__(active=True, webhook_url="http://fake", **kwargs)
        self.sent: list[dict[str, Any]] = []
        self._send_lock = threading.Lock()

    def _send(self, payload: dict[str, Any]) -> bool:  # type: ignore[override]
        with self._send_lock:
            self.sent.append(payload)
        return True

    def last_payload(self) -> dict[str, Any] | None:
        with self._send_lock:
            return self.sent[-1] if self.sent else None

    def payloads_of_type(self, title_substr: str) -> list[dict[str, Any]]:
        with self._send_lock:
            return [p for p in self.sent if title_substr in p.get("title", "")]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_monitoring_progress() -> tuple[bool, float, str | None]:
    start = time.time()
    try:
        # ── 1. phase_started_at resets on done=0 ─────────────────────────────
        mon = _CapturingAgent(machine_name="test-machine")
        t_before = datetime.now()
        time.sleep(0.05)
        mon.update(mode="build_db", phase="inserting chunks", done=0, total=1000)
        time.sleep(0.05)
        with mon._lock:
            ps = mon._state["phase_started_at"]
        if not (t_before < ps <= datetime.now()):
            return False, time.time() - start, "phase_started_at not reset on done=0"

        # ── 2. phase_started_at resets on phase change ────────────────────────
        mon2 = _CapturingAgent(machine_name="test-machine")
        mon2.update(mode="build_db", phase="chunking", done=100, total=200)
        time.sleep(0.05)
        t_before2 = datetime.now()
        time.sleep(0.01)
        mon2.update(phase="inserting chunks")  # phase changed
        with mon2._lock:
            ps2 = mon2._state["phase_started_at"]
        if ps2 < t_before2:
            return False, time.time() - start, "phase_started_at not reset on phase change"

        # ── 3. Live ETA and rate use wall-clock time ──────────────────────────
        mon3 = _CapturingAgent(machine_name="test-machine")
        mon3.update(mode="build_db", phase="inserting chunks", done=0, total=10_000)
        time.sleep(0.1)  # let 100ms pass
        # Simulate 500 chunks done (last mon.update elapsed_sec intentionally wrong/stale)
        mon3.update(done=500, total=10_000, elapsed_sec=0.0)  # stale elapsed_sec

        state = mon3._snapshot()
        payload = mon3._build_progress_payload(state)

        # Rate should be ~500/0.1 = 5000 items/s (we slept 100ms then set done=500)
        # Extract rate from subtitle/message
        msg = payload["message"]
        if "n/a" in msg and state["done"] > 0:
            return False, time.time() - start, (
                f"ETA is 'n/a' despite done=500 total=10000. Payload: {msg}"
            )
        # Rate must be positive
        if "Speed: 0 items/s" in msg:
            return False, time.time() - start, f"Rate is 0 items/s with done=500. Msg: {msg}"

        # ── 4. Percentage uses 2 decimal places ──────────────────────────────
        # Simulate large dataset: 500 out of 1,500,000 → 0.03%
        mon4 = _CapturingAgent(machine_name="test-machine")
        mon4.update(mode="build_db", phase="inserting chunks", done=500, total=1_500_000)
        state4 = mon4._snapshot()
        payload4 = mon4._build_progress_payload(state4)
        subtitle = payload4["subtitle"]
        # With %.2f, "0.03%" should appear, not "0.0%"
        if "0.03" not in subtitle and "0.0%" in subtitle:
            return False, time.time() - start, (
                f"Subtitle shows 0.0% instead of 0.03% for 500/1_500_000. Got: {subtitle}"
            )

        # ── 5. report_done() fires a well-formed payload ──────────────────────
        mon5 = _CapturingAgent(machine_name="test-machine")
        mon5.report_done(
            context="wikipedia_db / insert_chunks",
            lines=["10,000 chunks in 2m 30s", "Avg speed: 67 ch/s"],
        )
        time.sleep(0.1)  # give background thread time to fire
        done_payloads = mon5.payloads_of_type("Done")
        if not done_payloads:
            return False, time.time() - start, "report_done() fired no payload"
        dp = done_payloads[0]
        if "wikipedia_db" not in dp.get("subtitle", ""):
            return False, time.time() - start, f"Done payload subtitle wrong: {dp}"
        if "10,000 chunks" not in dp.get("message", ""):
            return False, time.time() - start, f"Done payload missing lines: {dp}"

        # ── 6. End-to-end: simulate batch insert loop (realistic Wikipedia scale) ──
        mon6 = _CapturingAgent(machine_name="test-machine")
        mon6.update(mode="build_db", phase="inserting chunks", done=0, total=1_500_000)
        time.sleep(0.02)

        # Simulate 5 checkpoints — 300k chunks each, ~60s per batch (≈ 5000 ch/s)
        for i in range(1, 6):
            done_n = i * 300_000
            elapsed_n = i * 60.0  # 60s per 300k batch
            mon6.update(done=done_n, total=1_500_000, elapsed_sec=elapsed_n)
            time.sleep(0.02)

        state6 = mon6._snapshot()
        p6 = mon6._build_progress_payload(state6)
        # After 5 batches of 300k / 1_500_000 = 100%
        pct6 = state6["done"] / max(state6["total"], 1) * 100
        if abs(pct6 - 100.0) > 0.01:
            return False, time.time() - start, f"Expected 100% after 1_500_000/1_500_000, got {pct6:.2f}%"
        msg6 = p6["message"]
        if "Speed: 0 items/s" in msg6:
            return False, time.time() - start, f"Speed is 0 items/s at end of large insert. Msg: {msg6}"

        # ── 7. Live notification (optional — only if BRRR_WEBHOOK_URL is set) ─
        webhook = os.getenv("BRRR_WEBHOOK_URL", "").strip()
        if webhook:
            import requests
            live_mon = MonitoringAgent(machine_name="test-machine")
            live_mon.update(
                mode="build_db",
                phase="test — monitoring accuracy check",
                done=1_234,
                total=1_500_000,
                elapsed_sec=10.0,
            )
            state_live = live_mon._snapshot()
            payload_live = live_mon._build_progress_payload(state_live)
            # Verify rate and elapsed time are meaningful before sending
            msg_live = payload_live.get("message", "")
            if "Speed: 0 items/s" in msg_live:
                return False, time.time() - start, (
                    f"Live payload speed is 0 items/s despite done=1234, elapsed=10s. Msg: {msg_live}"
                )
            if "Running 0s" in msg_live:
                return False, time.time() - start, (
                    f"Live payload shows 0s running time despite elapsed_sec=10.0. Msg: {msg_live}"
                )
            payload_live["title"] = f"🧪 TEST — {payload_live['title']}"
            try:
                resp = requests.post(
                    webhook,
                    json=payload_live,
                    timeout=10,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code not in (200, 201, 202, 204):
                    return False, time.time() - start, (
                        f"Live notification failed: HTTP {resp.status_code} — {resp.text[:200]}"
                    )
            except Exception as e:
                return False, time.time() - start, f"Live notification request failed: {e}"

        return True, time.time() - start, None

    except Exception as e:
        import traceback
        return False, time.time() - start, f"{e}\n{traceback.format_exc()}"


if __name__ == "__main__":
    success, elapsed, err = test_monitoring_progress()
    if success:
        print(f"PASSED ({elapsed:.3f}s)")
    else:
        print(f"FAILED ({elapsed:.3f}s): {err}")
