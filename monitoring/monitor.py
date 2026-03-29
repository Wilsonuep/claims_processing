"""
MonitoringAgent — periodic push notifications via brrr
=======================================================

Sends push notifications to a mobile device using the brrr API.
Designed for non-invasive, read-only observation of the eval loop
and embedding progress.

Configuration (.env)
---------------------
    MONITORING_ACTIVE=true          # "true" / "false" — master on/off switch
    BRRR_WEBHOOK_URL=https://api.brrr.now/v1/br_usr_...  # your secret URL
    MACHINE_NAME=my-machine         # label included in every notification

Behaviour
---------
- **Scheduled updates** — fires at 08:00, 14:00 and 19:00 local time.
  Uses a background daemon thread so it never blocks the eval loop.
- **Crash alerts** — call ``monitoring.report_crash(exc)`` from any
  except-block; the notification is sent immediately (in a separate thread
  so it doesn't raise even if the network is down).
- **Non-invasive** — the agent only reads counters that callers pass in;
  it never queries the process, DB, or filesystem on its own.

Usage
-----
    from monitoring.monitor import MonitoringAgent

    monitoring = MonitoringAgent()          # reads config from .env
    monitoring.start()                      # launch scheduler thread

    # Inside the eval loop — update counters as you go:
    monitoring.update(
        agent_name="uam_ga1",
        benchmark="demagog",
        done=idx,
        total=total_claims,
        correct=correct_count,
        errors=error_count,
        tokens=total_tokens_sum,
        elapsed_sec=total_time_sum,
    )

    # On unhandled exception:
    monitoring.report_crash(exc, context="eval_loop/uam_ga1")

    monitoring.stop()  # graceful shutdown (optional, it's a daemon thread)
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import threading
import time
import traceback
from datetime import datetime, time as dtime
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Scheduled notification times (local clock, 24-h)
_SCHEDULE_TIMES: tuple[dtime, ...] = (
    dtime(8, 0),   # morning
    dtime(14, 0),  # afternoon
    dtime(19, 0),  # evening
)

_POLL_INTERVAL_SEC: int = 30   # how often the scheduler thread wakes up


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _next_scheduled_time(now: datetime) -> datetime | None:
    """Return the next scheduled datetime that is still in the future today,
    or None if all have passed (caller then checks tomorrow)."""
    today = now.date()
    for t in _SCHEDULE_TIMES:
        candidate = datetime.combine(today, t)
        if candidate > now:
            return candidate
    return None


def _seconds_until_next(now: datetime) -> float:
    """Seconds until the next scheduled notification (could be tomorrow)."""
    nxt = _next_scheduled_time(now)
    if nxt is None:
        # All today's slots passed — find tomorrow's first slot
        import datetime as _dt
        tomorrow = now.date() + _dt.timedelta(days=1)
        nxt = datetime.combine(tomorrow, _SCHEDULE_TIMES[0])
    return max(0.0, (nxt - now).total_seconds())


# ---------------------------------------------------------------------------
# MonitoringAgent
# ---------------------------------------------------------------------------


class MonitoringAgent:
    """Push-notification monitoring agent for long-running eval loops.

    Parameters
    ----------
    active : bool | None
        Override the ``MONITORING_ACTIVE`` env var. Pass ``None`` to read
        from env (default).
    webhook_url : str | None
        Override ``BRRR_WEBHOOK_URL``. Pass ``None`` to read from env.
    machine_name : str | None
        Override ``MACHINE_NAME``. Falls back to hostname if both are unset.
    """

    def __init__(
        self,
        *,
        active: bool | None = None,
        webhook_url: str | None = None,
        machine_name: str | None = None,
    ) -> None:
        # --- Config ---
        _active_env = os.getenv("MONITORING_ACTIVE", "true").strip().lower()
        self.active: bool = active if active is not None else _active_env in ("1", "true", "yes")

        self.webhook_url: str = (
            webhook_url or os.getenv("BRRR_WEBHOOK_URL", "")
        ).strip().strip('"').strip("'")

        self.machine_name: str = (
            machine_name
            or os.getenv("MACHINE_NAME", "")
            or socket.gethostname()
        ).strip()

        # --- Internal state (thread-safe via lock) ---
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "agent_name": "—",
            "benchmark": "—",
            "done": 0,
            "total": 0,
            "correct": 0,
            "errors": 0,
            "tokens": 0,
            "elapsed_sec": 0.0,
            "started_at": datetime.now(),
        }

        # --- Scheduler thread ---
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        if not self.active:
            log.info("[MonitoringAgent] Monitoring is DISABLED (MONITORING_ACTIVE=false).")
        elif not self.webhook_url:
            log.warning(
                "[MonitoringAgent] BRRR_WEBHOOK_URL is not set — notifications will be skipped."
            )
        else:
            log.info(
                "[MonitoringAgent] Active. Machine='%s'. Scheduled at %s.",
                self.machine_name,
                ", ".join(t.strftime("%H:%M") for t in _SCHEDULE_TIMES),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "MonitoringAgent":
        """Start the background scheduler thread. Returns self for chaining."""
        if not self.active:
            return self
        if self._thread and self._thread.is_alive():
            log.debug("[MonitoringAgent] Scheduler already running.")
            return self

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._scheduler_loop,
            name="monitoring-scheduler",
            daemon=True,   # won't block process exit
        )
        self._thread.start()
        log.info("[MonitoringAgent] Scheduler thread started.")
        return self

    def stop(self) -> None:
        """Signal the scheduler to stop and wait briefly for it to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        log.info("[MonitoringAgent] Scheduler thread stopped.")

    def update(
        self,
        *,
        agent_name: str = "",
        benchmark: str = "",
        done: int = 0,
        total: int = 0,
        correct: int = 0,
        errors: int = 0,
        tokens: int = 0,
        elapsed_sec: float = 0.0,
    ) -> None:
        """Update the internal progress state.

        Call this from inside the eval loop (or at its end) to keep the
        monitoring agent up-to-date. This method is thread-safe and
        returns immediately — it never blocks.

        Parameters
        ----------
        agent_name : str   Current agent being evaluated.
        benchmark  : str   Current benchmark name.
        done       : int   Number of claims processed so far.
        total      : int   Total claims to process.
        correct    : int   Correctly classified claims so far.
        errors     : int   Error count so far.
        tokens     : int   Cumulative token usage.
        elapsed_sec: float Wall-clock seconds elapsed for current agent.
        """
        with self._lock:
            if agent_name:
                self._state["agent_name"] = agent_name
            if benchmark:
                self._state["benchmark"] = benchmark
            if done:
                self._state["done"] = done
            if total:
                self._state["total"] = total
            if correct:
                self._state["correct"] = correct
            if errors:
                self._state["errors"] = errors
            if tokens:
                self._state["tokens"] = tokens
            if elapsed_sec:
                self._state["elapsed_sec"] = elapsed_sec

    def report_crash(
        self,
        exc: BaseException,
        context: str = "",
    ) -> None:
        """Send an immediate crash notification.

        Runs in a separate thread so it never raises or blocks the caller,
        even when the network is unavailable.

        Parameters
        ----------
        exc     : BaseException    The exception that was caught.
        context : str              Optional human-readable context label.
        """
        if not self.active:
            return

        tb = traceback.format_exc()
        threading.Thread(
            target=self._send_crash,
            args=(exc, context, tb),
            daemon=True,
            name="monitoring-crash-alert",
        ).start()

    # ------------------------------------------------------------------
    # Private — scheduler loop
    # ------------------------------------------------------------------

    def _scheduler_loop(self) -> None:
        """Run in background thread. Fires notifications at scheduled times."""
        log.debug("[MonitoringAgent] Scheduler loop entering.")

        # Track which slots we already fired today (avoids double-firing)
        _fired_today: set[dtime] = set()
        _last_date = datetime.now().date()

        while not self._stop_event.is_set():
            now = datetime.now()

            # Reset fired-set on new day
            if now.date() != _last_date:
                _fired_today.clear()
                _last_date = now.date()

            # Check each scheduled slot
            for slot in _SCHEDULE_TIMES:
                if slot in _fired_today:
                    continue
                slot_dt = datetime.combine(now.date(), slot)
                # Fire if we're within the poll interval after the slot time
                delta = (now - slot_dt).total_seconds()
                if 0 <= delta < _POLL_INTERVAL_SEC:
                    _fired_today.add(slot)
                    log.info(
                        "[MonitoringAgent] Firing scheduled notification at %s.",
                        slot.strftime("%H:%M"),
                    )
                    # Fire in own thread so a slow network doesn't delay polling
                    threading.Thread(
                        target=self._send_progress,
                        daemon=True,
                        name=f"monitoring-notify-{slot.strftime('%H%M')}",
                    ).start()

            self._stop_event.wait(timeout=_POLL_INTERVAL_SEC)

        log.debug("[MonitoringAgent] Scheduler loop exiting.")

    # ------------------------------------------------------------------
    # Private — notification builders & sender
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict[str, Any]:
        """Thread-safe copy of the current state."""
        with self._lock:
            return dict(self._state)

    def _sys_info(self) -> str:
        """One-liner OS / Python info — for crash alerts."""
        try:
            return (
                f"{platform.system()} {platform.release()} | "
                f"Python {platform.python_version()}"
            )
        except Exception:
            return "unknown system"

    def _build_progress_payload(self, state: dict[str, Any]) -> dict[str, Any]:
        """Build brrr JSON payload for a scheduled progress update."""
        done = state["done"]
        total = state["total"]
        correct = state["correct"]
        errors = state["errors"]
        tokens = state["tokens"]
        elapsed = state["elapsed_sec"]
        agent = state["agent_name"]
        bench = state["benchmark"]
        started = state["started_at"]

        pct = done / max(total, 1) * 100
        acc = correct / max(done - errors, 1) * 100
        hours_running = (datetime.now() - started).total_seconds() / 3600
        tps = tokens / max(elapsed, 0.1)

        # ETA
        if done > 0 and elapsed > 0:
            sec_per_claim = elapsed / done
            eta_sec = sec_per_claim * (total - done)
            eta_str = _fmt_duration(eta_sec)
        else:
            eta_str = "n/a"

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        message_lines = [
            f"📊 Progress: {done}/{total} claims ({pct:.1f}%)",
            f"🎯 Accuracy: {acc:.1f}%  |  ❌ Errors: {errors}",
            f"⚡ Speed: {tps:.0f} tok/s  |  ⏱ ETA: {eta_str}",
            f"🤖 Agent: {agent}  |  📂 Bench: {bench}",
            f"🕐 Running {_fmt_duration(hours_running * 3600)} since start",
            f"🖥 Machine: {self.machine_name}",
            f"🕑 {now_str}",
        ]

        return {
            "title": f"[{self.machine_name}] Eval Progress Update",
            "subtitle": f"{done}/{total} claims · {pct:.1f}%",
            "message": "\n".join(message_lines),
            "sound": "calm1",
        }

    def _build_crash_payload(
        self,
        exc: BaseException,
        context: str,
        tb: str,
    ) -> dict[str, Any]:
        """Build brrr JSON payload for a crash alert."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        exc_type = type(exc).__name__
        exc_msg = str(exc)[:300]  # truncate very long messages

        # Last 5 lines of traceback (keep payload small)
        tb_lines = [l for l in tb.splitlines() if l.strip()]
        tb_short = "\n".join(tb_lines[-5:])

        message_lines = [
            f"💥 {exc_type}: {exc_msg}",
            f"📍 Context: {context}" if context else "",
            f"🖥 Machine: {self.machine_name}",
            f"🕑 {now_str}",
            "",
            "— Traceback (last 5 lines) —",
            tb_short,
        ]
        message = "\n".join(l for l in message_lines if l is not None)

        return {
            "title": f"🚨 CRASH — {self.machine_name}",
            "subtitle": f"{exc_type}: {exc_msg[:80]}",
            "message": message,
            "sound": "warm_soft_error",
            "interruption-level": "time-sensitive",
        }

    def _send(self, payload: dict[str, Any]) -> bool:
        """POST payload to the brrr webhook. Returns True on success."""
        if not self.webhook_url:
            log.warning("[MonitoringAgent] No webhook URL — skipping notification.")
            return False

        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code in (200, 201, 204):
                log.info(
                    "[MonitoringAgent] Notification sent OK (%d): '%s'",
                    resp.status_code,
                    payload.get("title", ""),
                )
                return True
            else:
                log.warning(
                    "[MonitoringAgent] Non-2xx response %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
        except Exception as e:
            log.error("[MonitoringAgent] Failed to send notification: %s", e)
            return False

    def _send_progress(self) -> None:
        state = self._snapshot()
        payload = self._build_progress_payload(state)
        self._send(payload)

    def _send_crash(
        self,
        exc: BaseException,
        context: str,
        tb: str,
    ) -> None:
        payload = self._build_crash_payload(exc, context, tb)
        self._send(payload)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable string like '1h 23m' or '45s'."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, s = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {s}s"
    hours, m = divmod(minutes, 60)
    return f"{hours}h {m}m"
