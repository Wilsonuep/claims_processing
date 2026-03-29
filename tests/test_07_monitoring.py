import os
import sys
import time
from unittest.mock import patch, MagicMock


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from monitoring.monitor import MonitoringAgent

BRRR_BASE = "https://api.brrr.now/v1/"


def _resolve_webhook_url() -> str:
    """Build a clean brrr.now webhook URL from the environment.

    Accepts any of these .env formats:
        BRRR_WEBHOOK_URL=https://api.brrr.now/v1/br_usr_abc123   (full URL)
        BRRR_WEBHOOK_URL=br_usr_abc123                            (secret only)
    """
    from dotenv import load_dotenv
    load_dotenv()

    raw = os.getenv("BRRR_WEBHOOK_URL", "").strip().strip('"').strip("'")

    if not raw:
        return f"{BRRR_BASE}your_secret"

    # Already a complete URL — use as-is
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw

    # Bare secret — prepend the base
    return f"{BRRR_BASE}{raw}"


def test_monitoring():
    start_time = time.time()
    try:
        webhook_url = _resolve_webhook_url()
        machine_name = os.getenv("MACHINE_NAME", "my-eval-machine").strip()

        # Create an agent using actual .env configuration (falling back to README defaults)
        agent = MonitoringAgent(
            active=True,  # Forced to true for the test to ensure execution
            webhook_url=webhook_url,
            machine_name=machine_name,
        )

        # Test 1: update() changes state
        agent.update(
            agent_name="test_agent",
            benchmark="test_bench",
            done=5,
            total=10,
            correct=4,
            errors=1,
            tokens=100,
            elapsed_sec=10.0,
        )

        state = agent._snapshot()
        if state["agent_name"] != "test_agent" or state["done"] != 5:
            return False, time.time() - start_time, "Niepoprawny stan po wykonaniu update()"

        # Test 2: _send_progress() triggers requests.post (actually send it)
        payload_progress = agent._build_progress_payload(state)
        # Check if progress payload contains inserted numbers
        if "📊 Progress: 5/10 claims" not in payload_progress.get("message", ""):
            return False, time.time() - start_time, "Brak aktualnych danych w payloadzie wiadomości (progress)"

        # Actually send the reminder
        agent._send_progress()

        # Test 3: report_crash() builds payload and sends HTTP POST (actually send it)
        try:
            1 / 0
        except Exception as e:
            import traceback
            tb = traceback.format_exc()

            payload_crash = agent._build_crash_payload(e, context="test_context", tb=tb)
            # Check if crash payload contains the exception details
            if (
                "ZeroDivisionError" not in payload_crash.get("message", "")
                or "test_context" not in payload_crash.get("message", "")
            ):
                return False, time.time() - start_time, "Brak poprawnego komunikatu o błędzie w payloadzie"

            # Calling the internal sync method since report_crash fires a background thread
            agent._send_crash(e, context="test_context", tb=tb)

        return True, time.time() - start_time, None
    except Exception as e:
        return False, time.time() - start_time, str(e)


if __name__ == "__main__":
    success, elapsed, err = test_monitoring()
    if success:
        print(f"PASSED (Time: {elapsed:.2f}s)")
    else:
        print(f"FAILED (Time: {elapsed:.2f}s, Error: {err})")