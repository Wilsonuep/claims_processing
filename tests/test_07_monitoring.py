import os
import sys
import time
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.monitor import MonitoringAgent

def test_monitoring():
    start_time = time.time()
    try:
        # Create an agent with explicit test config
        agent = MonitoringAgent(
            active=True,
            webhook_url="http://mocked.url/webhook",
            machine_name="test-machine"
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
            elapsed_sec=10.0
        )
        
        state = agent._snapshot()
        if state["agent_name"] != "test_agent" or state["done"] != 5:
            return False, time.time() - start_time, "Niepoprawny stan po wykonaniu update()"
            
        # Test 2: _send_progress() triggers requests.post
        with patch('monitoring.monitor.requests.post') as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            agent._send_progress()
            
            if not mock_post.called:
                return False, time.time() - start_time, "Brak wywołania HTTP POST po _send_progress()"
                
            args, kwargs = mock_post.call_args
            payload = kwargs.get('json', {})
            
            # Check if progress payload contains inserted numbers
            if "📊 Progress: 5/10 claims" not in payload.get('message', ''):
                return False, time.time() - start_time, "Brak aktualnych danych w payloadzie wiadomości (progress)"

        # Test 3: report_crash() builds payload and sends HTTP POST
        with patch('monitoring.monitor.requests.post') as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            
            try:
                1 / 0
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                # Calling the internal sync method since report_crash fires a background thread
                agent._send_crash(e, context="test_context", tb=tb)
                
            if not mock_post.called:
                return False, time.time() - start_time, "Brak wywołania HTTP POST po awarii (_send_crash)"
                
            args, kwargs = mock_post.call_args
            payload = kwargs.get('json', {})
            
            # Check if crash payload contains the exception details
            if "ZeroDivisionError" not in payload.get('message', '') or "test_context" not in payload.get('message', ''):
                return False, time.time() - start_time, "Brak poprawnego komunikatu o błędzie w payloadzie"
                
        return True, time.time() - start_time, None
    except Exception as e:
        return False, time.time() - start_time, str(e)
        
if __name__ == "__main__":
    success, elapsed, err = test_monitoring()
    if success: 
        print(f"PASSED (Time: {elapsed:.2f}s)")
    else: 
        print(f"FAILED (Time: {elapsed:.2f}s, Error: {err})")
