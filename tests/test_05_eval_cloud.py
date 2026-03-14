import os
import time
import sqlite3
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gen_agent.base_agent import BaseAgent
from eval.eval_loop import eval_benchmark_cloud

class DummyCloudAgent(BaseAgent):
    name = "dummy_cloud"
    def eval(self, claim):
        return {
            "model_label": "REFUTES",
            "original_label": claim.get("label", "SUPPORTS"),
            "is_correct": False,
            "total_tokens": 15,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "time_thought": 0.2,
            "raw_output": "Mock output cloud"
        }

def test_eval_cloud():
    start_time = time.time()
    try:
        input_db = "test_eval_in_cloud.db"
        output_db = "test_eval_out_cloud.db"
        if os.path.exists(input_db): os.remove(input_db)
        if os.path.exists(output_db): os.remove(output_db)
        
        conn = sqlite3.connect(input_db)
        conn.execute("CREATE TABLE claims (id INTEGER PRIMARY KEY, claim_text TEXT, label TEXT)")
        conn.execute("INSERT INTO claims (claim_text, label) VALUES ('TestCloud', 'SUPPORTS')")
        conn.commit()
        conn.close()
        
        eval_benchmark_cloud(
            benchmark_name="test_bench_cloud",
            input_db_path=input_db,
            results_db_path=output_db,
            agents=[DummyCloudAgent()],
            workers=2,
            limit=1
        )
        
        res_conn = sqlite3.connect(output_db)
        res_conn.row_factory = sqlite3.Row
        
        count = res_conn.execute("SELECT count(*) FROM agent_results").fetchone()[0]
        if count != 1: return False, time.time() - start_time, "Brak wyników w chmurze"
        
        res = dict(res_conn.execute("SELECT * FROM agent_results").fetchone())
        required_attrs = ['is_correct', 'total_tokens', 'time_thought', 'model_label']
        for attr in required_attrs:
            if attr not in res or res[attr] is None:
                return False, time.time() - start_time, f"Brak w bazie atrybutu (cloud): {attr}"
                
        if res["is_correct"] != 0:
            return False, time.time() - start_time, "Niepoprawnie zweryfikowano poprawność is_correct"

        res_conn.close()
        os.remove(input_db)
        os.remove(output_db)
        
        return True, time.time() - start_time, None
    except Exception as e:
        return False, time.time() - start_time, str(e)
        
if __name__ == "__main__":
    success, elapsed, err = test_eval_cloud()
    if success: print(f"PASSED (Time: {elapsed:.2f}s)")
    else: print(f"FAILED (Time: {elapsed:.2f}s, Error: {err})")
