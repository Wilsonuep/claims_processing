import os
import time
import json
import sqlite3
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprep.demagog_db import init_db, ingest_demagog

def test_demagog_db():
    start_time = time.time()
    try:
        mock_data = [{
            "external_id": "123",
            "statement": "Słońce krąży wokół Ziemi.",
            "person_name": "Anonim",
            "rating": "fałsz",
            "detail_url": "http://example.com"
        }]
        json_path = "test_demagog.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(mock_data, f)
            
        db_path = "test_demagog.db"
        if os.path.exists(db_path): os.remove(db_path)
        
        conn = init_db(db_path)
        ingest_demagog(json_path, conn)
        
        count = conn.execute("SELECT count(*) FROM claims").fetchone()[0]
        if count != 1:
            return False, time.time() - start_time, "Błąd przy iniekcji JSON"
            
        label = conn.execute("SELECT label FROM claims").fetchone()[0]
        if label != "REFUTES":
            return False, time.time() - start_time, f"Oczekiwano REFUTES, otrzymano {label}"
            
        conn.close()
        os.remove(json_path)
        os.remove(db_path)
        
        return True, time.time() - start_time, None
    except Exception as e:
        return False, time.time() - start_time, str(e)
        
if __name__ == "__main__":
    success, elapsed, err = test_demagog_db()
    if success: print(f"PASSED (Time: {elapsed:.2f}s)")
    else: print(f"FAILED (Time: {elapsed:.2f}s, Error: {err})")
