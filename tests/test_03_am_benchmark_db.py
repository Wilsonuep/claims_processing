import os
import time
import csv
import sqlite3
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprep.am_benchmark_db import init_db, ingest_am_benchmark

def test_am_benchmark_db():
    start_time = time.time()
    try:
        csv_content = [
            ["question", "year", "correct_answer_index", "name"],
            ["W którym roku był chrzest Polski?", "2024", "SUPPORTS", "historia"]
        ]
        csv_path = "test_am_benchmark.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_content)
            
        db_path = "test_amb.db"
        if os.path.exists(db_path): os.remove(db_path)
        
        conn = init_db(db_path)
        ingest_am_benchmark(csv_path, conn)
        
        count = conn.execute("SELECT count(*) FROM claims").fetchone()[0]
        if count != 1:
             return False, time.time() - start_time, "Oczekiwano 1 rekordu"
            
        conn.close()
        os.remove(csv_path)
        os.remove(db_path)
        
        return True, time.time() - start_time, None
    except Exception as e:
        return False, time.time() - start_time, str(e)
        
if __name__ == "__main__":
    success, elapsed, err = test_am_benchmark_db()
    if success: print(f"PASSED (Time: {elapsed:.2f}s)")
    else: print(f"FAILED (Time: {elapsed:.2f}s, Error: {err})")
