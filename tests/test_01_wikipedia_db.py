import os
import json
import sqlite3
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprep.build_wikipedia_db import parse_article_text
from dataprep.wikipeda_chunking import build_wiki_chunks
from dataprep.wikipedia_db import init_db, insert_chunks_with_embeddings

def test_wikipedia_db():
    start_time = time.time()
    try:
        mock_data = [{
            "id": 1,
            "title": "Polska",
            "text": "Polska to kraj w Europie. \n\nHistoria\n\nPolska ma długą historię."
        }]
        
        structured_articles = []
        for raw in mock_data:
            text = raw.get("text", "")
            parsed_sections = parse_article_text(text)
            structured_articles.append({
                "page_id": raw.get("id", 0),
                "title": raw.get("title", ""),
                "sections": parsed_sections
            })

        chunks = build_wiki_chunks(structured_articles)
        if len(chunks) == 0:
            return False, time.time() - start_time, "Brak wygenerowanych chunków"

        embed_dim = 4
        def dummy_embed_fn(t): return [0.1, 0.2, 0.3, 0.4]

        db_path = "test_wiki.db"
        if os.path.exists(db_path): os.remove(db_path)
        
        conn = init_db(db_path, embedding_dim=embed_dim)
        insert_chunks_with_embeddings(conn, chunks, dummy_embed_fn, batch_size=10)
        
        count = conn.execute("SELECT count(*) FROM wiki_chunks").fetchone()[0]
        if count != len(chunks):
            return False, time.time() - start_time, f"Oczekiwano {len(chunks)} wierszy, otrzymano {count}"
        
        conn.close()
        os.remove(db_path)
            
        elapsed = time.time() - start_time
        return True, elapsed, None
    except Exception as e:
        return False, time.time() - start_time, str(e)

if __name__ == "__main__":
    success, elapsed, err = test_wikipedia_db()
    if success: print(f"PASSED (Time: {elapsed:.2f}s)")
    else: print(f"FAILED (Time: {elapsed:.2f}s, Error: {err})")
