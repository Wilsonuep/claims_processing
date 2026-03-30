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
        # Uwaga: scraper Wikipedii używa \n\n\n (potrójny) jako separator sekcji,
        # \n\n (podwójny) jako separator akapitów, \n (pojedynczy) jako łamanie wierszy.
        mock_data = [{
            "id": 1,
            "title": "Polska",
            "text": (
                "Polska to kraj w Europie. Leży w Europie Środkowej."
                "\n\n\n"
                "Historia"
                "\n\n"
                "Polska ma długą historię. Pierwsza wzmianka pochodzi z X wieku."
                "\n\n\n"
                "Geografia"
                "\n\n"
                "Polska leży nad Wisłą."
            )
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

        # Sprawdź, czy parser poprawnie rozpoznał sekcje
        sections = structured_articles[0]["sections"]
        section_titles = [s["section_title"] for s in sections]
        if "Wprowadzenie" not in section_titles:
            return False, time.time() - start_time, f"Brak sekcji 'Wprowadzenie', znaleziono: {section_titles}"
        if "Historia" not in section_titles:
            return False, time.time() - start_time, f"Brak sekcji 'Historia', znaleziono: {section_titles}"
        if "Geografia" not in section_titles:
            return False, time.time() - start_time, f"Brak sekcji 'Geografia', znaleziono: {section_titles}"

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

        # Sprawdź, że duplikaty chunk_id nie powodują crash (INSERT OR IGNORE)
        try:
            insert_chunks_with_embeddings(conn, chunks, dummy_embed_fn, batch_size=10)
        except Exception as e:
            conn.close()
            if os.path.exists(db_path): os.remove(db_path)
            return False, time.time() - start_time, f"Duplikaty chunk_id powinny być ignorowane, a nie crash: {e}"

        # Liczba rekordów nie powinna się zmienić po ponownym insercie duplikatów
        count_after = conn.execute("SELECT count(*) FROM wiki_chunks").fetchone()[0]
        if count_after != count:
            conn.close()
            if os.path.exists(db_path): os.remove(db_path)
            return False, time.time() - start_time, f"Po re-insercie duplikatów oczekiwano {count}, otrzymano {count_after}"
        
        conn.close()
        os.remove(db_path)
            
        elapsed = time.time() - start_time
        return True, elapsed, None
    except Exception as e:
        # Cleanup na wypadek błędu
        if os.path.exists("test_wiki.db"):
            try: os.remove("test_wiki.db")
            except: pass
        return False, time.time() - start_time, str(e)

if __name__ == "__main__":
    success, elapsed, err = test_wikipedia_db()
    if success: print(f"PASSED (Time: {elapsed:.2f}s)")
    else: print(f"FAILED (Time: {elapsed:.2f}s, Error: {err})")
