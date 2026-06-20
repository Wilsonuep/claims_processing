"""
Narzędzie do scalania wielu baz wyników (SQLite) w jedną.
Użyteczne przy ewaluacji rozproszonej na wielu maszynach.

Użycie:
    python -m scripts.merge_results --target results/merged.db --sources results/machine1.db results/machine2.db
"""

import argparse
import logging
import os
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CREATE_AGENT_RESULTS_SQL = """\
CREATE TABLE IF NOT EXISTS agent_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name        TEXT    NOT NULL,
    claim_id          INTEGER NOT NULL,
    benchmark_name    TEXT    NOT NULL,
    original_label    TEXT    NOT NULL,
    model_label       TEXT    NOT NULL,
    is_correct        INTEGER NOT NULL,
    total_tokens      INTEGER NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    time_thought      REAL    NOT NULL,
    raw_output        TEXT,
    created_at        TEXT    NOT NULL
);
"""

CREATE_IDX_AGENT_NAME = """\
CREATE INDEX IF NOT EXISTS idx_agent_results_agent_name
    ON agent_results(agent_name);
"""

CREATE_IDX_CLAIM_ID = """\
CREATE INDEX IF NOT EXISTS idx_agent_results_claim_id
    ON agent_results(claim_id);
"""

def init_target_db(db_path: str) -> sqlite3.Connection:
    """Inicjalizuje docelową bazę danych wyników."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(CREATE_AGENT_RESULTS_SQL)
    cur.execute(CREATE_IDX_AGENT_NAME)
    cur.execute(CREATE_IDX_CLAIM_ID)
    conn.commit()
    return conn

def merge_databases(target_path: str, source_paths: list[str]) -> None:
    """Scala wiele baz źródłowych do bazy docelowej, ignorując duplikaty."""
    log.info("Inicjalizacja docelowej bazy: %s", target_path)
    conn = init_target_db(target_path)
    
    total_inserted = 0
    total_skipped = 0
    
    for source_db in source_paths:
        if not os.path.exists(source_db):
            log.warning("Plik źródłowy nie istnieje, pomijam: %s", source_db)
            continue
            
        log.info("Dołączanie bazy źródłowej: %s", source_db)
        # Używamy ujednoliconego podłączenia SQLite'owego ATTACH
        # Zabezpieczenie na wypadek dziwnych znaków w ścieżce
        conn.execute(f"ATTACH DATABASE '{source_db}' AS src_db")
        
        cur = conn.cursor()
        
        # Sprawdzenie, ile rekordów jest w bazie źródłowej
        try:
            cur.execute("SELECT COUNT(*) FROM src_db.agent_results")
            source_count = cur.fetchone()[0]
            log.info("Znaleziono %d rekordów w %s", source_count, source_db)
        except sqlite3.OperationalError:
            log.warning("Tabela agent_results nie istnieje w %s, pomijam ten plik.", source_db)
            conn.execute("DETACH DATABASE src_db")
            continue
        
        # Przenoszenie rekordów z obsługą duplikatów 
        # (bazujemy na unikalności par [agent_name, claim_id, benchmark_name])
        try:
            cur.execute("""
                INSERT INTO agent_results (
                    agent_name, claim_id, benchmark_name, original_label,
                    model_label, is_correct, total_tokens, prompt_tokens,
                    completion_tokens, time_thought, raw_output, created_at
                )
                SELECT 
                    s.agent_name, s.claim_id, s.benchmark_name, s.original_label,
                    s.model_label, s.is_correct, s.total_tokens, s.prompt_tokens,
                    s.completion_tokens, s.time_thought, s.raw_output, s.created_at
                FROM src_db.agent_results s
                LEFT JOIN agent_results t 
                    ON s.agent_name = t.agent_name 
                    AND s.claim_id = t.claim_id 
                    AND s.benchmark_name = t.benchmark_name
                WHERE t.id IS NULL
            """)
            inserted_count = cur.rowcount
            skipped_count = source_count - inserted_count
            
            total_inserted += inserted_count
            total_skipped += skipped_count
            
            conn.commit()
            log.info("Dodano %d nowy(ch) rekord(ów), pominięto %d duplikat(ów).", 
                     inserted_count, skipped_count)
            
        except sqlite3.Error as e:
            log.error("Błąd podczas scalania %s: %s", source_db, e)
            conn.rollback()
            
        finally:
            conn.execute("DETACH DATABASE src_db")
            
    conn.close()
    log.info("Scalanie zakończone. Łącznie dodano: %d docelowym DB, pominięto: %d duplikatów ze źródeł.", 
             total_inserted, total_skipped)

def main():
    parser = argparse.ArgumentParser(description="Scala wiele baz wyników (SQLite) z testów w jedną.")
    parser.add_argument(
        "--target", 
        required=True, 
        help="Ścieżka do docelowej bazy danych (np. results/merged_eval.db)"
    )
    parser.add_argument(
        "--sources", 
        nargs="+", 
        required=True, 
        help="Ścieżki do baz źródłowych do scalenia (np. results/machine_1.db results/machine_2.db)"
    )
    args = parser.parse_args()
    
    merge_databases(args.target, args.sources)

if __name__ == "__main__":
    main()
