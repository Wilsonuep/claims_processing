"""
build_wikipedia_db.py
=====================

Skrypt pobiera JSON wygenerowany przez scrapper Wikipedii, mapuje płaski tekst (extract)
na ustrukturyzowane sekcje i akapity, a następnie korzysta z `wikipeda_chunking.py` oraz 
`wikipedia_db.py` w celu chunkowania, zwektoryzowania i zapisania do bazy danych wektorowej (SQLite + vec0).
"""

import argparse
import json
import logging
import os
import sys

from pathlib import Path

# Dodanie ścieżki do lokalnych modułów (katalogu wyżej)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprep.wikipeda_chunking import build_wiki_chunks, Article, Section
from dataprep.wikipedia_db import init_db, insert_chunks_with_embeddings
from dataprep.wikipedia_embedding import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Monitoring integration (lazy — no side-effects on import)
# ---------------------------------------------------------------------------

_monitoring = None  # type: ignore[assignment]


def _get_monitoring():
    global _monitoring
    if _monitoring is None:
        try:
            from monitoring.monitor import MonitoringAgent
            _monitoring = MonitoringAgent()
        except Exception:
            class _NoOp:
                def start(self): return self
                def stop(self): pass
                def report_crash(self, *_, **__): pass
                def update(self, **_): pass
            _monitoring = _NoOp()
    return _monitoring


def parse_article_text(text: str) -> list[Section]:
    """
    Mapuje płaski tekst artykułu na listę obiektów Section (zgodnych z Article TypedDict).
    Scrapper zapisuje plain-text w polu 'extract'. Używamy prostej heurystyki,
    aby na podstawie podwójnych znaków nowej linii (\n\n) podzielić zawartość na akapity i sekcje.
    """
    sections: list[Section] = []
    lines = text.split("\n\n")
    
    current_section = "Wprowadzenie"
    current_paragraphs: list[str] = []
    
    for block in lines:
        block = block.strip()
        if not block:
            continue
        
        # Heurystyka do łapania nagłówków sekcji:
        # Krótki tekst, 1 linia, bez standardowych znaków kończących zdanie.
        if len(block) < 100 and '\n' not in block and not block.endswith(('.', '!', '?', ':')):
            if current_paragraphs:
                sections.append({
                    "section_title": current_section,
                    "paragraphs": current_paragraphs
                })
            current_section = block
            current_paragraphs = []
        else:
            current_paragraphs.append(block)
            
    if current_paragraphs:
        sections.append({
            "section_title": current_section,
            "paragraphs": current_paragraphs
        })
        
    return sections


def main():
    parser = argparse.ArgumentParser(description="Zbuduj bazę wektorową Wikipedii z pliku JSON.")
    parser.add_argument("--input", default="polish_wikipedia_articles.json", help="Plik wejściowy JSON ze zrzutu Wikipedii")
    parser.add_argument("--db", default="dataprep/wiki.db", help="Ścieżka do docelowej bazy danych sqlite (vector info)")
    parser.add_argument("--limit", type=int, default=None, help="Liczba artykułów do przetworzenia w ramach bazy (do celów testowych)")
    parser.add_argument("--embed-model", default="sdadas/mmlw-retrieval-roberta-large-v2", help="Rodzaj modelu wczytywanego z HF (sentence-transformers)")
    parser.add_argument("--batch-size", type=int, default=500, help="Rozmiar batcha do zapisu bazy danych i wektoryzacji")
    
    args = parser.parse_args()

    mon = _get_monitoring()
    mon.start()
    try:
        _run(args, mon)
    finally:
        mon.stop()


def _run(args, mon) -> None:
    """Core pipeline logic, separated from main() for clean monitoring wrap."""
    # Sprawdzenie, czy istnieje plik wejściowy
    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"!!! Błąd: Nie znaleziono pliku '{args.input}'.")
        log.error("Zanim uruchomisz ten skrypt, odpal datascrap/polish_wikipedia_webscrapper.py")
        sys.exit(1)

    # Wczytanie JSON
    log.info(f"Wczytywanie artykułów z pliku {args.input}...")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_articles = json.load(f)

    if args.limit:
        log.info(f"Nałożono limit. Wybieram pierwsze --limit {args.limit} z {len(raw_articles)} artykułów.")
        raw_articles = raw_articles[:args.limit]

    # Parsowanie płaskiego JSON do struktury Article
    log.info(f"Parsowanie {len(raw_articles)} artykułów na ustrukturyzowane obiekty Article...")
    structured_articles: list[Article] = []
    
    for raw in raw_articles:
        text = raw.get("text", "")
        parsed_sections = parse_article_text(text)
        
        structured_articles.append({
            "page_id": raw.get("pageid", raw.get("id", 0)), 
            "title": raw.get("title", "Brak Tytułu"),
            "sections": parsed_sections
        })

    # Chunkowanie za pomocą dataprep.wikipeda_chunking
    log.info("Rozpoczynam cięcie artykułów na chunki (okna wielozdaniowe)...")
    try:
        chunks = build_wiki_chunks(structured_articles)
    except Exception as e:
        mon.report_crash(e, context="build_wikipedia_db/build_wiki_chunks")
        log.error(f"Błąd podczas cięcia artykułów (chunking): {e}")
        sys.exit(1)
        
    log.info(f"Zakończono cięcie: wygenerowano w sumie {len(chunks)} chunków.")

    if not chunks:
        log.warning("Brak chunków do zwektoryzowania — prawdopodobnie zbiór był pusty.")
        sys.exit(0)

    # Inicjalizacja Modelu
    log.info(f"Inicjalizacja modelu sentence-transformers '{args.embed_model}'...")
    try:
        embed_model = load_model(args.embed_model)
    except Exception as e:
        mon.report_crash(e, context="build_wikipedia_db/load_model")
        log.error(f"Nie udało się załadować modelu {args.embed_model}: {e}")
        sys.exit(1)
    
    # Przetworzenie wymiaru wektorów
    log.info("Zbadanie wymiarów generowanych wektorów z modelu (embed dim)...")
    test_vec = embed_model.encode("test").tolist()
    embed_dim = len(test_vec)
    log.info(f"Wymiar wektorów: {embed_dim} (model: {args.embed_model})")

    def embed_fn(text: str) -> list[float]:
        return embed_model.encode(text, normalize_embeddings=True).tolist()

    # Tworzenie pliku docelowej bazy (jeśli brakuje w ogóle katalogu, to tworzymy wpierw by uniknąć błędu SQLite)
    db_path_obj = Path(args.db)
    if not db_path_obj.parent.exists():
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Inicjalizacja wektorowej bazy na docelowej przestrzeni
    log.info(f"Inicjalizacja środowiska SQLite (z vec0). Plik docelowy: {args.db}")
    try:
        conn = init_db(str(args.db), embedding_dim=embed_dim)
    except Exception as e:
        mon.report_crash(e, context="build_wikipedia_db/init_db")
        log.error(f"Nie udało się zainicjalizować bazy SQLite-vec: {e}")
        sys.exit(1)

    # Zapis fragmentów
    log.info(f"Zapis i wektoryzacja (rozmiar transakcji / batch-size: {args.batch_size}). Cierpliwości...")
    
    try:
        insert_chunks_with_embeddings(conn, chunks, embed_fn, batch_size=args.batch_size)
    except Exception as e:
        mon.report_crash(e, context="build_wikipedia_db/insert_chunks_with_embeddings")
        log.error(f"Nie powiodła się próba zapisu bazy danych: {e}")
        conn.close()
        sys.exit(1)
        
    conn.close()
    log.info("Zakończono! Baza wiedzy (wektorowa i BM25 sqlite) zasiliona poprawnie.")

if __name__ == "__main__":
    main()
