"""
build_wikipedia_db.py
=====================

Skrypt pobiera JSONL wygenerowany przez scrapper Wikipedii, mapuje płaski tekst (extract)
na ustrukturyzowane sekcje i akapity, a następnie korzysta z `wikipeda_chunking.py` oraz 
`wikipedia_db.py` w celu chunkowania, zwektoryzowania i zapisania do bazy danych wektorowej (SQLite + vec0).
"""

import argparse
import json
import logging
import os
import re
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

    Scrapper Wikipedii używa następującej konwencji znaków nowej linii:
        - ``\\n\\n\\n`` (potrójny) — granica sekcji (nagłówek sekcji jest blokiem
          pomiędzy dwoma potrójnymi znakami nowej linii)
        - ``\\n\\n``  (podwójny) — granica akapitu wewnątrz tej samej sekcji
        - ``\\n``    (pojedynczy) — łamanie wierszy w akapicie (np. listy wypunktowane)

    Algorytm:
        1. Dzielimy tekst na sekcje po ``\\n\\n\\n``.
        2. Pierwsza sekcja = „Wprowadzenie" (brak nagłówka).
        3. Kolejne sekcje: pierwsza linia po splicie to nagłówek,
           reszta to akapity oddzielone ``\\n\\n``.
    """
    sections: list[Section] = []

    # Podział na sekcje (separowane co najmniej potrójnym \n)
    raw_sections = re.split(r'\n{3,}', text)

    for idx, raw_section in enumerate(raw_sections):
        raw_section = raw_section.strip()
        if not raw_section:
            continue

        if idx == 0:
            # Pierwszy blok to zawsze "Wprowadzenie" — nie ma nagłówka
            section_title = "Wprowadzenie"
            body = raw_section
        else:
            # W kolejnych blokach pierwszy wiersz to tytuł sekcji
            lines = raw_section.split('\n', 1)
            section_title = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""

        if not body:
            # Sekcja bez treści (np. „Przypisy", „Uwagi") — pomijamy
            # lub dodajemy jako pustą, żeby nie zgubić metadanych
            continue

        # Akapity wewnątrz sekcji rozdzielone podwójnym \n
        paragraphs = [p.strip() for p in body.split('\n\n') if p.strip()]

        if paragraphs:
            sections.append({
                "section_title": section_title,
                "paragraphs": paragraphs,
            })

    return sections


def main():
    parser = argparse.ArgumentParser(description="Zbuduj bazę wektorową Wikipedii z pliku JSONL.")
    parser.add_argument("--input", default="polish_wikipedia_articles.jsonl", help="Plik wejściowy JSONL ze zrzutu Wikipedii")
    parser.add_argument("--db", default="data/wiki.db", help="Ścieżka do docelowej bazy danych sqlite (vector info)")
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

    # Wczytanie JSONL
    log.info(f"Wczytywanie artykułów z pliku JSONL {args.input}...")
    raw_articles: list[dict] = []
    invalid_lines_count = 0
    with open(input_path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_articles.append(json.loads(line))
            except json.JSONDecodeError as e:
                invalid_lines_count += 1
                if invalid_lines_count <= 10:
                    log.warning(
                        f"Pomijam błędną linię nr {line_no} (JSONDecodeError): {e} "
                        f"- pierwsze 80 znaków: {line[:80]}"
                    )
                elif invalid_lines_count == 11:
                    log.warning("Zbyt wiele błędnych linii, wstrzymuję logowanie dla kolejnych błędów parsowania JSON.")
                continue
            
            if args.limit and len(raw_articles) >= args.limit:
                break

    if invalid_lines_count > 0:
        log.warning(f"Zignorowano łącznie {invalid_lines_count} błędnych linii w procesie ładowania pliku.")

    log.info(f"Wczytano {len(raw_articles)} artykułów z pliku JSONL.")

    if args.limit:
        log.info(f"Nałożono limit: {args.limit}. Przeczytano {len(raw_articles)} artykułów.")

    if not raw_articles:
        log.error("Nie wczytano żadnych artykułów — sprawdź plik wejściowy.")
        sys.exit(1)

    # Deduplikacja artykułów po pageid (scraper może zwrócić duplikaty)
    seen_pageids: set[int] = set()
    deduped_articles: list[dict] = []
    for raw in raw_articles:
        pid = raw.get("pageid", raw.get("id", 0))
        if pid in seen_pageids:
            log.warning("Duplikat pageid %s ('%s') — pomijam.", pid, raw.get("title", ""))
            continue
        seen_pageids.add(pid)
        deduped_articles.append(raw)

    if len(deduped_articles) < len(raw_articles):
        log.info(f"Usunięto {len(raw_articles) - len(deduped_articles)} duplikatów pageid.")
    raw_articles = deduped_articles

    # Parsowanie płaskiego JSON do struktury Article
    log.info(f"Parsowanie {len(raw_articles)} artykułów na ustrukturyzowane obiekty Article...")
    structured_articles: list[Article] = []
    
    for raw in raw_articles:
        text = raw.get("text", "")
        if not text or not text.strip():
            continue
        
        parsed_sections = parse_article_text(text)
        
        structured_articles.append({
            "page_id": raw.get("pageid", raw.get("id", 0)), 
            "title": raw.get("title", "Brak Tytułu"),
            "sections": parsed_sections
        })

    log.info(f"Sparsowano {len(structured_articles)} artykułów (pominięto {len(raw_articles) - len(structured_articles)} pustych).")

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