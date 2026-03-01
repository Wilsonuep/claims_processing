import time
import json
import sys
import cloudscraper
from datetime import datetime, timezone
from urllib.parse import quote

# Reconfigure stdout for UTF-8 on Windows (prevents UnicodeEncodeError
# when printing emoji/special chars in cmd.exe / PowerShell)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


API_URL = "https://pl.wikipedia.org/w/api.php"

session = cloudscraper.create_scraper()

# Timeout dla requestów (sekundy)
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Pobieranie listy artykułów (namespace 0 = artykuły)
# ---------------------------------------------------------------------------

def fetch_article_batch(ap_continue: str | None = None,
                        limit: int = 50) -> tuple[list[dict], str | None]:
    """
    Pobiera partię tytułów artykułów z polskiej Wikipedii
    za pomocą list=allpages (namespace 0 = główna przestrzeń).

    Zwraca:
    - listę słowników z kluczami 'pageid' i 'title'
    - token kontynuacji (None jeśli koniec)
    """
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": 0,
        "aplimit": limit,
        "apfilterredir": "nonredirects",
        "format": "json",
    }
    if ap_continue:
        params["apcontinue"] = ap_continue

    r = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    pages = data.get("query", {}).get("allpages", [])
    cont = data.get("continue", {}).get("apcontinue")

    return pages, cont


# ---------------------------------------------------------------------------
# Pomocnicze: zapytanie batch z obsługą kontynuacji i retries
# ---------------------------------------------------------------------------

def _query_batch(titles: list[str], retries: int = 3, **extra_params) -> list[dict]:
    """
    Wykonuje action=query z podanymi parametrami dla partii tytułów.
    Obsługuje kontynuację wewnątrz zapytania oraz ponawia próby.
    """
    params = {
        "action": "query",
        "titles": "|".join(titles),
        "format": "json",
        "formatversion": 2,
        **extra_params,
    }

    all_pages: dict[int, dict] = {}

    while True:
        data = None
        for attempt in range(1, retries + 1):
            try:
                r = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                data = r.json()

                # Sprawdź czy API nie zwróciło błędu
                if "error" in data:
                    raise RuntimeError(f"API error: {data['error'].get('info', data['error'])}")
                break
            except Exception as e:
                if attempt == retries:
                    raise
                wait = 2 ** attempt
                print(f"    ↻ Retry {attempt}/{retries} za {wait}s ({e})")
                time.sleep(wait)

        for p in data.get("query", {}).get("pages", []):
            pid = p.get("pageid")
            if pid is None or p.get("missing"):
                continue
            if pid in all_pages:
                for key in p:
                    if key in ("pageid", "title", "ns"):
                        continue
                    if isinstance(p[key], list) and key in all_pages[pid]:
                        all_pages[pid][key].extend(p[key])
                    else:
                        all_pages[pid][key] = p[key]
            else:
                all_pages[pid] = p

        if "continue" not in data:
            break
        params.update(data["continue"])

    return list(all_pages.values())



# ---------------------------------------------------------------------------
# Pobieranie metadanych i treści artykułów
# ---------------------------------------------------------------------------

def fetch_article_details(titles: list[str], delay: float = 0.3) -> list[dict]:
    """
    Pobiera szczegóły dla partii artykułów (max 50 tytułów).
    Wszystkie zapytania są batch (szybkie, wiele artykułów na raz):
    1. extracts + info     — tekst artykułu, długość, data ostatniej modyfikacji
    2. extlinks            — liczba linków zewnętrznych (≈ referencje)
    3. categories          — kategorie artykułu
    """

    # --- 1. Tekst (extract) + info (batch) ---
    pages_main = _query_batch(
        titles,
        prop="extracts|info",
        exlimit="max",
        explaintext="1",
        exsectionformat="plain",
    )

    page_map: dict[int, dict] = {}
    for p in pages_main:
        pid = p["pageid"]
        page_map[pid] = {
            "pageid": pid,
            "title": p.get("title"),
            "url": f"https://pl.wikipedia.org/wiki/{quote(p.get('title', '').replace(' ', '_'), safe='/:@!$&()*+,;=')}",
            "text": p.get("extract"),
            "content_length": p.get("length"),
            "last_edited": p.get("touched"),
            "number_of_references": 0,
            "categories": [],
        }

    time.sleep(delay)

    # --- 2. Linki zewnętrzne (batch) ---
    try:
        pages_extlinks = _query_batch(
            titles,
            prop="extlinks",
            ellimit="max",
        )
        for p in pages_extlinks:
            pid = p.get("pageid")
            if pid and pid in page_map:
                page_map[pid]["number_of_references"] = len(p.get("extlinks", []))
    except Exception as e:
        print(f"    ⚠ Nie udało się pobrać referencji: {e}")

    time.sleep(delay)

    # --- 3. Kategorie (batch) ---
    try:
        pages_cats = _query_batch(
            titles,
            prop="categories",
            cllimit="max",
        )
        for p in pages_cats:
            pid = p.get("pageid")
            if pid and pid in page_map:
                cats = p.get("categories", [])
                page_map[pid]["categories"] = [
                    c["title"].replace("Kategoria:", "") for c in cats
                ]
    except Exception as e:
        print(f"    ⚠ Nie udało się pobrać kategorii: {e}")

    return list(page_map.values())


# ---------------------------------------------------------------------------
# Zapis JSON i stan (resume)
# ---------------------------------------------------------------------------

def _save_json(data: list[dict], path: str):
    """Zapisuje listę słowników do pliku JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _state_path(output_path: str) -> str:
    """Zwraca ścieżkę do pliku stanu (obok pliku wynikowego)."""
    return output_path + ".state.json"


def _save_state(output_path: str, ap_continue: str | None,
                total_fetched: int, batch_num: int):
    """Zapisuje stan scrapingu — token kontynuacji i licznik artykułów."""
    state = {
        "apcontinue": ap_continue,
        "total_fetched": total_fetched,
        "batch_num": batch_num,
    }
    with open(_state_path(output_path), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_state(output_path: str) -> tuple[list[dict], str | None, int, int]:
    """
    Próbuje wczytać istniejące wyniki i stan.
    Zwraca (results, apcontinue, total_fetched, batch_num).
    Jeśli brak pliku stanu — zwraca puste wartości (start od zera).
    """
    import os
    sp = _state_path(output_path)

    if not os.path.exists(sp) or not os.path.exists(output_path):
        return [], None, 0, 0

    try:
        with open(sp, "r", encoding="utf-8") as f:
            state = json.load(f)
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        return (
            results,
            state.get("apcontinue"),
            state.get("total_fetched", len(results)),
            state.get("batch_num", 0),
        )
    except Exception:
        return [], None, 0, 0


def _delete_state(output_path: str):
    """Usuwa plik stanu po zakończeniu scrapingu."""
    import os
    sp = _state_path(output_path)
    if os.path.exists(sp):
        os.remove(sp)


# ---------------------------------------------------------------------------
# Główna funkcja: scraping polskiej Wikipedii
# ---------------------------------------------------------------------------

def scrape_all_to_json(
    output_path: str = "polish_wikipedia_articles.json",
    delay: float = 1.0,
    max_articles: int | None = None,
    batch_size: int = 50,
    save_every: int = 500,
):
    """
    Zbiera artykuły z polskiej Wikipedii i zapisuje je
    do pliku JSON jako listę obiektów.

    Obsługuje wznawianie (resume) — jeśli scraper zostanie przerwany,
    ponowne uruchomienie kontynuuje od ostatniego zapisanego stanu.
    Stan jest przechowywany w pliku <output_path>.state.json.

    Każdy artykuł zawiera:
    - id:                    numer porządkowy
    - pageid:                identyfikator strony w Wikipedii
    - title:                 tytuł artykułu
    - url:                   pełny URL do artykułu
    - text:                  treść artykułu (plain text)
    - content_length:        długość artykułu (bajty)
    - last_edited:           data ostatniej modyfikacji
    - number_of_references:  liczba linków zewnętrznych (przybliżenie referencji)
    - categories:            lista kategorii
    - scraped_at:            data scrapingu

    Parametry:
    - output_path:    ścieżka do pliku wynikowego
    - delay:          opóźnienie między partiami (w sekundach)
    - max_articles:   opcjonalny limit artykułów (None = wszystkie)
    - batch_size:     liczba artykułów w jednej partii (max 50)
    - save_every:     zapisuj plik co N artykułów (backup przyrostowy)
    """

    # Próba wznowienia z poprzedniego stanu
    results, ap_continue, total_fetched, batch_num = _load_state(output_path)

    if results:
        print(f"▶ Wznawiam scraping — znaleziono {len(results)} artykułów z poprzedniego uruchomienia")
        print(f"  Kontynuacja od partii {batch_num + 1}, apcontinue={ap_continue}")
    else:
        print("Rozpoczynam scraping polskiej Wikipedii od początku…")

    print(f"  Limit: {max_articles or 'brak (wszystkie)'}")
    print(f"  Batch: {batch_size}, delay: {delay}s")
    print(f"  Output: {output_path}")
    print()

    scraped_at = datetime.now(timezone.utc).isoformat()

    while True:
        batch_num += 1

        # Pobierz partię tytułów
        try:
            pages, ap_continue = fetch_article_batch(ap_continue, limit=batch_size)
        except Exception as e:
            print(f"⚠ Błąd przy pobieraniu listy: {e}")
            time.sleep(delay * 5)
            continue

        if not pages:
            print("Brak kolejnych artykułów — koniec.")
            break

        titles = [p["title"] for p in pages]
        print(f"[Partia {batch_num}] Pobieram {len(titles)} artykułów…", end=" ")

        try:
            details = fetch_article_details(titles, delay=0.3)
            for article in details:
                total_fetched += 1
                article["id"] = total_fetched
                article["scraped_at"] = scraped_at
                results.append(article)

            print(f"OK (razem: {total_fetched})")
        except Exception as e:
            print(f"⚠ Błąd: {e}")

        # Zapis przyrostowy co save_every artykułów + stan
        if total_fetched % save_every < batch_size:
            _save_json(results, output_path)
            _save_state(output_path, ap_continue, total_fetched, batch_num)
            print(f"  💾 Backup: {len(results)} rekordów → {output_path}")

        # Sprawdź limit
        if max_articles and total_fetched >= max_articles:
            print(f"\nOsiągnięto limit {max_articles} artykułów.")
            break

        # Koniec paginacji
        if ap_continue is None:
            print("\nKoniec listy artykułów.")
            break

        time.sleep(delay)

    # Zapis końcowy + usunięcie pliku stanu (scraping zakończony)
    _save_json(results, output_path)
    _delete_state(output_path)
    print(f"\n{'='*60}")
    print(f"Zapisano {len(results)} artykułów do {output_path}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Punkt wejścia
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Domyślnie: 1000 artykułów. Zmień max_articles=None aby pobrać wszystkie.
    # UWAGA: polska Wikipedia ma ~1.5M+ artykułów — pełny scraping trwa wiele godzin.
    # Scraper obsługuje wznawianie — ponowne uruchomienie kontynuuje od ostatniego stanu.
    scrape_all_to_json(max_articles=1000)

