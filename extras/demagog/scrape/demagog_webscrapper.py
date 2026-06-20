import time
import json
import sys
import cloudscraper
from bs4 import BeautifulSoup

# Reconfigure stdout for UTF-8 on Windows (prevents UnicodeEncodeError
# when printing emoji/special chars in cmd.exe / PowerShell)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
from urllib.parse import urljoin
from datetime import datetime, timezone

BASE = "https://demagog.org.pl"
START_URL = f"{BASE}/wypowiedzi/"
AJAX_URL = f"{BASE}/wp-admin/admin-ajax.php"

session = cloudscraper.create_scraper()


# ---------------------------------------------------------------------------
# Ustalanie liczby stron archiwum
# ---------------------------------------------------------------------------

def get_max_pages() -> int:
    """
    Pobiera pierwszą stronę i odczytuje atrybut data-max-pages
    z kontenera kafelków. Alternatywnie AJAX na stronie 2 zwraca
    klucz 'max' w odpowiedzi JSON.
    """
    r = session.get(START_URL)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    tiles = soup.find(attrs={"data-max-pages": True})
    if tiles:
        return int(tiles["data-max-pages"])

    # Fallback: pobierz stronę 2 z AJAX i odczytaj 'max' z JSON
    data = {"action": "statements_filter", "paged": 2,
            "wybory_page": "no", "wybory_eu_page": "no"}
    r2 = session.post(AJAX_URL, data=data)
    r2.raise_for_status()
    return int(r2.json()["max"])


# ---------------------------------------------------------------------------
# Pobieranie HTML strony listingowej
# ---------------------------------------------------------------------------

def fetch_list_page(page: int) -> str:
    """
    Zwraca HTML listy wypowiedzi dla danej „strony":
    - page == 1: pełne HTML z /wypowiedzi/
    - page > 1:  AJAX zwraca JSON {"max":…, "html":"…"} → zwracamy html
    """
    if page == 1:
        r = session.get(START_URL)
        r.raise_for_status()
        return r.text

    data = {
        "action": "statements_filter",
        "paged": page,
        "wybory_page": "no",
        "wybory_eu_page": "no",
    }
    r = session.post(AJAX_URL, data=data)
    r.raise_for_status()

    # AJAX zwraca JSON z kluczem "html" zawierającym fragment HTML
    try:
        return r.json()["html"]
    except (ValueError, KeyError):
        # Fallback: jeśli format się zmienił, zwróć surowy tekst
        return r.text


# ---------------------------------------------------------------------------
# Parsowanie strony listingowej
# ---------------------------------------------------------------------------

def parse_list_page(html: str):
    """
    Parsuje pojedynczą „stronę" (pełną lub doładowaną) i zwraca słowniki
    z title, url, person, rating, snippet, tags.

    Filtruje duplikaty desktop/mobile — bierze tylko wersje desktop
    (lub elementy bez modyfikatora, np. fake_news na stronie głównej).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Bierzemy tylko .dg-item--desktop (pomijamy .dg-item--mobile)
    # Na stronach AJAX mogą nie mieć modyfikatora — wtedy bierzemy wszystkie .dg-item
    items = soup.select(".dg-item--desktop")
    if not items:
        items = soup.select(".dg-item")

    seen_urls = set()

    for it in items:
        # --- Tytuł i URL ---
        title_el = it.select_one(".dg-item__title a, .dg-itemtitle a")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        url_rel = title_el.get("href")
        url_abs = urljoin(BASE, url_rel) if url_rel else None

        # Deduplikacja po URL (na wypadek gdyby desktop/mobile obie pasowały)
        if url_abs in seen_urls:
            continue
        seen_urls.add(url_abs)

        # --- Osoba ---
        person_el = it.select_one(".dg-item__person a, .dg-itemperson a")
        person = person_el.get_text(strip=True) if person_el else None

        # --- Ocena ---
        rating_el = it.select_one(".dg-item__evaluation p, .dg-itemevaluation p")
        rating = rating_el.get_text(strip=True) if rating_el else None

        # --- Skrót opisu ---
        snippet_el = it.select_one(".dg-item__description, .dg-itemdescription")
        snippet = snippet_el.get_text(strip=True) if snippet_el else None

        # --- Tagi (tematy) ---
        tags_container = it.select_one(".dg-item__tags, .dg-itemtags")
        if tags_container:
            tags = [a.get_text(strip=True) for a in tags_container.select("a")]
        else:
            tags = []

        yield {
            "title": title,
            "url": url_abs,
            "person": person,
            "rating": rating,
            "snippet": snippet,
            "tags": tags,
        }


# ---------------------------------------------------------------------------
# Główna funkcja: scraping listy wypowiedzi
# ---------------------------------------------------------------------------

def scrape_all_to_json(
    output_path: str = "demagog_wypowiedzi_general.json",
    delay: float = 0.5,
):
    """
    Zbiera wszystkie wypowiedzi z archiwum /wypowiedzi/ i zapisuje je
    do jednego pliku JSON jako listę obiektów.

    Parametry:
    - output_path: ścieżka do pliku wynikowego
    - delay:       opóźnienie między requestami (w sekundach)
    """
    max_pages = get_max_pages()
    print(f"Znaleziono {max_pages} stron archiwum wypowiedzi")

    results = []
    scraped_at = datetime.now(timezone.utc).isoformat()

    for page in range(1, max_pages + 1):
        print(f"[{page}/{max_pages}] Scraping strony…", end=" ")
        try:
            html = fetch_list_page(page)
            count_before = len(results)
            for item in parse_list_page(html):
                item_with_meta = {
                    "id": len(results) + 1,
                    "scraped_at": scraped_at,
                    **item,
                }
                results.append(item_with_meta)
            print(f"OK (+{len(results) - count_before} rekordów, razem: {len(results)})")
        except Exception as e:
            print(f"⚠ Błąd: {e}")

        time.sleep(delay)

    # Zapis wyników
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nZapisano {len(results)} rekordów do {output_path}")


# ---------------------------------------------------------------------------
# Punkt wejścia
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scrape_all_to_json()
