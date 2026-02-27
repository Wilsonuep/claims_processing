import time
import json
import cloudscraper
from bs4 import BeautifulSoup
from datetime import datetime, timezone

BASE = "https://demagog.org.pl"

session = cloudscraper.create_scraper()


# ---------------------------------------------------------------------------
# Ładowanie linków z pliku JSON (output oryginalnego webscrapera)
# ---------------------------------------------------------------------------

def load_urls(json_path: str = "demagog_wypowiedzi_general.json") -> list[dict]:
    """
    Wczytuje plik JSON wygenerowany przez demagog_webscrapper.py
    i zwraca listę obiektów (każdy zawiera co najmniej klucz 'url').
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Pobieranie HTML strony szczegółowej
# ---------------------------------------------------------------------------

def fetch_detail_page(url: str) -> str:
    """Pobiera pełne HTML pojedynczej strony szczegółowej wypowiedzi."""
    r = session.get(url)
    r.raise_for_status()
    return r.text


# ---------------------------------------------------------------------------
# Parsowanie strony szczegółowej
# ---------------------------------------------------------------------------

def parse_detail_page(html: str, source_url: str) -> dict:
    """
    Parsuje stronę szczegółową wypowiedzi i wyciąga istotne dane:
    - title:             tytuł analizy
    - publication_date:  data i godzina publikacji (np. "13.02.2026 godz.17:03")
    - reading_time:      szacowany czas czytania (np. "9 min czytania")
    - category:          kategoria wpisu (np. "Wypowiedzi")
    - person_name:       imię i nazwisko osoby, której wypowiedź sprawdzono
    - person_function:   funkcja / stanowisko osoby (np. "Poseł")
    - person_party:      partia / afiliacja polityczna
    - person_image_url:  URL do zdjęcia osoby
    - statement:         pełna treść sprawdzanej wypowiedzi (cytat)
    - statement_source:  źródło wypowiedzi (tekst, np. "Facebook, 30.01.2026")
    - statement_source_url: link do oryginalnego źródła wypowiedzi
    - rating:            ocena Demagoga (np. "Manipulacja", "Fałsz", "Prawda")
    - tags:              lista tagów/tematów (np. ["Energetyka"])
    - summary:           skrót/podsumowanie kluczowych wniosków
    - full_analysis:     pełna treść analizy fact-checkowej (plain text)
    - detail_url:        URL strony szczegółowej
    """
    soup = BeautifulSoup(html, "html.parser")
    result = {"detail_url": source_url}

    # --- Tytuł ---
    title_el = soup.select_one(
        ".dg-post-content__title--desktop h2, .dg-post-content__title h2"
    )
    result["title"] = title_el.get_text(strip=True) if title_el else None

    # --- Data publikacji i czas czytania ---
    info_el = soup.select_one(".dg-post__info--desktop")
    if not info_el:
        info_el = soup.select_one(".dg-post__info")
    if info_el:
        spans = info_el.select("span")
        result["publication_date"] = spans[0].get_text(strip=True) if len(spans) > 0 else None
        result["reading_time"] = spans[1].get_text(strip=True) if len(spans) > 1 else None
    else:
        result["publication_date"] = None
        result["reading_time"] = None

    # --- Kategoria (np. Wypowiedzi, Fake newsy) ---
    category_el = soup.select_one(".dg-post-content__category span")
    result["category"] = category_el.get_text(strip=True) if category_el else None

    # --- Osoba (sprawdzana) ---
    person_block = soup.select_one(".dg-post-person")
    if person_block:
        name_el = person_block.select_one(".dg-post-person__desc h4 a")
        result["person_name"] = name_el.get_text(strip=True) if name_el else None

        func_el = person_block.select_one(".dg-post-person__function")
        result["person_function"] = func_el.get_text(strip=True) if func_el else None

        party_el = person_block.select_one(".dg-post-person__party")
        result["person_party"] = party_el.get_text(strip=True) if party_el else None

        img_el = person_block.select_one(".dg-post-person__image img")
        result["person_image_url"] = img_el.get("src") if img_el else None
    else:
        result["person_name"] = None
        result["person_function"] = None
        result["person_party"] = None
        result["person_image_url"] = None

    # --- Cytat / sprawdzana wypowiedź ---
    quote_el = soup.select_one(".dg-post-quote__statement")
    if quote_el:
        # Usuwamy ikonę cytatu (img) i zostawiamy sam tekst
        for img in quote_el.find_all("img"):
            img.decompose()
        result["statement"] = quote_el.get_text(strip=True)
    else:
        result["statement"] = None

    # --- Źródło wypowiedzi ---
    source_el = soup.select_one(
        ".dg-post-quote__footer--desktop .dg-post-quote__source"
    )
    if not source_el:
        source_el = soup.select_one(".dg-post-quote__source")
    if source_el:
        result["statement_source"] = source_el.get_text(strip=True)
        source_link = source_el.select_one("a")
        result["statement_source_url"] = source_link.get("href") if source_link else None
    else:
        result["statement_source"] = None
        result["statement_source_url"] = None

    # --- Ocena (rating) ---
    eval_el = soup.select_one(
        ".dg-post-quote__footer--desktop .dg-post-quote__evaluation p"
    )
    if not eval_el:
        eval_el = soup.select_one(".dg-post-quote__evaluation p")
    result["rating"] = eval_el.get_text(strip=True) if eval_el else None

    # --- Tagi / tematy ---
    tags_container = soup.select_one(
        ".dg-post-content__tags--desktop"
    )
    if not tags_container:
        tags_container = soup.select_one(".dg-post-content__tags")
    if tags_container:
        result["tags"] = [
            a.get_text(strip=True) for a in tags_container.select("a")
        ]
    else:
        result["tags"] = []

    # --- Podsumowanie (summary / kluczowe wnioski) ---
    summary_el = soup.select_one(".summary-text")
    if summary_el:
        result["summary"] = summary_el.get_text("\n", strip=True)
    else:
        result["summary"] = None

    # --- Pełna treść analizy ---
    # Treść artykułu mieści się w .dg-post-content__inner
    # (może być kilka takich bloków – łączymy je)
    content_blocks = soup.select(".dg-post-content__inner")
    full_text_parts = []
    for block in content_blocks:
        text = block.get_text("\n", strip=True)
        if text:
            full_text_parts.append(text)
    result["full_analysis"] = "\n\n".join(full_text_parts) if full_text_parts else None

    return result


# ---------------------------------------------------------------------------
# Główna funkcja: scraping szczegółów
# ---------------------------------------------------------------------------

def scrape_details(
    input_path: str = "demagog_wypowiedzi_general.json",
    output_path: str = "demagog_wypowiedzi_detailed.json",
    delay: float = 1.0,
    limit: int | None = None,
):
    """
    Iteruje po linkach z pliku general JSON i dla każdego pobiera
    stronę szczegółową, parsuje dane i zapisuje wynik do nowego pliku JSON.

    Parametry:
    - input_path:  ścieżka do pliku z linkami (output demagog_webscrapper.py)
    - output_path: ścieżka do pliku wynikowego ze szczegółami
    - delay:       opóźnienie między requestami (w sekundach)
    - limit:       opcjonalny limit rekordów (None = wszystkie)
    """
    general_data = load_urls(input_path)
    urls = [item["url"] for item in general_data if item.get("url")]

    if limit:
        urls = urls[:limit]

    print(f"Załadowano {len(urls)} linków do pobrania szczegółów")

    results = []
    scraped_at = datetime.now(timezone.utc).isoformat()

    for i, url in enumerate(urls, start=1):
        print(f"[{i}/{len(urls)}] Pobieram: {url}")
        try:
            html = fetch_detail_page(url)
            detail = parse_detail_page(html, source_url=url)
            detail["id"] = i
            detail["scraped_at"] = scraped_at
            results.append(detail)
        except Exception as e:
            print(f"  ⚠ Błąd: {e}")
            results.append({
                "id": i,
                "detail_url": url,
                "scraped_at": scraped_at,
                "error": str(e),
            })

        time.sleep(delay)

    # Zapis wyników
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok_count = sum(1 for r in results if "error" not in r)
    err_count = sum(1 for r in results if "error" in r)
    print(f"\nZapisano {ok_count} rekordów do {output_path} ({err_count} błędów)")


# ---------------------------------------------------------------------------
# Punkt wejścia
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scrape_details()
