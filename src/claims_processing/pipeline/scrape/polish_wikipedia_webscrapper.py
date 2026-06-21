import time
import json
import sys
import os
import requests
from datetime import datetime, timezone
from urllib.parse import quote

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


API_URL = "https://pl.wikipedia.org/w/api.php"

USER_AGENT = (
    "PolishWikipediaScraper/1.0 "
    "(claims_processing project; contact: piotr.wilma@hotmail.com) "
    "python-requests/2.x"
)

MAXLAG = 5
REQUEST_TIMEOUT = 30
MIN_DELAY = 1.0

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


# ---------------------------------------------------------------------------
# Core HTTP layer — all retry/backoff/maxlag logic lives here ONLY
# ---------------------------------------------------------------------------

def _request_get(params: dict, retries: int = 6) -> dict:
    """
    Single GET to the API. Handles:
      - maxlag (503 + Retry-After, or JSON error.code == 'maxlag')
      - 429 Too Many Requests + Retry-After
      - network errors (ConnectionError, Timeout) — exponential backoff
      - HTTP errors — exponential backoff
      - API-level errors in JSON — raises RuntimeError immediately
        (no retry, these are logical errors like bad params)

    All callers can assume: if this returns, data is valid.
    If it raises, the error is unrecoverable after all retries.
    """
    params = {**params, "maxlag": MAXLAG}

    for attempt in range(1, retries + 1):
        try:
            r = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)

            # Maxlag throttle from server
            if r.status_code == 503:
                wait = float(r.headers.get("Retry-After", 30))
                print(f"    ⏳ maxlag 503 — czekam {wait:.0f}s (próba {attempt}/{retries})")
                time.sleep(wait)
                continue

            # Rate limit
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 60))
                print(f"    ⏳ rate limit 429 — czekam {wait:.0f}s (próba {attempt}/{retries})")
                time.sleep(wait)
                continue

            r.raise_for_status()
            data = r.json()

            # Maxlag returned as JSON error (some MW versions)
            if "error" in data:
                err = data["error"]
                if err.get("code") == "maxlag":
                    wait = float(err.get("lag", MAXLAG)) + 1
                    print(f"    ⏳ maxlag JSON — lag={err.get('lag')}s, czekam {wait:.0f}s (próba {attempt}/{retries})")
                    time.sleep(wait)
                    continue
                # Any other API error is a programming/logic error — don't retry
                raise RuntimeError(f"API error [{err.get('code')}]: {err.get('info', err)}")

            return data

        except RuntimeError:
            raise  # Programming errors bubble up immediately

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt == retries:
                raise
            wait = min(2 ** attempt, 120)
            print(f"    ⏳ błąd sieci — czekam {wait}s (próba {attempt}/{retries})")
            time.sleep(wait)

        except requests.exceptions.HTTPError:
            if attempt == retries:
                raise
            wait = min(2 ** attempt, 120)
            print(f"    ⏳ HTTP {r.status_code} — czekam {wait}s (próba {attempt}/{retries})")
            time.sleep(wait)

    raise RuntimeError(f"Nie udało się wykonać zapytania po {retries} próbach.")


# ---------------------------------------------------------------------------
# Fetch page title list via allpages
# ---------------------------------------------------------------------------

def fetch_article_batch(ap_continue: str | None = None,
                        limit: int = 50) -> tuple[list[dict], str | None]:
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": 0,
        "aplimit": limit,
        "apfilterredir": "nonredirects",
        "format": "json",
        "formatversion": 2,
    }
    if ap_continue:
        params["apcontinue"] = ap_continue

    data = _request_get(params)
    pages = data.get("query", {}).get("allpages", [])
    cont = data.get("continue", {}).get("apcontinue")
    return pages, cont


# ---------------------------------------------------------------------------
# Generic paginated query for a batch of titles
# ---------------------------------------------------------------------------

def _query_batch(titles: list[str], **extra_params) -> list[dict]:
    """
    Runs action=query for given titles, merging all continuation pages.
    No exception handling here — errors propagate to the caller.
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
        data = _request_get(params)

        for p in data.get("query", {}).get("pages", []):
            pid = p.get("pageid")
            if pid is None or p.get("missing"):
                continue
            if pid in all_pages:
                for key, val in p.items():
                    if key in ("pageid", "title", "ns"):
                        continue
                    if isinstance(val, list) and key in all_pages[pid]:
                        all_pages[pid][key].extend(val)
                    else:
                        all_pages[pid][key] = val
            else:
                all_pages[pid] = p

        if "continue" not in data:
            break

        params.update(data["continue"])
        time.sleep(0.5)  # Small pause between continuation pages

    return list(all_pages.values())


# ---------------------------------------------------------------------------
# Fetch full article details — 3 batched API calls
# ---------------------------------------------------------------------------

def fetch_article_details(titles: list[str], delay: float = MIN_DELAY) -> list[dict]:
    """
    Makes 3 sequential batch API calls for the given titles:
      1. extracts + info  — article text, length, last edit date
      2. extlinks         — external links count (≈ references)
      3. categories       — article categories

    Raises on any failure — no silent data loss.
    """
    delay = max(delay, MIN_DELAY)

    # 1. Text + metadata
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
        title = p.get("title", "")
        page_map[pid] = {
            "pageid": pid,
            "title": title,
            "url": (
                "https://pl.wikipedia.org/wiki/"
                + quote(title.replace(" ", "_"), safe="/:@!$&()*+,;=")
            ),
            "text": p.get("extract"),
            "content_length": p.get("length"),
            "last_edited": p.get("touched"),
            "number_of_references": 0,
            "categories": [],
        }

    time.sleep(delay)

    # 2. External links
    pages_extlinks = _query_batch(titles, prop="extlinks", ellimit="max")
    for p in pages_extlinks:
        pid = p.get("pageid")
        if pid and pid in page_map:
            page_map[pid]["number_of_references"] = len(p.get("extlinks", []))

    time.sleep(delay)

    # 3. Categories
    pages_cats = _query_batch(titles, prop="categories", cllimit="max")
    for p in pages_cats:
        pid = p.get("pageid")
        if pid and pid in page_map:
            page_map[pid]["categories"] = [
                c["title"].replace("Kategoria:", "").strip()
                for c in p.get("categories", [])
            ]

    return list(page_map.values())


# ---------------------------------------------------------------------------
# JSONL output + atomic state management
# ---------------------------------------------------------------------------

def _save_jsonl_append(data: list[dict], path: str):
    """Atomicznie dopisuje artykuły do JSONL — buduje cały blok w pamięci,
    potem robi jeden write(), co minimalizuje ryzyko częściowego zapisu."""
    block = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in data)
    with open(path, "a", encoding="utf-8") as f:
        f.write(block)
        f.flush()
        os.fsync(f.fileno())


def _state_path(output_path: str) -> str:
    return output_path + ".state.json"


def _save_state(output_path: str, ap_continue: str | None,
                total_fetched: int, batch_num: int,
                seen_pageids: set[int]):
    """Atomic write — prevents corrupt state file on interrupted save."""
    state = {
        "apcontinue": ap_continue,
        "total_fetched": total_fetched,
        "batch_num": batch_num,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "seen_pageids": sorted(seen_pageids),  # persist the dedup set
    }
    tmp = _state_path(output_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, _state_path(output_path))


def _load_state(output_path: str) -> tuple[str | None, int, int, set[int]]:
    sp = _state_path(output_path)
    if not os.path.exists(sp):
        return None, 0, 0, set()
    with open(sp, "r", encoding="utf-8") as f:
        state = json.load(f)
    return (
        state.get("apcontinue"),
        state.get("total_fetched", 0),
        state.get("batch_num", 0),
        set(state.get("seen_pageids", [])),
    )


def _delete_state(output_path: str):
    sp = _state_path(output_path)
    if os.path.exists(sp):
        os.remove(sp)


# ---------------------------------------------------------------------------
# Main scraper loop
# ---------------------------------------------------------------------------

def scrape_all_to_json(
    output_path: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "polish_wikipedia_articles.jsonl",
    ),
    delay: float = MIN_DELAY,
    max_articles: int | None = None,
    batch_size: int = 50,
    save_every: int = 500,
):
    delay = max(delay, MIN_DELAY)
    batch_size = min(batch_size, 50)

    ap_continue, total_fetched, batch_num, seen_pageids = _load_state(output_path)

    if total_fetched > 0:
        print(f"▶ Wznawiam od partii {batch_num + 1} | pobrano: {total_fetched} | apcontinue={ap_continue!r}")
        print(f"  Znanych pageid: {len(seen_pageids)} (zostaną pominięte)")
    else:
        if os.path.exists(output_path):
            os.remove(output_path)
        print("▶ Rozpoczynam scraping polskiej Wikipedii od początku…")

    print(f"  Limit    : {max_articles or 'brak (wszystkie)'}")
    print(f"  Batch    : {batch_size} | Delay: {delay}s | Maxlag: {MAXLAG}s")
    print(f"  Output   : {output_path}\n")

    scraped_at = datetime.now(timezone.utc).isoformat()
    dupes_skipped = 0

    while True:
        batch_num += 1

        # Fetch title list — retry handled inside _request_get
        pages, ap_continue = fetch_article_batch(ap_continue, limit=batch_size)

        if not pages:
            if ap_continue is None:
                print("\n✓ Koniec listy artykułów.")
                break
            print(f"[Partia {batch_num}] Pusta partia, kontynuuję…")
            time.sleep(delay)
            continue

        titles = [p["title"] for p in pages]
        print(f"[Partia {batch_num:>5}] {len(titles):>2} artykułów… ", end="", flush=True)

        # Fetch details — also retried internally
        details = fetch_article_details(titles, delay=delay)

        # --- Deduplikacja na poziomie scrapera ---
        new_details: list[dict] = []
        for article in details:
            pid = article["pageid"]
            if pid in seen_pageids:
                dupes_skipped += 1
                continue
            seen_pageids.add(pid)
            total_fetched += 1
            article["id"] = total_fetched
            article["scraped_at"] = scraped_at
            new_details.append(article)

        if new_details:
            _save_jsonl_append(new_details, output_path)

        skipped_msg = f"  [{dupes_skipped} dup]" if dupes_skipped else ""
        print(f"OK  (nowe: {len(new_details)}, razem: {total_fetched}){skipped_msg}")

        # Save state AFTER every successful write (not on a flaky modulo condition)
        if batch_num % (save_every // batch_size or 1) == 0:
            _save_state(output_path, ap_continue, total_fetched, batch_num, seen_pageids)
            print(f"  💾 Stan zapisany ({total_fetched} rekordów, {len(seen_pageids)} uniq pageids)")

        if max_articles and total_fetched >= max_articles:
            print(f"\n✓ Osiągnięto limit {max_articles} artykułów.")
            break

        if ap_continue is None:
            print("\n✓ Koniec listy artykułów.")
            break

        time.sleep(delay)

    # Final state save before cleanup
    _save_state(output_path, ap_continue, total_fetched, batch_num, seen_pageids)
    _delete_state(output_path)
    print(f"\n{'='*60}")
    print(f"  Zapisano {total_fetched} artykułów → {output_path}")
    if dupes_skipped:
        print(f"  Pominięto {dupes_skipped} duplikatów (po pageid)")
    print(f"{'='*60}")


if __name__ == "__main__":
    scrape_all_to_json(max_articles=None)