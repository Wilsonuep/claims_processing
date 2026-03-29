import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from datascrap.polish_wikipedia_webscrapper import fetch_article_batch, fetch_article_details, _query_batch

total = 0
ap_cont = None
for i in range(1, 5):
    print(f"Fetching batch {i} with ap_continue={ap_cont}...")
    try:
        pages, ap_cont = fetch_article_batch(ap_cont, limit=50)
        print(f"Got {len(pages)} titles, NEXT apcontinue={ap_cont}")
        total += len(pages)
        if not pages:
            print("0 pages in this batch, but continue is present!")
            continue
        titles = [p["title"] for p in pages]
        details = fetch_article_details(titles, delay=0)
        print(f"Fetched {len(details)} details")
    except Exception as e:
        print("EXCEPTION:", e)
        break
print(f"Total loop fetched: {total}")
