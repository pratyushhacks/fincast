r"""
fetch_gdelt_news.py — fetches article URLs from GDELT DOC API using watchlist from gdelt_config.yaml.

Usage:
  python fetch_gdelt_news.py --days 1          # nightly run
  python fetch_gdelt_news.py --days 30         # bootstrap
  python fetch_gdelt_news.py --days 30 --limit 5000
"""
import re
import json
import os
import time
import argparse

import requests
import urllib3
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from scraper import save_article
from utils import get_watchlist, get_trusted_sources

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

DATA_DIR     = os.getenv("GDELT_DATA_DIR",    "data/backfill/gdelt")
GDELT_API    = os.getenv("GDELT_API",         "http://api.gdeltproject.org/api/v2/doc/doc")
DEFAULT_DAYS = int(os.getenv("GDELT_DEFAULT_DAYS", 30))

TRUSTED_SOURCES = get_trusted_sources()


def is_trusted(url):
    if not TRUSTED_SOURCES:
        return True
    try:
        host = urlparse(url).netloc.lower()
        host = host[4:] if host.startswith("www.") else host
        return any(host == s or host.endswith("." + s) for s in TRUSTED_SOURCES)
    except Exception:
        return False


def fetch_urls_for_query(query, timespan_days, retries=5):
    params = {
        "query":         query,
        "mode":          "artlist",
        "maxrecords":    250,
        "format":        "json",
        "timespan":      f"{timespan_days}d",
        "sourcelang":    "english",
        "sourcecountry": os.getenv("GDELT_SOURCE_COUNTRIES", "US,GB,CA,AU,SN,HK,IN,JA,KS"),
    }

    for attempt in range(retries):
        try:
            r = requests.get(GDELT_API, params=params, timeout=60, verify=False)

            if r.status_code == 429:
                wait = min(15 * (attempt + 1), 60)
                print(f"  [{query}] Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if r.status_code != 200:
                print(f"  [{query}] HTTP {r.status_code}: {r.text.strip()}")
                return []

            try:
                data = r.json()
            except Exception:
                cleaned = r.content.decode('utf-8', errors='ignore')
                cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned)
                try:
                    data = json.loads(cleaned)
                except Exception as e:
                    print(f"  [{query}] Malformed JSON: {e}")
                    return []

            articles = data.get("articles", [])
            urls = [a["url"] for a in articles if "url" in a]
            urls = [u for u in urls if is_trusted(u)]
            print(f"  [{query}] -> {len(urls)} articles after source filter")
            return urls

        except requests.exceptions.Timeout:
            wait = min(10 * (attempt + 1), 60)
            print(f"  [{query}] Timeout (attempt {attempt+1}/{retries}), waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  [{query}] ERROR: {type(e).__name__}: {e}")
            return []

    print(f"  [{query}] Failed after {retries} retries")
    return []


def collect_all_urls(days, companies, sectors, macro):
    all_urls    = []
    all_queries = (
        [(q, "company") for q in companies] +
        [(q, "sector")  for q in sectors]   +
        [(q, "macro")   for q in macro]
    )

    print(f"Querying {len(all_queries)} watchlist items...\n")

    failed_queries = []
    for query, category in all_queries:
        print(f"  [{category}] {query}")
        urls = fetch_urls_for_query(query, timespan_days=days)
        if not urls:
            failed_queries.append((query, category))
        for url in urls:
            all_urls.append((url, query))
        time.sleep(2)

    # Retry failed queries once at the end
    if failed_queries:
        print(f"\nRetrying {len(failed_queries)} failed queries...")
        for query, category in failed_queries:
            print(f"  [RETRY] {query}")
            urls = fetch_urls_for_query(query, timespan_days=days, retries=10)
            for url in urls:
                all_urls.append((url, query))
            time.sleep(2)

    # Deduplicate preserving order
    seen, deduped = set(), []
    for url, query in all_urls:
        if url not in seen:
            seen.add(url)
            deduped.append((url, query))

    return deduped


def fetch_and_save(args):
    url, query = args
    try:
        meta = save_article(url, feed_entry=None, gdelt_query=query)
        return meta, None
    except Exception as e:
        return None, str(e)


def main(days=DEFAULT_DAYS, limit=None, sleep=0.0, workers=5):
    os.makedirs(DATA_DIR, exist_ok=True)

    companies, sectors, macro = get_watchlist()
    print(
        f"Watchlist: {len(companies)} companies, "
        f"{len(sectors)} sectors, "
        f"{len(macro)} macro topics\n"
    )

    urls = collect_all_urls(days, companies, sectors, macro)
    if limit:
        urls = urls[:limit]

    print(f"\nTotal unique URLs to fetch: {len(urls)}")
    print("=" * 60)

    fetched, skipped, errors = 0, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_and_save, item): item for item in urls}
        for idx, future in enumerate(as_completed(futures), 1):
            url, query = futures[future]
            meta, err  = future.result()
            if err:
                print(f"  [ERROR {idx}/{len(urls)}] {url}: {err}")
                errors += 1
            elif meta:
                print(f"  [OK    {idx}/{len(urls)}] {url}")
                fetched += 1
            else:
                print(f"  [SKIP  {idx}/{len(urls)}] {url}")
                skipped += 1

    print(f"\nDone: {fetched} saved, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch news via GDELT DOC API")
    parser.add_argument("--days",    type=int,   default=DEFAULT_DAYS)
    parser.add_argument("--limit",   type=int,   default=None)
    parser.add_argument("--sleep",   type=float, default=0.0)
    parser.add_argument("--workers", type=int,   default=5)
    args = parser.parse_args()
    main(days=args.days, limit=args.limit, sleep=args.sleep, workers=args.workers)
