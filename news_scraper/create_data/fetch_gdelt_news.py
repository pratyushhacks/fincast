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
from pathlib import Path

from dotenv import load_dotenv

from scraper import save_article
from utils import get_watchlist, get_trusted_sources
from datetime import datetime, timedelta
import zoneinfo
import random

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv(dotenv_path="../.env")

DATA_DIR     = os.getenv("DATA_DIR",    "data/raw")
GDELT_API    = os.getenv("GDELT_API",         "http://api.gdeltproject.org/api/v2/doc/doc")
DEFAULT_DAYS = int(os.getenv("GDELT_DEFAULT_DAYS", 30))
GDELT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}
GDELT_INTER_REQUEST_SLEEP = float(os.getenv("GDELT_INTER_REQUEST_SLEEP_SECONDS", "8"))
TRUSTED_SOURCES = get_trusted_sources()

RETRY_COOLDOWN_SECONDS = int(os.getenv("GDELT_RETRY_COOLDOWN_SECONDS", "900"))  # 15 minutes, to avoid repeated rate-limiting
MAX_OUTER_RETRY_PASSES = int(os.getenv("GDELT_MAX_OUTER_RETRIES", "5"))

def is_trusted(url):
    if not TRUSTED_SOURCES:
        return True
    try:
        host = urlparse(url).netloc.lower()
        host = host[4:] if host.startswith("www.") else host
        return any(host == s or host.endswith("." + s) for s in TRUSTED_SOURCES)
    except Exception:
        return False


def fetch_urls_for_query_by_range(query, start_dt, end_dt) -> list[dict]:
    """Fetch one query across a single date window using the GDELT date-range API."""
    if end_dt < start_dt:
        raise ValueError(f"end_dt must be >= start_dt for query '{query}'")

    print(f"  Fetching {query} from {start_dt.date()} to {end_dt.date()} in one range...")
    articles = fetch_urls_for_query(query, start_dt=start_dt, end_dt=end_dt, retries=1)

    filtered_articles = [a for a in articles if a.get("url") and is_trusted(a["url"])]
    for article in filtered_articles:
        article["query"] = query
        article["fetch_date"] = start_dt.isoformat()

        if not any(article.get(k) for k in ("published", "updated", "seendate", "date")):
            article["seendate"] = end_dt.isoformat()

    return filtered_articles


def fetch_urls_for_query(query, start_dt, end_dt, retries=5) -> list[dict]:
    days_span = max(1, (end_dt - start_dt).days + 1)
    base_cap = int(os.getenv("GDELT_MAX_RECORDS_BASE", "50"))
    maxrecords = max(base_cap, int((days_span / 7.0) * base_cap))
    maxrecords = min(maxrecords, int(os.getenv("GDELT_MAX_RECORDS_CAP", "250")))

    params = {
        "query":         query,
        "mode":          "artlist",
        "maxrecords":    maxrecords,
        "format":        "json",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   end_dt.strftime("%Y%m%d%H%M%S"),
        "sourcelang":    "english",
        "sourcecountry": os.getenv("GDELT_SOURCE_COUNTRIES", "US,UK,CA,AS,SN,HK,IN,JA,KS"),
    }

    # print(f"fetching news with following params: {params}")

    for attempt in range(retries):
        try:
            time.sleep(GDELT_INTER_REQUEST_SLEEP)
            r = requests.get(GDELT_API, params=params, timeout=120, verify=False, headers=GDELT_HEADERS)

            if r.status_code == 429:
                wait = 300 # 5 minutes
                print(f"  [{query}] Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if r.status_code != 200:
                print(f"  [{query}] HTTP {r.status_code}: {r.text.strip()}")
                return []

            try:
                data = r.json()
            except json.JSONDecodeError as e:
                cleaned = r.content.decode('utf-8', errors='ignore')
                print(f"  [{query}] JSON decode failed, raw response: {cleaned[:500]}")
                cleaned = re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', cleaned)
                try:
                    data = json.loads(cleaned)
                except json.JSONDecodeError as e2:
                    print(f"  [{query}] Cleaned JSON also failed: {e2}")
                    raise

            articles = data.get("articles", [])
            print(f"  [{query}] {start_dt.date()} -> {len(articles)} articles")
            return articles

        except requests.exceptions.Timeout:
            timeout_wait = min(GDELT_INTER_REQUEST_SLEEP * (attempt + 1), 120)
            print(f"  [{query}] Timeout (attempt {attempt+1}/{retries}), waiting {timeout_wait}s...")
            time.sleep(timeout_wait)
        except Exception as e:
            print(f"  [{query}] ERROR: {type(e).__name__}: {e}")
            return []

    print(f"  [{query}] Failed after {retries} retries")
    return []



def collect_all_urls(days, companies, sectors, macro, now):
    """Collect one GDELT date-range result per watchlist term."""
    if days <= 0:
        return []

    start_dt = (now - timedelta(days=days - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    all_articles = []
    company_queries = [(q, "company") for q in companies]
    sector_queries = [(q, "sector") for q in sectors]
    macro_queries = [(q, "macro") for q in macro]

    # Shuffle within each category, then append categories in a stable order.
    random.shuffle(company_queries)
    random.shuffle(sector_queries)
    random.shuffle(macro_queries)
    all_queries = company_queries + sector_queries + macro_queries
    
    print(f"Querying {len(all_queries)} watchlist items... for {days} days "
          f"(anchored to {now.isoformat()}) using range queries\n")

    failed_queries = []
    for query, category in all_queries:
        print(f"  [{category}] {query}")
        articles = fetch_urls_for_query_by_range(query, start_dt=start_dt, end_dt=now)
        if not articles:
            failed_queries.append((query, category))
        all_articles.extend(articles)
        time.sleep(GDELT_INTER_REQUEST_SLEEP)

    outer_pass = 0
    while failed_queries and outer_pass < MAX_OUTER_RETRY_PASSES:
        outer_pass += 1
        print(f"\nCooling down {RETRY_COOLDOWN_SECONDS}s before retry pass "
              f"{outer_pass}/{MAX_OUTER_RETRY_PASSES} ({len(failed_queries)} failed queries)...")
        time.sleep(RETRY_COOLDOWN_SECONDS)

        pending_failures = []
        for query, category in failed_queries:
            print(f"  [RETRY {outer_pass}/{MAX_OUTER_RETRY_PASSES}] {query}")
            articles = fetch_urls_for_query_by_range(
                query, start_dt=start_dt, end_dt=now
            )
            if not articles:
                pending_failures.append((query, category))
            all_articles.extend(articles)
            time.sleep(GDELT_INTER_REQUEST_SLEEP)
        failed_queries = pending_failures

    if failed_queries:
        print(f"\n[WARN] {len(failed_queries)} queries still failed after "
              f"{MAX_OUTER_RETRY_PASSES} retry pass(es): "
              f"{[q for q, _ in failed_queries]}")

    seen, deduped = set(), []
    for article in all_articles:
        url = article.get("url")
        if url and url not in seen:
            seen.add(url)
            deduped.append(article)

    return deduped


def fetch_and_save(article):
    try:
        base = Path(__file__).resolve().parent.parent # /news_scraper folder
        data_dir_path = os.path.join(str(base), "data", "raw") # /news_scraper/data/raw
        os.makedirs(data_dir_path, exist_ok=True)

        url = article.get("url")
        query = article.get("query")
        meta = save_article(url, feed_entry=article, gdelt_query=query, data_dir_path=data_dir_path)
        return meta, None
    except Exception as e:
        return None, str(e)


def main(days=DEFAULT_DAYS, limit=None):
    run_anchor = datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
    print(f"Run anchor time: {run_anchor.isoformat()}")

    companies, sectors, macro = get_watchlist()
    print(
        f"Watchlist: {len(companies)} companies, "
        f"{len(sectors)} sectors, "
        f"{len(macro)} macro topics\n"
    )

    urls = collect_all_urls(days, companies, sectors, macro, now=run_anchor)
    if limit:
        urls = urls[:limit]

    print(f"\nTotal unique URLs to fetch: {len(urls)}")
    print("=" * 60)

    fetched, skipped, errors = 0, 0, 0
    for idx, article in enumerate(urls, 1):
        url = article.get("url")
        meta, err = fetch_and_save(article)
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
    args = parser.parse_args()
    main(days=args.days, limit=args.limit)