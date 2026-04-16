r"""
fetch_prices.py — fetches stock price trend and next-day bucket for labeled articles.

For each article:
  - trend:      7 trading day closes BEFORE article date (input context)
  - base_price: article day close
  - next_price: next trading day close
  - change_1d:  % change (reference only)
  - bucket_1d:  prediction label (strong_down/down/flat/up/strong_up)
  - magnitude:  size of move (minimal/small/moderate/large)

Usage:
  python fetch_prices.py --data-dir data/raw
"""
import os
import json
import argparse
import time
from datetime import datetime, timedelta

import yfinance as yf

from utils import get_tickers, get_benchmarks, get_trusted_sources, get_private_companies

ALL_TICKERS       = get_tickers()
BENCHMARKS        = get_benchmarks()
TRUSTED_SOURCES   = get_trusted_sources()
PRIVATE_COMPANIES = get_private_companies()

_price_cache = {}


def get_ticker(gdelt_query):
    return ALL_TICKERS.get(gdelt_query)


def price_bucket(pct):
    if pct is None: return None
    if pct <= -5:   return "strong_down"
    if pct <= -2:   return "down"
    if pct <   2:   return "flat"
    if pct <   5:   return "up"
    return                 "strong_up"


def magnitude_bucket(pct):
    if pct is None: return None
    pct = abs(pct)
    if pct < 1:   return "minimal"
    if pct < 2:   return "small"
    if pct < 5:   return "moderate"
    return              "large"


def get_price_data(ticker, article_date_str, trend_days=7):
    if not ticker:
        return None

    cache_key = (ticker, article_date_str)
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    try:
        article_date = datetime.fromisoformat(article_date_str).date()
    except Exception:
        return None

    try:
        start = article_date - timedelta(days=trend_days + 7)
        end   = article_date + timedelta(days=7)

        df = yf.download(ticker, start=str(start), end=str(end),
                         progress=False, auto_adjust=True)
        if df.empty:
            return None

        df.sort_index(inplace=True)
        dates  = [d.date() for d in df.index]
        closes = [round(float(x), 2) for x in df['Close'].squeeze().tolist()]

        if article_date not in dates:
            before = [d for d in dates if d <= article_date]
            if not before:
                return None
            base_idx = dates.index(before[-1])
        else:
            base_idx = dates.index(article_date)

        trend = closes[max(0, base_idx - trend_days):base_idx]

        if base_idx + 1 >= len(closes):
            next_price = None
            change_1d  = None
            bucket_1d  = None
            magnitude  = None
        else:
            base_price = closes[base_idx]
            next_price = closes[base_idx + 1]
            # avoid division by zero
            if base_price == 0:
                change_1d = None
            else:
                change_1d = round((next_price - base_price) / base_price * 100, 4)

            bucket_1d  = price_bucket(change_1d) if change_1d is not None else None
            magnitude  = magnitude_bucket(change_1d) if change_1d is not None else None

        result = {
            'ticker':     ticker,
            'trend':      trend,
            'base_price': closes[base_idx],
            'next_price': next_price,
            'change_1d':  change_1d,
            'bucket_1d':  bucket_1d,
            'magnitude':  magnitude,
        }
        _price_cache[cache_key] = result
        return result

    except Exception as e:
        print(f"  [PRICE ERROR] {ticker}: {e}")
        return None


def enrich_article(json_path):
    with open(json_path, 'r', encoding='utf-8') as fh:
        meta = json.load(fh)

    if TRUSTED_SOURCES and meta.get('site') not in TRUSTED_SOURCES:
        return None

    if not meta.get('sentiment'):
        return None

    gdelt_query = meta.get('gdelt_query', '')
    ticker      = get_ticker(gdelt_query)

    if not ticker:
        if gdelt_query not in PRIVATE_COMPANIES:
            print(f"  [NO TICKER] gdelt_query='{gdelt_query}'")
        return None

    # Skip only if primary ticker already has a complete label
    existing = meta.get('price_changes', {})
    primary  = existing.get(gdelt_query)
    if primary and primary.get('bucket_1d') is not None and primary.get('magnitude') is not None:
        return None

    date_str = meta.get('publish_date') or meta.get('scrape_date')
    if not date_str:
        return None

    price_changes = {}

    data = get_price_data(ticker, date_str)
    if data:
        price_changes[gdelt_query] = data

    benchmark      = BENCHMARKS.get(ticker, "SPY")
    benchmark_data = get_price_data(benchmark, date_str)
    if benchmark_data:
        price_changes[benchmark] = benchmark_data

    if price_changes:
        meta['price_changes'] = price_changes
        with open(json_path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        bucket = price_changes.get(gdelt_query, {}).get('bucket_1d', 'n/a')
        print(
            f"  [OK] {meta.get('site',''):<20} "
            f"ticker={ticker:<6} "
            f"benchmark={benchmark:<4} "
            f"bucket_1d={bucket}"
        )

    return meta


def collect_json_paths(data_dir):
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json'):
                paths.append(os.path.join(root, f))
    return paths


def main(data_dir):
    paths = collect_json_paths(data_dir)
    print(f"Found {len(paths)} articles\n")
    if TRUSTED_SOURCES:
        print(f"Trusted sources filter: {len(TRUSTED_SOURCES)} sites\n")
    for p in paths:
        enrich_article(p)
        time.sleep(0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw')
    args = parser.parse_args()
    main(args.data_dir)
