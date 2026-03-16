r"""
fetch_prices.py — fetches stock price trend and next-day bucket for labeled articles.

For each article:
  - trend: 7 trading day closes BEFORE article date (input context)
  - base_price: article day close
  - next_price: next trading day close
  - change_1d: % change (reference only)
  - bucket_1d: prediction label (strong_down/down/flat/up/strong_up)

Usage:
  python fetch_prices.py --data-dir data/raw
"""
import os
import json
import argparse
import time
from datetime import datetime, timedelta

import yfinance as yf

import yaml

def load_trusted_sources(config_path="gdelt_config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh) or {}
        return set(cfg.get('trusted_sources', []))
    except Exception:
        return set()

TRUSTED_SOURCES = load_trusted_sources()

COMPANY_TICKERS = {
    # gdelt query format
    '"NVIDIA" OR NVDA':     "NVDA",
    '"Apple" OR AAPL':      "AAPL",
    '"Microsoft" OR MSFT':  "MSFT",
    '"Meta" OR META':       "META",
    '"Tesla" OR TSLA':      "TSLA",
    '"Amazon" OR AMZN':     "AMZN",
    '"Google" OR GOOGL':    "GOOGL",
    '"Intel" OR INTC':      "INTC",
    '"AMD"':                "AMD",
    '"OpenAI"':             None,
    # plain name fallback
    "NVIDIA":               "NVDA",
    "Apple":                "AAPL",
    "Microsoft":            "MSFT",
    "Meta":                 "META",
    "Tesla":                "TSLA",
    "Amazon":               "AMZN",
    "Google":               "GOOGL",
    "Intel":                "INTC",
    "AMD":                  "AMD",
    "OpenAI":               None,
}

SECTOR_TICKERS = {
    "semiconductor":                    "SOXX",
    "cloud computing":                  "SKYY",
    "electric vehicles":                "DRIV",
    "healthcare stocks":                "XLV",
    "financial sector":                 "XLF",
    "artificial intelligence market":   "BOTZ",
    '"Federal Reserve" OR "Fed rates"': "SPY",
    "US inflation":                     "SPY",
    "oil prices markets":               "USO",
    "trade war tariffs technology":     "XLK",
    "US recession":                     "SPY",
    "treasury yields":                  "TLT",
    "dollar index markets":             "UUP",
    "geopolitical risk markets":        "SPY",
    "antitrust big tech":               "XLK",
    "AI regulation policy":             "BOTZ",
    "China US technology ban":          "SOXX",
    "cybersecurity breach":             "CIBR",
}

_price_cache = {}


def price_bucket(pct):
    if pct is None:  return None
    if pct <= -5:    return "strong_down"
    if pct <= -2:    return "down"
    if pct <   2:    return "flat"
    if pct <   5:    return "up"
    return                  "strong_up"


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
        # Single window: trend_days before + 5 days after article date
        start = article_date - timedelta(days=trend_days + 7)  # buffer for weekends
        end   = article_date + timedelta(days=7)               # buffer for weekends

        df = yf.download(ticker, start=str(start), end=str(end),
                         progress=False, auto_adjust=True)

        if df.empty:
            return None

        closes = [round(float(x), 2) for x in df['Close'].squeeze().tolist()]

        # Find the index of article date (or nearest trading day)
        dates = [d.date() for d in df.index]
        if article_date not in dates:
            # Use last date before article date
            before = [d for d in dates if d <= article_date]
            if not before:
                return None
            base_idx = dates.index(before[-1])
        else:
            base_idx = dates.index(article_date)

        # Trend: trend_days closes before article date
        trend = closes[max(0, base_idx - trend_days):base_idx]

        # Label: next trading day after article date
        if base_idx + 1 >= len(closes):
            print(f"  [NO NEXT DAY] {ticker} | date={article_date}")
            next_price = None
            change_1d  = None
            bucket_1d  = None
        else:
            base_price = closes[base_idx]
            next_price = closes[base_idx + 1]
            change_1d  = round((next_price - base_price) / base_price * 100, 4)
            bucket_1d  = price_bucket(change_1d)

        result = {
            'ticker':     ticker,
            'trend':      trend,
            'base_price': closes[base_idx],
            'next_price': next_price,
            'change_1d':  change_1d,
            'bucket_1d':  bucket_1d,
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

    # Skip if already complete (all non-SPY entries have bucket_1d)
    price_changes = meta.get('price_changes', {})
    if price_changes:
        all_labeled = all(
            v.get('bucket_1d') is not None
            for k, v in price_changes.items()
            if k != 'SPY'
        )
        if all_labeled:
            return None  # already complete

    date_str = meta.get('publish_date') or meta.get('scrape_date')
    if not date_str:
        return None

    gdelt_query = meta.get('gdelt_query', '')
    price_changes = {}

    ticker = COMPANY_TICKERS.get(gdelt_query) or SECTOR_TICKERS.get(gdelt_query)
    if ticker:
        data = get_price_data(ticker, date_str)
        if data:
            price_changes[gdelt_query] = data

    spy = get_price_data("SPY", date_str)
    if spy:
        price_changes["SPY"] = spy

    if price_changes:
        meta['price_changes'] = price_changes
        with open(json_path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        bucket = price_changes.get(gdelt_query, {}).get('bucket_1d', 'n/a')
        print(
            f"  [OK] {meta.get('site',''):<20} "
            f"query={gdelt_query:<30} "
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
    for p in paths:
        enrich_article(p)
        time.sleep(0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw')
    args = parser.parse_args()
    main(args.data_dir)