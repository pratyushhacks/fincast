r"""
fetch_prices.py — fetches stock price trend and next-day bucket for labeled articles.

For each article:
  - trend:            7 trading day closes BEFORE article date (input context)
  - base_price:        article day close
  - next_price:         next trading day close
  - change_1d:          % change (reference only)
  - trailing_vol_pct:   stdev of daily % returns over the VOL_WINDOW trading
                         days up to and including the article/base day
                         (point-in-time — never looks past base_price, so no
                         next-day leakage). Used to normalize bucket_1d.
  - bucket_1d:          prediction label (strong_down/down/flat/up/strong_up),
                         assigned by how many standard deviations change_1d
                         is from 0, using that ticker's own trailing_vol_pct.
                         This replaces the old fixed +/-2%/+/-5% thresholds,
                         which caused low-volatility instruments (SPY, TLT,
                         UUP, XLF, XLV) to land in "flat" 90%+ of the time
                         and high-volatility names (AMD, TSLA) to spread
                         across all five buckets under the same cutoffs.
                         Vol-normalizing makes "flat" mean "flat for this
                         instrument" instead of "flat in absolute percent",
                         which balances label distribution across tickers of
                         very different volatility without needing per-
                         ticker manual tuning.
  - magnitude:          size of move (minimal/small/moderate/large), still
                         computed from absolute % change — unaffected by
                         this change, kept for backward compatibility /
                         debugging. Downstream training code (build_ticker_
                         jsonl.py) derives its own move_bin from change_1d
                         directly, so bucket_1d/magnitude here are primarily
                         for the analysis notebook and human inspection.

Usage:
  python fetch_prices.py --data-dir data/raw
  python fetch_prices.py --data-dir data/raw --recompute   # re-derive
      bucket_1d/magnitude for articles that already have price_changes,
      using the new vol-normalized logic (re-fetches price history from
      yfinance since trailing_vol needs a longer window than what's stored
      in "trend"; does NOT re-scrape articles or re-run sentiment scoring)
"""
import os
import json
import argparse
import time
import statistics
from datetime import datetime, timedelta

import yfinance as yf

from utils import get_tickers, get_benchmarks, get_trusted_sources, get_private_companies

ALL_TICKERS       = get_tickers()
BENCHMARKS        = get_benchmarks()
TRUSTED_SOURCES   = get_trusted_sources()
PRIVATE_COMPANIES = get_private_companies()

_price_cache = {}

# Trading days used to compute trailing volatility for z-score bucketing.
# 20 trading days ~= 1 calendar month, a standard short-horizon realized-vol
# window. Needs a handful more days than TREND_DAYS below.
VOL_WINDOW = 20

# Minimum trailing_vol_pct floor to avoid exploding z-scores for near-zero-
# volatility stretches (e.g. some macro ETFs in quiet periods).
MIN_VOL_FLOOR_PCT = 0.05

# Minimum number of daily returns required before we trust a vol estimate.
# Below this we fall back to the old fixed-threshold bucketing rather than
# risk a noisy/undersampled stdev driving the bucket assignment.
MIN_VOL_SAMPLES = 5

# z-score thresholds for bucket_1d. Mirrors the MOVE_BIN_THRESHOLDS scheme
# used downstream in build_ticker_jsonl.py's compute_move_bin — same
# 5-way structure, just expressed in std-devs instead of raw percent, since
# this is now instrument-relative.
Z_THRESHOLDS = {
    "strong_down": -1.5,
    "down": -0.5,
    # flat band is (-0.5, 0.5)
    "up": 0.5,
    "strong_up": 1.5,
}


def get_ticker(gdelt_query):
    return ALL_TICKERS.get(gdelt_query)


def price_bucket_fixed(pct):
    """Old fixed +/-2%/+/-5% threshold bucketing. Kept only as a fallback
    for the rare case where trailing_vol can't be computed (insufficient
    price history). Do not use this as the primary bucketing method — see
    module docstring for why it produces severe flat-label imbalance.
    """
    if pct is None: return None
    if pct <= -5:   return "strong_down"
    if pct <= -2:   return "down"
    if pct <   2:   return "flat"
    if pct <   5:   return "up"
    return                 "strong_up"


def price_bucket_zscore(change_pct, vol_pct):
    """Volatility-normalized bucket: how many std-devs is change_pct from 0,
    using this ticker's own trailing daily-return volatility as the unit.
    Falls back to price_bucket_fixed if vol_pct is unusable.
    """
    if change_pct is None:
        return None
    if vol_pct is None or vol_pct <= 0:
        return price_bucket_fixed(change_pct)

    vol_pct = max(vol_pct, MIN_VOL_FLOOR_PCT)
    z = change_pct / vol_pct

    t = Z_THRESHOLDS
    if z <= t["strong_down"]: return "strong_down"
    if z <= t["down"]:        return "down"
    if z <   t["up"]:         return "flat"
    if z <   t["strong_up"]:  return "up"
    return                          "strong_up"


def magnitude_bucket(pct):
    if pct is None: return None
    pct = abs(pct)
    if pct < 1:   return "minimal"
    if pct < 2:   return "small"
    if pct < 5:   return "moderate"
    return              "large"


def trailing_volatility(closes, base_idx, window=VOL_WINDOW):
    """Stdev of daily % returns over the `window` trading days up to and
    including base_idx. Point-in-time only — never touches closes[base_idx+1:]
    so this cannot leak next-day information into the label.
    Returns None if there isn't enough history for a stable estimate.
    """
    start = max(0, base_idx - window)
    window_closes = closes[start:base_idx + 1]
    if len(window_closes) < MIN_VOL_SAMPLES + 1:
        return None

    returns = []
    for prev, curr in zip(window_closes, window_closes[1:]):
        if prev:
            returns.append((curr - prev) / prev * 100)

    if len(returns) < MIN_VOL_SAMPLES:
        return None

    try:
        return statistics.stdev(returns)
    except statistics.StatisticsError:
        return None


def get_price_data(ticker, article_date_str, trend_days=7, vol_window=VOL_WINDOW):
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
        # Need enough calendar days back to cover vol_window trading days
        # plus the trend_days display window, with padding for weekends/
        # holidays (roughly 1.5x calendar-to-trading-day ratio).
        lookback_days = max(trend_days, vol_window) + 15
        start = article_date - timedelta(days=lookback_days)
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
        trailing_vol = trailing_volatility(closes, base_idx, window=vol_window)

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

            bucket_1d  = price_bucket_zscore(change_1d, trailing_vol) if change_1d is not None else None
            magnitude  = magnitude_bucket(change_1d) if change_1d is not None else None

        result = {
            'ticker':           ticker,
            'trend':            trend,
            'base_price':       closes[base_idx],
            'next_price':       next_price,
            'change_1d':        change_1d,
            'trailing_vol_pct': round(trailing_vol, 4) if trailing_vol is not None else None,
            'bucket_1d':        bucket_1d,
            'magnitude':        magnitude,
        }
        _price_cache[cache_key] = result
        return result

    except Exception as e:
        print(f"  [PRICE ERROR] {ticker}: {e}")
        return None


def enrich_article(json_path, force=False):
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

    # Skip only if primary ticker already has a complete label — unless
    # --recompute was passed, in which case we always re-fetch and
    # re-derive bucket_1d/magnitude under the current bucketing logic.
    existing = meta.get('price_changes', {})
    primary  = existing.get(gdelt_query)
    if not force and primary and primary.get('bucket_1d') is not None and primary.get('magnitude') is not None:
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
        vol    = price_changes.get(gdelt_query, {}).get('trailing_vol_pct', 'n/a')
        print(
            f"  [OK] {meta.get('site',''):<20} "
            f"ticker={ticker:<6} "
            f"benchmark={benchmark:<4} "
            f"vol={vol!s:<8} "
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


def main(data_dir, force=False):
    paths = collect_json_paths(data_dir)
    print(f"Found {len(paths)} articles\n")
    if TRUSTED_SOURCES:
        print(f"Trusted sources filter: {len(TRUSTED_SOURCES)} sites\n")
    if force:
        print("--recompute enabled: re-fetching price history and re-deriving "
              "bucket_1d/magnitude for ALL articles, even ones already labeled.\n")
    for p in paths:
        enrich_article(p, force=force)
        time.sleep(0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw')
    parser.add_argument('--recompute', action='store_true',
                         help='Re-fetch price history and re-derive bucket_1d/magnitude '
                              'for all articles using current bucketing logic, even ones '
                              'already labeled. Does not re-scrape or re-run sentiment.')
    args = parser.parse_args()
    main(args.data_dir, force=args.recompute)