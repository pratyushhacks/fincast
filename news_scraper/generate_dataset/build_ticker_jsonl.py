#!/usr/bin/env python3
"""
build_ticker_jsonl.py — build per-ticker JSONL files directly from the raw
nightly pipeline output, formatted as chat-style SFT examples for QLoRA
fine-tuning of Phi-3-mini.

Raw articles are stored by scraper.py as
  {data_dir}/{domain}/{date}/{article_id}.json
with no per-ticker folder — the ticker is only discoverable by resolving
each article's gdelt_query against the watchlist config. This script does
that resolution + quality filtering (formerly a separate
`build_dataset.py --collect-by-ticker` copy step) in one pass, so there's
no intermediate ticker_data/ folder to keep in sync with the source.

Usage:
  python build_ticker_jsonl.py --ticker MSFT --data-dir data/raw --out data/train/MSFT.jsonl
  python build_ticker_jsonl.py --all-tickers --data-dir data/raw --out-dir data/train

Output format (one line per example):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Ticker: MSFT\\nDate: ...\\nPast returns (%): [...]\\n\\nNews:\\n1. ..."},
    {"role": "assistant", "content": "{\"move_bin\": \"flat\"}"}
  ]
}

Pass --emit-format raw if you want the old intermediate (non-chat) record
shape for debugging/inspection instead of training-ready output.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import requests
from dotenv import load_dotenv

# utils.py lives in create_data/, a sibling package to this file's folder
# (generate_dataset/) — not in the same directory. Same sys.path convention
# build_dataset.py already used for `from create_data.utils import ...`.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from create_data.utils import get_tickers

SCRIPT_DIR = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(SCRIPT_DIR, "..", ".env"))

DEFAULT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "Phi-3-mini-128k-instruct-cuda-gpu:2")
DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))
LLM_BASE_URL = f"http://{os.getenv('HOST', '127.0.0.1')}:{os.getenv('PORT', '8091')}/v1"
PRINT_PROMPT = False

ALL_TICKERS = get_tickers()

# Quality gate applied per-article before it's even considered for a day's
# news digest — mirrors what build_dataset.py's --collect-by-ticker used to
# do as a separate copy step, now folded in directly.
DEFAULT_MIN_RELEVANCE = 2
DEFAULT_MIN_SUMMARY_WORDS = 10

# data/ lives at the project root (news_scraper/data/), not inside
# generate_dataset/ — so defaults are anchored to ROOT_DIR rather than a
# bare relative path, which would only work if you happened to run this
# script from news_scraper/ itself rather than from generate_dataset/.
DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, 'data', 'train')

DEDUPE_PROMPT = """You are a news curator for Microsoft (ticker MSFT).

Group these candidate articles into unique news stories using only title and summary.
Return exactly one valid JSON array of strings and nothing else.
Each string must be the articleID of the first article to keep for a unique story.
If multiple candidates describe the same story, return only one articleID.
Do not include markdown fences or any extra text.

Candidates:
{candidates}
"""

# ---------------------------------------------------------------------------
# Training-time system prompt. Keep this byte-for-byte identical across every
# training example AND at inference time — small models are sensitive to
# system prompt drift, and mismatches here are a classic cause of malformed
# JSON output at serve time.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a financial analyst. Given a stock's ticker, recent price "
    "returns, and recent news with sentiment scores, predict tomorrow's "
    "price movement. Respond with JSON only: "
    '{"move_bin": <one of strong_down|down|flat|up|strong_up>}.'
)

# ---------------------------------------------------------------------------
# move_bin thresholds, expressed as 1-day percent return.
# These are placeholders — tune against your actual return distribution.
# Given the flat-label dominance issue you flagged (~74-78%), you likely
# want the "flat" band narrower than this default, not wider, once you have
# a full 2-month dataset to inspect.
# ---------------------------------------------------------------------------
MOVE_BIN_THRESHOLDS = {
    "strong_down": -2.0,   # pct_change <= -2.0
    "down": -0.5,          # -2.0 < pct_change <= -0.5
    # flat band is implicitly (-0.5, 0.5)
    "up": 0.5,             # 0.5 <= pct_change < 2.0
    "strong_up": 2.0,      # pct_change >= 2.0
}

# Fallback mapping used ONLY when no raw change_1d/next_price/base_price is
# available in price_changes and we have to derive move_bin from the older
# bucket_1d + magnitude fields instead. This collapses information (it can't
# recover a true 5-way split from a 3-way bucket + magnitude), so treat any
# records that hit this path as lower-quality and consider excluding them
# once you have real change_1d in the source data.
BUCKET_MAGNITUDE_FALLBACK = {
    ("down", "large"): "strong_down",
    ("down", "moderate"): "down",
    ("down", "medium"): "down",
    ("down", "small"): "down",
    ("flat", "large"): "flat",
    ("flat", "moderate"): "flat",
    ("flat", "medium"): "flat",
    ("flat", "small"): "flat",
    ("up", "small"): "up",
    ("up", "moderate"): "up",
    ("up", "medium"): "up",
    ("up", "large"): "strong_up",
}


def get_article_quality_reason(meta: dict, min_relevance: int, min_summary_words: int) -> tuple[bool, str]:
    """Per-article quality gate, folded in from build_dataset.py's
    --collect-by-ticker step. Applied once here rather than as a separate
    copy pass, so there's no ticker_data/ folder that can drift out of sync
    with data/raw.
    """
    if not meta.get('summary'):
        return False, 'missing_summary'
    if meta.get('sentiment') is None:
        return False, 'missing_sentiment'
    if not meta.get('price_changes', {}):
        return False, 'missing_price_changes'
    if len(meta.get('summary', '').split()) < min_summary_words:
        return False, 'too_short_summary'
    if meta.get('relevance', 1) < min_relevance:
        return False, 'low_relevance'
    return True, 'ok'


# Cache of {data_dir: {TICKER: [(path, meta), ...]}} so that building every
# ticker in one run (--all-tickers) walks the raw data tree exactly once,
# instead of once per ticker.
_ticker_index_cache: dict[str, dict[str, list[tuple[str, dict]]]] = {}


def build_ticker_index(
    data_dir: str,
    min_relevance: int = DEFAULT_MIN_RELEVANCE,
    min_summary_words: int = DEFAULT_MIN_SUMMARY_WORDS,
) -> dict[str, list[tuple[str, dict]]]:
    """Walk the raw data_dir ONCE (structure: {domain}/{date}/{id}.json —
    there is no per-ticker folder) and group already-parsed, quality-passing
    article metadata by resolved ticker (via gdelt_query -> watchlist
    mapping, the same lookup fetch_prices.py uses — NOT the 'ticker' field
    inside each article, which is just an LLM-echoed value and not
    authoritative).
    """
    cache_key = (data_dir, min_relevance, min_summary_words)
    if cache_key in _ticker_index_cache:
        return _ticker_index_cache[cache_key]

    index: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    skip_reasons: dict[str, int] = defaultdict(int)
    scanned = 0

    for root, _, files in os.walk(data_dir):
        for name in files:
            if not name.endswith('.json'):
                continue
            full_path = os.path.join(root, name)
            scanned += 1
            try:
                with open(full_path, 'r', encoding='utf-8') as fh:
                    meta = json.load(fh)
            except (json.JSONDecodeError, OSError):
                skip_reasons['invalid_json'] += 1
                continue

            gdelt_query = meta.get('gdelt_query', '')
            ticker = ALL_TICKERS.get(gdelt_query)
            if not ticker:
                skip_reasons['unresolved_ticker'] += 1
                continue

            good, reason = get_article_quality_reason(meta, min_relevance, min_summary_words)
            if not good:
                skip_reasons[reason] += 1
                continue

            index[ticker.upper()].append((full_path, meta))

    print(f"[INDEX] scanned {scanned} files under {data_dir}, "
          f"resolved into {len(index)} tickers, "
          f"kept {sum(len(v) for v in index.values())} articles")
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  skipped ({reason}): {count}")

    result = dict(index)
    _ticker_index_cache[cache_key] = result
    return result


def parse_date(meta: dict) -> str:
    date = meta.get('publish_date') or meta.get('scrape_date') or ''
    return date[:10]


def load_meta(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _extract_json_array(raw: str) -> list:
    if '```' in raw:
        parts = raw.split('```')
        raw = parts[1].lstrip('json').strip() if len(parts) > 1 else raw
    start = raw.find('[')
    end = raw.rfind(']')
    if start == -1 or end == -1:
        raise ValueError(f'Unable to extract JSON array from model output: {raw!r}')
    return json.loads(raw[start:end + 1])


def call_dedupe_llm(date: str, candidates: list[dict], retries: int = 2, backoff_seconds: float = 4.0) -> list[dict]:
    prompt = DEDUPE_PROMPT.format(
        date=date,
        candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
    )
    if PRINT_PROMPT:
        print("\n=== DEDUPE PROMPT ===")
        print(prompt)
        print("=== END PROMPT ===\n")
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 800,
    }

    last_exc = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                f"{LLM_BASE_URL}/chat/completions",
                json=payload,
                timeout=DEFAULT_TIMEOUT,
            )
            if response.status_code >= 500:
                # 5xx from the wrapper (e.g. "Error contacting foundry") is
                # usually the local Foundry server transiently crashing or
                # reloading under GPU/VRAM pressure — worth a short retry
                # before falling back to no-dedup, rather than giving up on
                # the first hiccup.
                print(f"[ERROR] wrapper HTTP {response.status_code} (attempt {attempt+1}/{retries+1})")
                print(response.text[:500])
                if attempt < retries:
                    time.sleep(backoff_seconds * (attempt + 1))
                    continue
                response.raise_for_status()
            elif response.status_code >= 400:
                # 4xx is a real request problem, not transient — no point retrying
                print(f"[ERROR] wrapper HTTP {response.status_code}")
                print(response.text[:2000])
                response.raise_for_status()

            try:
                data = response.json()
            except json.JSONDecodeError:
                print("[ERROR] wrapper returned invalid JSON:")
                print(response.text[:2000])
                raise

            try:
                raw = data["choices"][0]["message"]["content"].strip()
            except Exception:
                print("[ERROR] unexpected response shape from wrapper:")
                print(json.dumps(data, ensure_ascii=False)[:2000])
                raise

            try:
                return _extract_json_array(raw)
            except json.JSONDecodeError:
                print("[ERROR] model returned invalid JSON payload:")
                print(raw)
                raise

        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                print(f"[WARN] dedupe request failed (attempt {attempt+1}/{retries+1}): {exc}")
                time.sleep(backoff_seconds * (attempt + 1))
                continue
            raise

    raise last_exc


def build_news_item(meta: dict) -> dict:
    return {
        'articleID': meta.get('id') or meta.get('guid') or meta.get('url'),
        'title': meta.get('title', ''),
        'summary': meta.get('summary', ''),
    }


def _finalize_news_items(articles: list[dict], deduped_ids: list[str]) -> list[dict]:
    meta_map = {str(a.get('id')): a for a in articles}
    final = []
    seen = set()
    for article_id in deduped_ids:
        article_id = str(article_id)
        if article_id in seen:
            continue
        seen.add(article_id)
        rep_meta = meta_map.get(article_id)
        if not rep_meta:
            continue

        final.append({
            'title': rep_meta.get('title', ''),
            'summary': rep_meta.get('summary', ''),
            'sentiment': rep_meta.get('sentiment'),
            'direction': rep_meta.get('direction'),
        })
    return final


def dedupe_articles(date: str, articles: list[dict], max_candidates: int = 0, fail_on_error: bool = False) -> list[dict]:
    candidates = [build_news_item(a) for a in articles]
    if len(candidates) <= 1:
        return _finalize_news_items(articles, [candidates[0]['articleID']])

    if max_candidates > 0 and len(candidates) > max_candidates:
        print(f"[DEBUG] limiting {date} to {max_candidates} candidates from {len(candidates)}")
        candidates = candidates[:max_candidates]

    deduped_ids = None
    try:
        deduped = call_dedupe_llm(date, candidates)
        if isinstance(deduped, list) and deduped:
            deduped_ids = []
            for item in deduped:
                if isinstance(item, dict):
                    article_id = item.get('articleID') or item.get('articleId') or item.get('id')
                else:
                    article_id = item
                if article_id:
                    deduped_ids.append(str(article_id))
    except Exception as exc:
        if fail_on_error:
            raise
        print(f"[WARN] LLM dedupe failed for {date}: {exc}")

    if not deduped_ids:
        deduped_ids = [str(c['articleID']) for c in candidates]

    return _finalize_news_items(articles, deduped_ids)


def news_signal_strength(news_item: dict) -> float:
    """Rank key for picking the most informative news items when a day has
    more distinct deduped stories than we want in one training example.
    Uses distance from neutral sentiment (3) as a proxy for how much signal
    an article actually carries — a 5 (strongly positive) or 1 (strongly
    negative) article is more informative to the model than a 3 (neutral).
    """
    sentiment = news_item.get('sentiment')
    try:
        return abs(int(sentiment) - 3)
    except (TypeError, ValueError):
        return 0


def dedupe_all_articles_for_date(date: str, articles: list[dict], batch_size: int = 5) -> list[dict]:
    """Dedupe a full day's articles, respecting the GPU-imposed batch_size
    limit on each LLM call, but merging results across batches BEFORE
    emitting any training row — so a day with 12 raw articles produces one
    final deduped news list, not 2-3 separate rows that split the same day's
    coverage arbitrarily by API batch boundary.

    If merging leaves more than `batch_size` surviving stories (common when
    a day has many raw articles across several batches, some of which are
    themselves duplicates of each other across batches), we run one final
    lightweight consolidation pass — reusing the same dedupe_batch machinery
    on the already-shrunk survivor list — since that list is small enough by
    then to fit in a single GPU-safe LLM call.
    """
    survivors = []
    for chunk in chunked(articles, batch_size):
        batch_rows = dedupe_batch(date, chunk, fallback_size=batch_size)
        for news in batch_rows:
            survivors.extend(news)

    if len(survivors) <= batch_size:
        return survivors

    # Cross-batch survivors can still contain duplicates (e.g. the same
    # story picked up by two different sources, each landing in a different
    # raw batch). One more consolidation pass, now cheap since len(survivors)
    # is small. Reuse dedupe_articles' underlying candidate/LLM-dedupe path
    # by wrapping survivors back into the shape dedupe_articles expects.
    pseudo_articles = [
        {'id': str(i), 'title': n.get('title', ''), 'summary': n.get('summary', ''),
         'sentiment': n.get('sentiment'), 'direction': n.get('direction')}
        for i, n in enumerate(survivors)
    ]
    try:
        deduped_ids = call_dedupe_llm(date, [build_news_item(a) for a in pseudo_articles])
        keep_ids = set()
        for item in deduped_ids:
            aid = item.get('articleID') if isinstance(item, dict) else item
            if aid is not None:
                keep_ids.add(str(aid))
        consolidated = [survivors[int(i)] for i in keep_ids if i.isdigit() and int(i) < len(survivors)]
        if consolidated:
            survivors = consolidated
    except Exception as exc:
        print(f"[WARN] cross-batch consolidation failed for {date}, keeping all {len(survivors)} survivors: {exc}")

    return survivors


def chunked(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def dedupe_batch(date: str, articles: list[dict], fallback_size: int = 3) -> list[list[dict]]:
    try:
        news = dedupe_articles(date, articles, fail_on_error=True)
        return [news]
    except Exception as exc:
        if len(articles) <= fallback_size:
            print(f"[WARN] LLM dedupe failed for batch size {len(articles)} on {date}; falling back to heuristic: {exc}")
            return [dedupe_articles(date, articles, fail_on_error=False)]

        print(f"[WARN] LLM dedupe failed for batch size {len(articles)} on {date}; splitting into {fallback_size}-article sub-batches: {exc}")
        rows = []
        for sub in chunked(articles, fallback_size):
            rows.extend(dedupe_batch(date, sub, fallback_size=fallback_size))
        return rows


# ---------------------------------------------------------------------------
# Price / label handling
# ---------------------------------------------------------------------------

def get_price_record(article: dict) -> dict:
    """Pull everything we might need from price_changes for a given article.

    Returns raw price_trend (list of price levels), base_price/next_price/
    change_1d if present in the source data, plus the legacy bucket_1d/
    magnitude fields as a fallback for move_bin derivation.
    """
    gdelt_query = article.get('gdelt_query', '')
    price_changes = article.get('price_changes', {})
    price_data = price_changes.get(gdelt_query, {})

    def first_present(*keys):
        for k in keys:
            if k in price_data and price_data[k] is not None:
                return price_data[k]
        return None

    return {
        'price_trend': price_data.get('trend', []),
        'base_price': first_present('base_price', 'current_price', 'close'),
        'next_price': first_present('next_price', 'next_close'),
        'change_1d': first_present('change_1d', 'pct_change_1d', 'pct_change'),
        'bucket_1d': price_data.get('bucket_1d'),
        'magnitude': price_data.get('magnitude'),
    }


def pct_returns(prices: list[float]) -> list[float]:
    """Day-over-day percent returns from a list of raw price levels."""
    out = []
    for prev, curr in zip(prices, prices[1:]):
        if prev in (0, None) or curr is None:
            continue
        out.append(round((curr - prev) / prev * 100, 3))
    return out


def compute_move_bin(price_record: dict) -> str | None:
    """Derive the single move_bin label.

    Preferred path: use an explicit 1-day pct change if the source data has
    one (change_1d, or base_price/next_price to derive it).
    Fallback path: collapse legacy bucket_1d + magnitude into a 5-way label.
    Returns None if neither is available (caller should skip the record —
    do not silently default to "flat", that's exactly how you get label
    collapse).
    """
    change = price_record.get('change_1d')
    if change is None and price_record.get('base_price') and price_record.get('next_price'):
        base = price_record['base_price']
        nxt = price_record['next_price']
        if base:
            change = (nxt - base) / base * 100

    if change is not None:
        t = MOVE_BIN_THRESHOLDS
        if change <= t["strong_down"]:
            return "strong_down"
        if change <= t["down"]:
            return "down"
        if change < t["up"]:
            return "flat"
        if change < t["strong_up"]:
            return "up"
        return "strong_up"

    # Fallback: no raw change available, use bucket_1d + magnitude
    bucket = price_record.get('bucket_1d')
    magnitude = price_record.get('magnitude')
    if bucket and magnitude:
        return BUCKET_MAGNITUDE_FALLBACK.get((bucket, magnitude), bucket)
    if bucket:
        return bucket

    return None


def day_of_week(date_str: str) -> str:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")
    except ValueError:
        return ""


def build_user_content(record: dict) -> str | None:
    """Build just the user-turn text (ticker/date/returns/day-of-week/news) —
    no label attached. This is the piece that MUST be identical whether
    you're building a training example or a live prediction payload, since
    any drift between the two is train/serve skew that quietly degrades a
    model that actually learned something useful. Both build_chat_example()
    (training) and predict_ticker.py (live inference) call this same
    function rather than each having their own copy of this formatting.

    Returns None if there's no usable price trend to build returns from —
    same "skip rather than guess" policy as the rest of this pipeline.
    """
    price_record = record.get('_price_record', {})

    returns = pct_returns(price_record.get('price_trend', []))
    if price_record.get('base_price') and price_record.get('price_trend'):
        last = price_record['price_trend'][-1]
        base = price_record['base_price']
        if last:
            returns.append(round((base - last) / last * 100, 3))

    if not returns:
        return None

    news_lines = []
    for i, n in enumerate(record.get('news', []), 1):
        news_lines.append(
            f'{i}. "{n["title"]}" — sentiment: {n.get("sentiment")}, '
            f'direction: {n.get("direction")}. {n.get("summary", "")}'
        )

    return (
        f"Ticker: {record['ticker']}\n"
        f"Date: {record['date']}\n"
        f"Past returns (%): {returns}\n"
        f"Day of week: {day_of_week(record['date'])}\n\n"
        f"News:\n" + "\n".join(news_lines)
    )


def build_chat_example(record: dict) -> dict | None:
    """Convert an intermediate raw record into a messages-format SFT example.
    Requires a resolvable move_bin (ground truth) — that's only available
    for historical dates where the next day's price is already known, which
    is exactly why this is training-only and not reused for live inference.
    """
    user_content = build_user_content(record)
    move_bin = compute_move_bin(record.get('_price_record', {}))

    if move_bin is None or user_content is None:
        # Not enough signal to build a valid training example — skip rather
        # than guess. Silent defaults here are how the flat-collapse problem
        # happens in the first place.
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps({"move_bin": move_bin})},
        ]
    }


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + '\n')


def build_and_write_records(
    ticker: str,
    data_dir: str,
    out_path: str,
    max_candidates: int = 0,
    fail_on_error: bool = False,
    max_days: int = 0,
    emit_format: str = "chat",
    min_relevance: int = DEFAULT_MIN_RELEVANCE,
    min_summary_words: int = DEFAULT_MIN_SUMMARY_WORDS,
) -> int:
    index = build_ticker_index(data_dir, min_relevance, min_summary_words)
    articles_with_paths = index.get(ticker.upper(), [])
    if not articles_with_paths:
        raise FileNotFoundError(
            f'No qualifying articles found for ticker {ticker} in {data_dir} '
            f'(min_relevance={min_relevance}, min_summary_words={min_summary_words})'
        )

    records_by_date = defaultdict(list)
    for _path, meta in articles_with_paths:
        date = parse_date(meta)
        if not date:
            continue
        records_by_date[date].append(meta)

    if os.path.exists(out_path):
        open(out_path, 'w', encoding='utf-8').close()

    total_records = 0
    skipped_records = 0
    processed_days = 0
    for date, articles in sorted(records_by_date.items()):
        if max_days and processed_days >= max_days:
            break

        print(f"[INFO] processing date {date} with {len(articles)} articles")
        processed_days += 1

        price_record = get_price_record(articles[0])
        if not price_record['price_trend']:
            for art in articles:
                pr = get_price_record(art)
                if pr['price_trend']:
                    price_record = pr
                    break

        # One row per ticker-date: dedupe across ALL of the day's raw
        # articles (batched internally for GPU safety), then keep only the
        # top-5 strongest-signal stories for the final training example.
        # This replaces the old per-batch-chunk loop, which emitted a
        # separate near-duplicate row for every group of <=5 raw articles —
        # splitting one trading day's coverage arbitrarily across rows by
        # whatever order articles happened to be batched in.
        news = dedupe_all_articles_for_date(date, articles, batch_size=max_candidates or 5)
        if not news:
            continue

        news = sorted(news, key=news_signal_strength, reverse=True)[:5]

        raw_record = {
            'date': date,
            'ticker': ticker.upper(),
            'news': news,
            '_price_record': price_record,
        }

        if emit_format == "raw":
            out_record = {
                'date': raw_record['date'],
                'ticker': raw_record['ticker'],
                'news': raw_record['news'],
                'price_trend': price_record.get('price_trend', []),
                'move_bin': compute_move_bin(price_record),
            }
            append_jsonl(out_path, out_record)
            total_records += 1
        else:
            chat_record = build_chat_example(raw_record)
            if chat_record is None:
                skipped_records += 1
                continue
            append_jsonl(out_path, chat_record)
            total_records += 1

    if skipped_records:
        print(f"[WARN] skipped {skipped_records} records with no usable move_bin/returns")

    return total_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build per-ticker JSONL file(s) directly from raw nightly pipeline output, '
                    'with LLM deduplication, formatted for chat-style SFT fine-tuning.'
    )
    parser.add_argument('--ticker', default=None, help='Ticker symbol, e.g. MSFT (required unless --all-tickers)')
    parser.add_argument('--all-tickers', action='store_true',
                         help='Build a JSONL for every ticker found in the watchlist in one pass '
                              '(single scan of --data-dir instead of one scan per ticker)')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help='Path to raw nightly pipeline output')
    parser.add_argument('--out', default=None, help='Output JSONL path (single-ticker mode only)')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR,
                         help='Output directory for --all-tickers mode; writes {out-dir}/{TICKER}.jsonl')
    parser.add_argument('--llm-timeout', type=int, default=None, help='Timeout in seconds for the local LLM wrapper call')
    parser.add_argument('--print-prompt', action='store_true', help='Print the exact prompt sent to the LLM')
    parser.add_argument('--max-candidates', type=int, default=0, help='Limit candidate articles sent to the LLM for deduping')
    parser.add_argument('--fail-on-llm-error', action='store_true', help='Abort immediately if LLM dedupe fails')
    parser.add_argument('--max-days', type=int, default=0, help='Process only the first N publish dates')
    parser.add_argument('--emit-format', choices=['chat', 'raw'], default='chat',
                         help='chat = training-ready messages JSONL (default). raw = intermediate debug format.')
    parser.add_argument('--min-relevance', type=int, default=DEFAULT_MIN_RELEVANCE,
                         help='Minimum relevance score (1-3) to include an article')
    parser.add_argument('--min-summary-words', type=int, default=DEFAULT_MIN_SUMMARY_WORDS,
                         help='Minimum summary length in words to include an article')
    args = parser.parse_args()
    if not args.all_tickers and not args.ticker:
        parser.error('Either --ticker or --all-tickers is required')
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.llm_timeout is not None:
        DEFAULT_TIMEOUT = args.llm_timeout
    if args.print_prompt:
        PRINT_PROMPT = True

    if args.all_tickers:
        # Single scan of data_dir (via build_ticker_index's cache), then one
        # output file per ticker discovered in the watchlist.
        index = build_ticker_index(args.data_dir, args.min_relevance, args.min_summary_words)
        os.makedirs(args.out_dir, exist_ok=True)
        grand_total = 0
        for ticker in sorted(index.keys()):
            out_path = os.path.join(args.out_dir, f'{ticker}.jsonl')
            try:
                total_records = build_and_write_records(
                    ticker,
                    args.data_dir,
                    out_path,
                    max_candidates=args.max_candidates,
                    fail_on_error=args.fail_on_llm_error,
                    max_days=args.max_days,
                    emit_format=args.emit_format,
                    min_relevance=args.min_relevance,
                    min_summary_words=args.min_summary_words,
                )
            except FileNotFoundError as exc:
                print(f'[SKIP] {ticker}: {exc}')
                continue
            print(f'Wrote {total_records} records to {out_path} (format={args.emit_format})')
            grand_total += total_records
        print(f'\nDone: {grand_total} total records across {len(index)} tickers in {args.out_dir}')
    else:
        out_path = args.out or os.path.join(DEFAULT_OUT_DIR, f'{args.ticker.upper()}.jsonl')
        total_records = build_and_write_records(
            args.ticker,
            args.data_dir,
            out_path,
            max_candidates=args.max_candidates,
            fail_on_error=args.fail_on_llm_error,
            max_days=args.max_days,
            emit_format=args.emit_format,
            min_relevance=args.min_relevance,
            min_summary_words=args.min_summary_words,
        )
        print(f'Wrote {total_records} records to {out_path} (format={args.emit_format})')