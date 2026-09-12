r"""
build_dataset.py — builds daily per-ticker JSONL for fine-tuning.

For each (date, ticker) pair:
  - Collects all article summaries + sentiments for that day
  - Includes last 7 days price trend
  - Label: bucket_1d (strong_down/down/flat/up/strong_up)

Usage:
  python build_dataset.py --data-dir data/raw --out dataset.jsonl
"""
import os
import sys
import json
import shutil
import argparse
from collections import defaultdict

# Ensure the sibling create_data package can be imported when running from generate_dataset/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from create_data.utils import get_tickers, get_benchmarks

SYSTEM_PROMPT = (
    "You are a financial market analyst. "
    "Given today's news summaries and recent price trend for a stock or sector, "
    "predict tomorrow's price movement as one of: "
    "strong_down, down, flat, up, strong_up."
)

ALL_TICKERS = get_tickers()
TICKER_BENCHMARK = get_benchmarks()


def collect_json_paths(data_dir):
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json'):
                paths.append(os.path.join(root, f))
    return paths


def get_ticker(gdelt_query):
    return ALL_TICKERS.get(gdelt_query)


def is_same_file(src, dst):
    if not os.path.exists(dst):
        return False
    try:
        return (os.path.getsize(src) == os.path.getsize(dst) and
                int(os.path.getmtime(src)) == int(os.path.getmtime(dst)))
    except OSError:
        return False


def get_article_quality_reason(meta, min_relevance=2, min_summary_words=10):
    if not meta.get('summary'):
        return False, 'missing_summary'
    if meta.get('sentiment') is None:
        return False, 'missing_sentiment'
    if not meta.get('price_changes', {}):
        return False, 'missing_price_changes'

    summary = meta.get('summary', '')
    if len(summary.split()) < min_summary_words:
        return False, 'too_short_summary'

    if meta.get('relevance', 1) < min_relevance:
        return False, 'low_relevance'

    gdelt_query = meta.get('gdelt_query', '')
    ticker = get_ticker(gdelt_query)
    if not ticker:
        return False, 'missing_ticker'

    ticker_prices = meta['price_changes'].get(gdelt_query, {})
    if not ticker_prices or ticker_prices.get('bucket_1d') is None:
        return False, 'missing_bucket'

    return True, 'ok'


def is_good_article(meta, min_relevance=2, min_summary_words=10):
    good, _ = get_article_quality_reason(meta, min_relevance, min_summary_words)
    return good


def collect_by_ticker(data_dir, out_root, min_relevance=2, min_summary_words=10):
    paths = collect_json_paths(data_dir)
    print(f"Scanning {len(paths)} articles...\n")
    os.makedirs(out_root, exist_ok=True)
    counts = defaultdict(int)
    skipped = 0
    skip_reasons = defaultdict(int)
    expected_paths = set()

    for json_path in paths:
        with open(json_path, 'r', encoding='utf-8') as fh:
            try:
                meta = json.load(fh)
            except json.JSONDecodeError:
                skipped += 1
                skip_reasons['invalid_json'] += 1
                print(f"  [SKIP] invalid JSON: {json_path}")
                continue

        good, reason = get_article_quality_reason(meta, min_relevance, min_summary_words)
        if not good:
            skipped += 1
            skip_reasons[reason] += 1
            continue

        ticker = get_ticker(meta.get('gdelt_query', ''))
        rel_path = os.path.relpath(json_path, data_dir)
        out_path = os.path.join(out_root, ticker, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not is_same_file(json_path, out_path):
            shutil.copy2(json_path, out_path)
        expected_paths.add(os.path.normpath(out_path))
        counts[ticker] += 1

    removed = 0
    for root, _, files in os.walk(out_root, topdown=False):
        for f in files:
            if not f.endswith('.json'):
                continue
            out_path = os.path.normpath(os.path.join(root, f))
            if out_path not in expected_paths:
                os.remove(out_path)
                removed += 1
        if not os.listdir(root):
            os.rmdir(root)

    print(f"Copied/updated {sum(counts.values())} articles into {out_root}")
    print(f"Skipped {skipped} articles due to quality/filtering or invalid JSON")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}")
    if removed:
        print(f"Removed {removed} stale article files from {out_root}")
    for ticker, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ticker}: {count}")


def main(data_dir, out_path, min_articles=1, min_relevance=2, min_summary_words=10):
    paths = collect_json_paths(data_dir)
    print(f"Scanning {len(paths)} articles...\n")

    # Group by (date, ticker, gdelt_query) to avoid mixing different watchlist forms.
    groups = defaultdict(list)

    skipped = 0
    for json_path in paths:
        print(f"Processing {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as fh:
            meta = json.load(fh)

        good, reason = get_article_quality_reason(meta, min_relevance, min_summary_words)
        if not good:
            skipped += 1
            print(f"  [SKIP] {reason}: {json_path}")
            continue

        price_changes = meta.get('price_changes', {})
        gdelt_query = meta.get('gdelt_query', '')
        ticker = get_ticker(gdelt_query)
        ticker_prices = price_changes.get(gdelt_query, {})

        # Use scrape_date as the article date
        date_str = (meta.get('publish_date') or meta.get('scrape_date', ''))[:10]
        groups[(date_str, ticker, gdelt_query)].append(meta)

    print(f"Skipped: {skipped}")
    print(f"Groups (date x ticker): {len(groups)}\n")

    examples = []
    for (date_str, ticker, gdelt_query), articles in sorted(groups.items()):
        if len(articles) < min_articles:
            continue

        # Build news digest for this ticker on this date
        news_items = []
        for a in articles:
            news_items.append({
                # "title":     a.get('title', ''),
                "summary":   a.get('summary', ''),
                "sentiment": a.get('sentiment'),
                # "reason":    a.get('sentiment_reason', ''),
            })

        # Get price data from first article that has it
        ticker     = get_ticker(gdelt_query)
        benchmark  = TICKER_BENCHMARK.get(ticker, "SPY")
        price_data     = articles[0]['price_changes'].get(gdelt_query, {})
        benchmark_data = articles[0]['price_changes'].get(benchmark, {})

        user_content = (
            f"Date: {date_str}\n"
            f"Ticker: {ticker}\n\n"
            f"Last 7 days closing prices ({ticker}):\n"
            f"{price_data.get('trend', [])}\n\n"
            f"Benchmark ({benchmark}) last 7 days:\n"
            f"{benchmark_data.get('trend', [])}\n\n"
            f"Today's news ({len(news_items)} articles):\n"
            f"{json.dumps(news_items, ensure_ascii=False, indent=2)}\n\n"
            f"Predict tomorrow's price movement for {ticker}."
        )

        assistant_content = {
            "prediction":  price_data.get('bucket_1d'),
            "change_1d":   price_data.get('change_1d'),
            "base_price":  price_data.get('base_price'),
            "next_price":  price_data.get('next_price'),
        }

        example = {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": json.dumps(assistant_content)},
            ],
            "_meta": {
                "date":         date_str,
                "ticker":       ticker,
                "gdelt_query":  gdelt_query,
                "num_articles": len(articles),
            }
        }
        examples.append(example)

    print(f"Built {len(examples)} training examples\n")

    with open(out_path, 'w', encoding='utf-8') as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"Written to {out_path}")

    # Stats
    from collections import Counter
    predictions = Counter(
        json.loads(ex['messages'][2]['content']).get('prediction')
        for ex in examples
    )
    print(f"\nPrediction distribution: {dict(sorted(predictions.items()))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',     default='data/raw')
    parser.add_argument('--out',          default='dataset.jsonl')
    parser.add_argument('--min-articles', type=int, default=1,
                        help='Min articles per ticker per day to include')
    parser.add_argument('--collect-by-ticker', default=None,
                        help='If set, copy raw JSON files into ticker folders under this directory')
    parser.add_argument('--min-relevance', type=int, default=2,
                        help='Minimum relevance score to include an article')
    parser.add_argument('--min-summary-words', type=int, default=10,
                        help='Minimum summary length in words to include an article')
    args = parser.parse_args()
    if args.collect_by_ticker:
        collect_by_ticker(args.data_dir, args.collect_by_ticker,
                          min_relevance=args.min_relevance,
                          min_summary_words=args.min_summary_words)
    else:
        main(args.data_dir, args.out, args.min_articles,
             min_relevance=args.min_relevance,
             min_summary_words=args.min_summary_words)