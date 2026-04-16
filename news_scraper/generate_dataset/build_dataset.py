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
import json
import argparse
from collections import defaultdict

SYSTEM_PROMPT = (
    "You are a financial market analyst. "
    "Given today's news summaries and recent price trend for a stock or sector, "
    "predict tomorrow's price movement as one of: "
    "strong_down, down, flat, up, strong_up."
)

COMPANY_TICKERS = {
    '(NVIDIA OR NVDA)':    "NVDA",
    '(Apple OR AAPL)':     "AAPL",
    '(Microsoft OR MSFT)': "MSFT",
    '(Meta OR META)':      "META",
    '(Tesla OR TSLA)':     "TSLA",
    '(Amazon OR AMZN)':    "AMZN",
    '(Google OR GOOGL)':   "GOOGL",
    '(Intel OR INTC)':     "INTC",
    '"AMD"':               "AMD",
}

SECTOR_TICKERS = {
    "semiconductor":                                    "SOXX",
    "cloud computing":                                  "SKYY",
    "electric vehicles":                                "DRIV",
    "healthcare stocks":                                "XLV",
    "financial sector":                                 "XLF",
    "artificial intelligence market":                   "BOTZ",
    '("Federal Reserve" OR "Fed rates")':               "SPY",
    '"US inflation"':                                   "SPY",
    '"oil prices"':                                     "USO",
    '"trade war" tariffs technology':                   "XLK",
    '"US recession"':                                   "SPY",
    '"treasury yields"':                                "TLT",
    '"dollar index"':                                   "UUP",
    '"geopolitical risk"':                              "SPY",
    '"antitrust" "big tech"':                           "XLK",
    '"AI regulation"':                                  "BOTZ",
    '("China" "technology ban" OR "export controls")':  "SOXX",
    '"cybersecurity breach"':                           "CIBR",
}

# Add at top
TICKER_BENCHMARK = {
    "NVDA":  "QQQ", "AAPL":  "QQQ", "MSFT": "QQQ",
    "META":  "QQQ", "TSLA":  "QQQ", "AMZN": "QQQ",
    "GOOGL": "QQQ", "INTC":  "QQQ", "AMD":  "QQQ",
    "SOXX": "SPY", "SKYY": "SPY", "DRIV": "SPY",
    "XLV":  "SPY", "XLF":  "SPY", "BOTZ": "SPY",
    "USO":  "SPY", "TLT":  "SPY", "UUP":  "SPY",
    "XLK":  "SPY", "CIBR": "SPY",
}

PLAIN_NAME_TICKERS = {
    "NVIDIA":    "NVDA",
    "Apple":     "AAPL",
    "Microsoft": "MSFT",
    "Meta":      "META",
    "Tesla":     "TSLA",
    "Amazon":    "AMZN",
    "Google":    "GOOGL",
    "Intel":     "INTC",
    "AMD":       "AMD",
    "OpenAI":    None,
}

ALL_TICKERS = {**COMPANY_TICKERS, **SECTOR_TICKERS, **PLAIN_NAME_TICKERS}


def collect_json_paths(data_dir):
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json'):
                paths.append(os.path.join(root, f))
    return paths


def get_ticker(gdelt_query):
    return ALL_TICKERS.get(gdelt_query)


def main(data_dir, out_path, min_articles=1):
    paths = collect_json_paths(data_dir)
    print(f"Scanning {len(paths)} articles...\n")

    # Group by (date, ticker)
    # key: (date_str, ticker) -> list of article metas
    groups = defaultdict(list)

    skipped = 0
    for json_path in paths:
        print(f"Processing {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as fh:
            meta = json.load(fh)

        # Must have summary, sentiment and price_changes with bucket_1d
        if not meta.get('summary') or meta.get('sentiment') is None or not meta.get('price_changes', {}):
            skipped += 1
            print(f"  [SKIP] Missing summary/sentiment/price: {json_path}")
            continue


        price_changes = meta.get('price_changes', {})
        gdelt_query   = meta.get('gdelt_query', '')
        ticker        = get_ticker(gdelt_query)
        if not ticker:
            skipped += 1
            print(f"  [SKIP] unable to get ticker for gdelt_query: {gdelt_query} in {json_path}")
            continue

        ticker_prices = price_changes.get(gdelt_query, {})
        if not ticker_prices or ticker_prices.get('bucket_1d') is None:
            skipped += 1  # next day not yet available
            print(f"  [SKIP] No price label for ticker '{ticker}' in {json_path}")
            continue

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
    args = parser.parse_args()
    main(args.data_dir, args.out, args.min_articles)