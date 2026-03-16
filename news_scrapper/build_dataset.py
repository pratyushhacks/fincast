r"""
Combines labeled + price-enriched articles into a JSONL fine-tuning dataset.

Each line is a training example in ChatML / instruction format:
  - input:  article text + company/industry context
  - output: sentiment score + price direction (optional)

Usage:
  python build_dataset.py --data-dir data/raw --out dataset.jsonl
"""
import os
import json
import argparse
from datetime import datetime


SYSTEM_PROMPT = (
    "You are a financial news analyst. Given a news article and a target entity "
    "(company or industry sector), rate the sentiment from 1 (very negative) to "
    "5 (very positive) and briefly explain your reasoning."
)


def format_example(meta, entity, entity_type, sentiment_score, price_change_1d):
    text_path = meta.get('text_path', '')
    if not os.path.exists(text_path):
        return None

    with open(text_path, 'r', encoding='utf-8') as fh:
        text = fh.read()[:2000]  # keep examples a consistent size

    # Build the user message
    user_content = (
        f"Article: {meta.get('title', '')}\n\n"
        f"{text}\n\n"
        f"Entity: {entity} ({entity_type})\n"
        f"Date: {(meta.get('publish_date') or meta.get('scrape_date', ''))[:10]}"
    )

    # Build the assistant response
    direction = None
    if price_change_1d is not None:
        direction = "up" if price_change_1d > 0.5 else "down" if price_change_1d < -0.5 else "flat"

    assistant_content = {
        "sentiment": sentiment_score,
        "reason": meta.get('sentiment_reason', ''),
    }
    if direction:
        assistant_content["price_direction_1d"] = direction
        assistant_content["price_change_1d_pct"] = price_change_1d

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(assistant_content)},
        ],
        # metadata (not used in training, useful for analysis)
        "_meta": {
            "url": meta.get('url'),
            "site": meta.get('site'),
            "entity": entity,
            "entity_type": entity_type,
            "scrape_date": meta.get('scrape_date'),
        }
    }


def collect_json_paths(data_dir):
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json'):
                paths.append(os.path.join(root, f))
    return paths


def main(data_dir, out_path, min_words):
    paths = collect_json_paths(data_dir)
    print(f"Scanning {len(paths)} articles...\n")

    examples = []
    skipped = 0

    for json_path in paths:
        with open(json_path, 'r', encoding='utf-8') as fh:
            meta = json.load(fh)

        sentiment = meta.get('sentiment', {})
        if not sentiment:
            skipped += 1
            continue

        if meta.get('word_count', 0) < min_words:
            skipped += 1
            continue

        price_changes = meta.get('price_changes', {})

        for entity, score in sentiment.items():
            pc = price_changes.get(entity, {})
            entity_type = (
                "company" if entity in meta.get('matched_companies', [])
                else "industry"
            )
            example = format_example(
                meta, entity, entity_type,
                sentiment_score=score,
                price_change_1d=pc.get('change_1d')
            )
            if example:
                examples.append(example)

    print(f"Built {len(examples)} training examples ({skipped} articles skipped)\n")

    with open(out_path, 'w', encoding='utf-8') as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"Written to {out_path}")

    # Quick stats
    from collections import Counter
    scores = [
        ex['messages'][2]['content']
        for ex in examples
    ]
    sentiment_counts = Counter(
        json.loads(s).get('sentiment') for s in scores
    )
    print(f"\nSentiment distribution: {dict(sorted(sentiment_counts.items()))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw')
    parser.add_argument('--out', default='dataset.jsonl')
    parser.add_argument('--min-words', type=int, default=100,
                        help='Skip articles shorter than this')
    args = parser.parse_args()
    main(args.data_dir, args.out, args.min_words)