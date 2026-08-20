#!/usr/bin/env python3
"""
combine_datasets.py — merge per-ticker training JSONL files (built by
build_ticker_jsonl.py) into a single train/val split for fine-tuning.

Per-ticker files stay as the build-time artifact (easy to inspect, re-fetch,
or exclude one ticker at a time). This script is the assembly step that
produces what actually gets fed to SFTTrainer.

Usage:
  python combine_datasets.py --data-dir data/train_data --out-dir data/combined
"""
import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict


def get_date(record: dict) -> str:
    # Date is embedded in the user message text, e.g. "Date: 2026-03-17"
    for line in record['messages'][1]['content'].split('\n'):
        if line.startswith('Date: '):
            return line[len('Date: '):]
    return ''


def get_ticker(record: dict) -> str:
    for line in record['messages'][1]['content'].split('\n'):
        if line.startswith('Ticker: '):
            return line[len('Ticker: '):]
    return ''


def get_move_bin(record: dict) -> str:
    content = record['messages'][2]['content']
    return json.loads(content).get('move_bin', 'unknown')


def load_all(data_dir: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(data_dir, '*.jsonl'))):
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def normalized_entropy(counts: Counter) -> float:
    import math
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    if len(probs) <= 1:
        return 0.0
    ent = -sum(p * math.log(p) for p in probs)
    return round(ent / math.log(5), 3)  # 5 possible move_bin classes


def report_distribution(records: list[dict], label: str) -> None:
    if not records:
        print(f"\n--- {label}: 0 rows ---")
        return
    bins = Counter(get_move_bin(r) for r in records)
    by_ticker = defaultdict(Counter)
    for r in records:
        by_ticker[get_ticker(r)][get_move_bin(r)] += 1

    print(f"\n--- {label}: {len(records):,} rows ---")
    for b, c in bins.most_common():
        print(f"  {b:<12} {c:>6}  ({c/len(records)*100:.1f}%)")
    print(f"  overall bucket entropy: {normalized_entropy(bins)}")
    print(f"  tickers represented: {len(by_ticker)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/train_data', help='Directory of per-ticker .jsonl files')
    parser.add_argument('--out-dir', default='data/combined')
    parser.add_argument('--val-frac', type=float, default=0.1, help='Fraction of dates held out for validation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    records = load_all(args.data_dir)
    if not records:
        raise SystemExit(f"No .jsonl files found in {args.data_dir}")

    print(f"Loaded {len(records):,} rows from {args.data_dir}")
    report_distribution(records, "Combined (pre-split)")

    # Date-based split — same date can appear across multiple tickers, but a
    # given date must land entirely in train or entirely in val to avoid
    # leakage (near-identical rows sharing labels ending up split across
    # train/val).
    dates = sorted(set(get_date(r) for r in records))
    random.seed(args.seed)
    random.shuffle(dates)
    split_idx = int(len(dates) * (1 - args.val_frac))
    train_dates = set(dates[:split_idx])
    val_dates = set(dates[split_idx:])

    train_records = [r for r in records if get_date(r) in train_dates]
    val_records = [r for r in records if get_date(r) in val_dates]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, 'train.jsonl')
    val_path = os.path.join(args.out_dir, 'val.jsonl')

    with open(train_path, 'w', encoding='utf-8') as fh:
        for r in train_records:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(val_path, 'w', encoding='utf-8') as fh:
        for r in val_records:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')

    report_distribution(train_records, f"Train ({len(train_dates)} dates)")
    report_distribution(val_records, f"Val ({len(val_dates)} dates)")

    print(f"\nWrote {train_path} and {val_path}")


if __name__ == '__main__':
    main()