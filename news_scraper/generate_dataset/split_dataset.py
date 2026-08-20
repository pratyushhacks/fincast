"""Split a build_dataset.py output into per-ticker train/eval/test JSONL files.

This script reads an existing dataset JSONL produced by build_dataset.py and
writes one directory per ticker. Each ticker directory contains:
  - data.jsonl   : all examples with `_meta.split`
  - train.jsonl  : 70% of examples for training
  - eval.jsonl   : 20% of examples for evaluation
  - test.jsonl   : 10% of examples for testing

The split is computed per ticker and is deterministic by default.
"""

import argparse
import json
import os
import random
from collections import defaultdict


DEFAULT_RATIOS = {
    'train': 0.7,
    'eval': 0.2,
    'test': 0.1,
}


def collect_examples(input_path):
    examples = []
    with open(input_path, 'r', encoding='utf-8') as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {input_path}: {exc}")
    return examples


def get_ticker_from_example(example):
    meta = example.get('_meta', {})
    ticker = meta.get('ticker')
    if not ticker:
        raise ValueError("Example is missing _meta.ticker and cannot be assigned to a ticker")
    return ticker


def compute_split_counts(n, ratios):
    if n <= 0:
        return {'train': 0, 'eval': 0, 'test': 0}

    raw_counts = {
        name: int(n * ratio)
        for name, ratio in ratios.items()
    }
    remainder = n - sum(raw_counts.values())
    order = ['train', 'eval', 'test']
    for i in range(remainder):
        raw_counts[order[i % len(order)]] += 1

    if raw_counts['train'] == 0:
        raw_counts['train'] = 1

    if n >= 2 and raw_counts['test'] == 0:
        raw_counts['test'] = 1
    if n >= 3 and raw_counts['eval'] == 0:
        raw_counts['eval'] = 1

    while sum(raw_counts.values()) > n:
        for name in order[::-1]:
            if raw_counts[name] > 1 and sum(raw_counts.values()) > n:
                raw_counts[name] -= 1
    return raw_counts


def split_examples(examples, ratios, seed=None):
    if seed is not None:
        random.Random(seed).shuffle(examples)
    counts = compute_split_counts(len(examples), ratios)

    split_order = []
    split_order.extend(['train'] * counts['train'])
    split_order.extend(['eval'] * counts['eval'])
    split_order.extend(['test'] * counts['test'])

    if len(split_order) != len(examples):
        raise AssertionError("Split counts do not sum to total examples")

    for example, split_name in zip(examples, split_order):
        example.setdefault('_meta', {})['split'] = split_name
    return {
        'train': examples[:counts['train']],
        'eval':  examples[counts['train']:counts['train'] + counts['eval']],
        'test':  examples[counts['train'] + counts['eval']:],
    }


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')


def build_per_ticker_splits(input_path, output_dir, ratios, seed=None):
    examples = collect_examples(input_path)
    by_ticker = defaultdict(list)
    for ex in examples:
        ticker = get_ticker_from_example(ex)
        by_ticker[ticker].append(ex)

    if not by_ticker:
        raise ValueError(f"No ticker examples found in {input_path}")

    summary = []
    for ticker, ticker_examples in sorted(by_ticker.items()):
        if not ticker_examples:
            continue

        split_examples_list = split_examples(list(ticker_examples), ratios, seed=seed)
        ticker_dir = os.path.join(output_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        write_jsonl(os.path.join(ticker_dir, 'data.jsonl'), ticker_examples)
        write_jsonl(os.path.join(ticker_dir, 'train.jsonl'), split_examples_list['train'])
        write_jsonl(os.path.join(ticker_dir, 'eval.jsonl'), split_examples_list['eval'])
        write_jsonl(os.path.join(ticker_dir, 'test.jsonl'), split_examples_list['test'])

        summary.append((ticker, len(ticker_examples), split_examples_list))

    print(f"Wrote ticker splits for {len(summary)} tickers to {output_dir}")
    for ticker, total, split_data in summary:
        counts = {k: len(v) for k, v in split_data.items()}
        print(f"  {ticker}: total={total}, train={counts['train']}, eval={counts['eval']}, test={counts['test']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create per-ticker train/eval/test split files from a build_dataset.py JSONL output.")
    parser.add_argument('--input', required=True,
                        help='Path to the dataset JSONL produced by build_dataset.py')
    parser.add_argument('--out-dir', required=True,
                        help='Directory where per-ticker split folders will be written')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deterministic shuffling')
    parser.add_argument('--train-ratio', type=float, default=DEFAULT_RATIOS['train'],
                        help='Proportion of examples assigned to train (default 0.7)')
    parser.add_argument('--eval-ratio', type=float, default=DEFAULT_RATIOS['eval'],
                        help='Proportion of examples assigned to eval (default 0.2)')
    parser.add_argument('--test-ratio', type=float, default=DEFAULT_RATIOS['test'],
                        help='Proportion of examples assigned to test (default 0.1)')
    return parser.parse_args()


def validate_ratios(train_ratio, eval_ratio, test_ratio):
    total = train_ratio + eval_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError('train_ratio + eval_ratio + test_ratio must equal 1.0')


if __name__ == '__main__':
    args = parse_args()
    validate_ratios(args.train_ratio, args.eval_ratio, args.test_ratio)
    build_per_ticker_splits(
        input_path=args.input,
        output_dir=args.out_dir,
        ratios={
            'train': args.train_ratio,
            'eval': args.eval_ratio,
            'test': args.test_ratio,
        },
        seed=args.seed,
    )
