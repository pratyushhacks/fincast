r"""
experiment_planner.py — reads analysis/ CSVs and uses LLM to suggest
optimal train/val/test split and sampling strategy for build_dataset.py.

Phase 1: dataset analysis only
Phase 2 (future): also reads training_logs/ and inference_outputs/

Usage:
  python experiment_planner.py
  python experiment_planner.py --out experiment_plan.json
"""
import os
import json
import argparse

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

CLIENT = OpenAI(
    base_url=f"http://{os.getenv('HOST','127.0.0.1')}:{os.getenv('PORT','8091')}/v1",
    api_key="not-needed"
)
MODEL = os.getenv("DEFAULT_CHAT_MODEL", "phi-3-mini-128k-instruct-cuda-gpu:1")

ANALYSIS_DIR = "analysis"

SYSTEM_PROMPT = (
    "You are a machine learning dataset strategist specializing in financial NLP fine-tuning. "
    "You respond ONLY with valid JSON. No markdown, no explanation, no extra text."
)

PLANNER_PROMPT = """You are advising on the optimal dataset construction strategy for fine-tuning 
a language model to predict next-day stock price movement from news sentiment.

Here is the current dataset analysis:

=== READINESS SUMMARY ===
{readiness}

=== SENTIMENT DISTRIBUTION ===
{sentiment}

=== PRICE BUCKET DISTRIBUTION ===
{buckets}

=== PRICE BUCKETS PER TICKER ===
{buckets_per_ticker}

=== MISMATCH RATE PER TICKER ===
{mismatch}

Based on this data, provide a dataset construction plan as JSON with these exact fields:

{{
  "assessment": "2-3 sentences on overall data quality and readiness",
  "total_examples_target": <int: how many total JSONL examples to aim for>,
  "train_pct": <int: % for training set>,
  "val_pct":   <int: % for validation set>,
  "test_pct":  <int: % for test set>,
  "sampling_strategy": {{
    "flat_oversample":      <bool: should we oversample non-flat buckets>,
    "flat_cap_pct":         <int: max % of examples that can be flat, e.g. 40>,
    "min_articles_per_day": <int: min articles per ticker per day to include>,
    "exclude_tickers":      [<list of tickers with too few examples to train on>]
  }},
  "data_gaps": [
    "gap 1 description",
    "gap 2 description"
  ],
  "recommendations": [
    "actionable recommendation 1",
    "actionable recommendation 2"
  ],
  "ready_to_train": <bool>
}}"""


def load_csv_summary(filename, max_rows=50):
    path = os.path.join(ANALYSIS_DIR, filename)
    if not os.path.exists(path):
        return f"[{filename} not found — run analyze_data.ipynb first]"
    try:
        df = pd.read_csv(path, index_col=0)
        return df.head(max_rows).to_string()
    except Exception as e:
        return f"[Error loading {filename}: {e}]"


def build_prompt():
    return PLANNER_PROMPT.format(
        readiness          = load_csv_summary('dataset_readiness.csv'),
        sentiment          = load_csv_summary('dist_sentiment.csv'),
        buckets            = load_csv_summary('dist_price_buckets.csv'),
        buckets_per_ticker = load_csv_summary('dist_price_buckets_per_ticker.csv'),
        mismatch           = load_csv_summary('mismatch_rate_per_ticker.csv'),
    )


def run_planner(out_path):
    # Check analysis CSVs exist
    required = [
        'dataset_readiness.csv',
        'dist_sentiment.csv',
        'dist_price_buckets.csv',
        'dist_price_buckets_per_ticker.csv',
        'mismatch_rate_per_ticker.csv',
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(ANALYSIS_DIR, f))]
    if missing:
        print(f"Missing CSVs: {missing}")
        print("Run analyze_data.ipynb first to generate analysis/ CSVs")
        return

    print("Reading analysis CSVs...")
    prompt = build_prompt()

    print("Calling LLM for dataset strategy...")
    raw = ""
    try:
        response = CLIENT.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        start = raw.find('{')
        end   = raw.rfind('}')
        if start != -1 and end != -1:
            raw = raw[start:end+1]

        plan = json.loads(raw)

        # Save plan
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(plan, fh, ensure_ascii=False, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT PLAN")
        print("="*60)
        print(f"Assessment:        {plan.get('assessment','')}")
        print(f"Ready to train:    {plan.get('ready_to_train')}")
        print(f"Target examples:   {plan.get('total_examples_target')}")
        print(f"Split:             {plan.get('train_pct')}% train / "
              f"{plan.get('val_pct')}% val / "
              f"{plan.get('test_pct')}% test")
        sampling = plan.get('sampling_strategy', {})
        print(f"Flat cap:          {sampling.get('flat_cap_pct')}%")
        print(f"Oversample non-flat: {sampling.get('flat_oversample')}")
        print(f"Min articles/day:  {sampling.get('min_articles_per_day')}")
        print(f"Exclude tickers:   {sampling.get('exclude_tickers', [])}")
        print(f"\nData gaps:")
        for g in plan.get('data_gaps', []):
            print(f"  - {g}")
        print(f"\nRecommendations:")
        for r in plan.get('recommendations', []):
            print(f"  - {r}")
        print(f"\nPlan saved to: {out_path}")

        return plan

    except json.JSONDecodeError:
        print(f"[BAD JSON] {raw[:200]}")
    except Exception as e:
        print(f"[ERROR] {e}")
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='experiment_plan.json')
    args = parser.parse_args()
    run_planner(args.out)
