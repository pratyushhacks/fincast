#!/usr/bin/env python3
"""
experiment_planner.py — reads analysis/ CSVs and uses an LLM to suggest
an optimal train/val/test split and sampling strategy for build_dataset.py.

Phase 1: dataset analysis only
Phase 2 (future): also reads training_logs/ and inference_outputs/

Usage:
  python experiment_planner.py
  python experiment_planner.py --out experiment_plan.json
"""
import os
import json
import argparse
import textwrap
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# JSON schema validation
try:
    from jsonschema import validate, ValidationError
except Exception as e:
    raise RuntimeError("Please install jsonschema: pip install jsonschema") from e

# OpenAI-compatible client (Foundry Local / phi-3-mini)
try:
    from openai import OpenAI
except Exception:
    # If the environment uses a different client, user should adapt this import.
    raise RuntimeError("Missing OpenAI client. Ensure 'openai' package is installed and available.")

# -------------------------
# Configuration
# -------------------------
load_dotenv(dotenv_path="../.env")
EXPERIMENT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "Phi-3-mini-128k-instruct-cuda-gpu:2")

# LLM call settings tuned for deterministic JSON output
LLM_TEMPERATURE = "0.0"
LLM_MAX_TOKENS = 800

ANALYSIS_DIR = os.path.join(os.pardir, "analyze_data", "analysis") #../news_scraper/analyze_data/analysis
OUTPUT_DIR =  os.path.curdir

CLIENT = OpenAI(
    base_url=f"http://{os.getenv('HOST', '127.0.0.1')}:{os.getenv('PORT', '8091')}/v1",
    api_key="not-needed",
)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------
# Prompts (phi-3 friendly)
# -------------------------
SYSTEM_PROMPT = (
    "You are a dataset strategist for financial NLP fine-tuning. "
    "Return exactly one JSON object and nothing else. "
    "Do not include markdown, commentary, or extra text. "
    "If you cannot produce a valid plan, return {\"ready_to_train\": false, \"errors\": [<explain>] }."
)

USER_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are advising on dataset construction for fine-tuning an LLM to predict next-day stock movement from news sentiment.

    Here are the dataset metrics:
    {metrics}

    Return a single JSON object with these exact fields and types:
    {{
      "assessment": "<string: 1-3 sentences>",
      "assumptions": ["string", ...],
      "total_examples_target": <int>,
      "train_pct": <int 0-100>,
      "val_pct": <int 0-100>,
      "test_pct": <int 0-100>,
      "flat_cap_pct": <int 0-100>,
      "flat_oversample": <bool>,
      "min_articles_per_day": <int>,
      "exclude_tickers": ["TICK1","TICK2", ...],
      "data_gaps": ["string", ...],
      "recommendations": ["string", ...],
      "ready_to_train": <bool>,
      "confidence": <float 0.0-1.0>
    }}

    For each numeric field, include a one-sentence rationale inside the 'recommendations' list (e.g., 'Set train_pct=70 because ...'). Ensure train_pct + val_pct + test_pct == 100.

    Example minimal valid output:
    {{
      "assessment": "Moderate dataset size with heavy neutral class.",
      "assumptions": ["sentiment mapped 1-5; flat=3 means no move"],
      "total_examples_target": 3000,
      "train_pct": 70,
      "val_pct": 15,
      "test_pct": 15,
      "flat_cap_pct": 40,
      "flat_oversample": false,
      "min_articles_per_day": 1,
      "exclude_tickers": ["TICK_A"],
      "data_gaps": ["Neutral class >50%"],
      "recommendations": ["Downsample neutral to 40% to improve directional signal."],
      "ready_to_train": false,
      "confidence": 0.3
    }}

    Now produce the JSON object only.
    """
)

# -------------------------
# JSON Schema for validation
# -------------------------
PLAN_SCHEMA = {
    "type": "object",
    "required": [
        "assessment",
        "assumptions",
        "total_examples_target",
        "train_pct",
        "val_pct",
        "test_pct",
        "flat_cap_pct",
        "flat_oversample",
        "min_articles_per_day",
        "exclude_tickers",
        "data_gaps",
        "recommendations",
        "ready_to_train",
        "confidence",
    ],
    "properties": {
        "assessment": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "total_examples_target": {"type": "integer", "minimum": 0},
        "train_pct": {"type": "integer", "minimum": 0, "maximum": 100},
        "val_pct": {"type": "integer", "minimum": 0, "maximum": 100},
        "test_pct": {"type": "integer", "minimum": 0, "maximum": 100},
        "flat_cap_pct": {"type": "integer", "minimum": 0, "maximum": 100},
        "flat_oversample": {"type": "boolean"},
        "min_articles_per_day": {"type": "integer", "minimum": 0},
        "exclude_tickers": {"type": "array", "items": {"type": "string"}},
        "data_gaps": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "ready_to_train": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


# -------------------------
# Utilities
# -------------------------
def load_csv(filename: str, index_col: Optional[Any] = None) -> pd.DataFrame:
    path = os.path.join(ANALYSIS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required analysis file not found: {path}")
    return pd.read_csv(path, index_col=index_col)


def safe_scalar(df: pd.DataFrame, idx: Any, col: Any, default: Any = None) -> Any:
    try:
        if idx in df.index and col in df.columns:
            return df.at[idx, col]
    except Exception:
        pass
    return default


def save_json(obj: Dict[str, Any], path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    return path


# -------------------------
# Metrics builder
# -------------------------
def build_metrics() -> Dict[str, Any]:
    """
    Load analysis CSVs and compute a compact metrics dict for the LLM.
    """
    readiness_df = load_csv("dataset_readiness.csv", index_col=None)
    sentiment_df = load_csv("dist_sentiment.csv", index_col=0)
    buckets_df = load_csv("dist_price_buckets.csv", index_col=0)
    buckets_per_ticker_df = load_csv("dist_price_buckets_per_ticker.csv", index_col=0)
    mismatch_df = load_csv("mismatch_rate_per_ticker.csv", index_col=0)

    # Basic counts
    total_examples = int(readiness_df["total_articles"].sum())
    total_ticker_groups = int(
        readiness_df[["ticker", "category"]].drop_duplicates().shape[0]
    )
    ready_tickers = int((readiness_df["readiness"] == "READY 200+").sum())
    medium_tickers = int((readiness_df["readiness"] == "MED 50-200").sum())
    low_tickers = int((readiness_df["readiness"] == "LOW <50").sum())

    high_flat_tickers = readiness_df[readiness_df["flat_pct"] > 70]["ticker"].unique().tolist()
    low_sample_tickers = readiness_df[readiness_df["total_articles"] < 50]["ticker"].unique().tolist()
    high_mismatch_tickers = (
        mismatch_df[mismatch_df.get("mismatch_pct", 0) >= 40].index.tolist()
        if "mismatch_pct" in mismatch_df.columns
        else []
    )

    overall_flat_pct = float(safe_scalar(buckets_df, "flat", "pct", 0) or 0)
    neutral_sentiment_pct = float(safe_scalar(sentiment_df, 3, "pct", 0) or 0)
    dataset_mismatch_pct = (
        float(mismatch_df["mismatch_pct"].mean()) if "mismatch_pct" in mismatch_df.columns else None
    )

    # Totals for directional buckets (best-effort extraction)
    def safe_count_from_buckets(df, key):
        try:
            if key in df.index and "count" in df.columns:
                return int(df.at[key, "count"])
        except Exception:
            pass
        return 0

    total_up = safe_count_from_buckets(buckets_df, "strong_up") + safe_count_from_buckets(buckets_df, "weak_up")
    total_down = safe_count_from_buckets(buckets_df, "strong_down") + safe_count_from_buckets(buckets_df, "weak_down")

    # Heuristic readiness
    ready_to_train = (
        total_examples >= 2000
        and overall_flat_pct <= 50
        and len(high_mismatch_tickers) <= max(3, int(total_ticker_groups * 0.15))
        and ready_tickers >= 3
    )

    # Target examples heuristic
    if total_examples < 500:
        target_examples = 2000
    elif total_examples < 2000:
        target_examples = 3000
    else:
        target_examples = total_examples

    # Data gaps
    data_gaps: List[str] = []
    if total_examples < 1000:
        data_gaps.append("Too few labeled examples for robust fine-tuning (<1000).")
    if overall_flat_pct > 50:
        data_gaps.append("Flat price moves dominate the dataset; consider downsampling flat examples or using a binary target.")
    if len(high_flat_tickers) > 0:
        data_gaps.append(f"High flat ratio tickers: {high_flat_tickers[:10]}")
    if len(high_mismatch_tickers) > 0:
        data_gaps.append(f"High mismatch tickers: {high_mismatch_tickers[:10]}")
    if len(low_sample_tickers) > 0:
        data_gaps.append(f"Tickers with low sample count: {low_sample_tickers[:10]}")
    if ready_tickers < 3:
        data_gaps.append("Too few ticker groups with 200+ examples for a generalizable model.")
    if neutral_sentiment_pct > 50:
        data_gaps.append("Neutral sentiment is the majority label, which may weaken directional training signals.")

    sampling_strategy = {
        "flat_oversample": overall_flat_pct > 40,
        "flat_cap_pct": 40 if overall_flat_pct > 40 else 50,
        "min_articles_per_day": 1,
        "exclude_tickers": sorted(set(low_sample_tickers + high_flat_tickers + high_mismatch_tickers))[:20],
    }

    metrics = {
        "total_examples": total_examples,
        "total_ticker_groups": total_ticker_groups,
        "ready_tickers": ready_tickers,
        "medium_tickers": medium_tickers,
        "low_tickers": low_tickers,
        "overall_flat_pct": overall_flat_pct,
        "neutral_sentiment_pct": neutral_sentiment_pct,
        "dataset_mismatch_pct": dataset_mismatch_pct,
        "high_flat_tickers": len(high_flat_tickers),
        "high_mismatch_tickers": len(high_mismatch_tickers),
        "low_sample_tickers": len(low_sample_tickers),
        "ready_to_train_heuristic": ready_to_train,
        "total_examples_target": target_examples,
        "sampling_strategy": sampling_strategy,
        "data_gaps": data_gaps,
        "recommendations": [
            "Exclude or downsample tickers with low sample counts and high flat ratios.",
            "Prefer a 70/15/15 split and cap flat examples at 40% if flat dominates.",
        ],
    }
    return metrics


# -------------------------
# Prompt builder
# -------------------------
def build_prompt(metrics: Dict[str, Any]) -> str:
    metrics_text = json.dumps(metrics, indent=2)
    return USER_PROMPT_TEMPLATE.format(metrics=metrics_text)


# -------------------------
# JSON extraction and validation
# -------------------------
def extract_json_text(raw_text: str) -> Optional[str]:
    """
    Heuristic extraction of the first JSON object in raw_text.
    Handles code fences and stray text.
    """
    if not raw_text:
        return None
    text = raw_text.strip()

    # Remove surrounding triple backticks blocks if present
    if text.startswith("```") and text.endswith("```"):
        # remove first and last fence lines
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    # Find the first balanced JSON object by scanning for braces
    start = text.find("{")
    if start == -1:
        return None

    # Attempt to find matching closing brace by simple stack
    stack = []
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack:
                    return text[start : i + 1]
    return None


def validate_plan_schema(plan: Dict[str, Any]) -> (bool, Optional[str]):
    try:
        validate(instance=plan, schema=PLAN_SCHEMA)
        # additional check: splits sum to 100
        if plan["train_pct"] + plan["val_pct"] + plan["test_pct"] != 100:
            return False, "train_pct + val_pct + test_pct must equal 100"
        return True, None
    except ValidationError as e:
        return False, str(e)


# -------------------------
# Planner runner
# -------------------------
def run_planner(out_path: str) -> Optional[Dict[str, Any]]:
    required = [
        "dataset_readiness.csv",
        "dist_sentiment.csv",
        "dist_price_buckets.csv",
        "dist_price_buckets_per_ticker.csv",
        "mismatch_rate_per_ticker.csv",
    ]
    
    missing = [f for f in required if not os.path.exists(os.path.join(ANALYSIS_DIR, f))]
    if missing:
        logging.error("Missing CSVs: %s", missing)
        logging.info("Run analyze_data.ipynb first to generate analysis/ CSVs")
        return None

    logging.info("Reading analysis CSVs and building metrics...")
    metrics = build_metrics()
    prompt = build_prompt(metrics)

    metrics_path = os.path.join(OUTPUT_DIR, os.path.basename(out_path).replace(".json", "_metrics.json"))
    save_json(metrics, metrics_path)
    logging.info("Metrics saved to: %s", metrics_path)

    # Call LLM
    logging.info("Calling LLM for dataset strategy (model=%s)...", EXPERIMENT_MODEL)
    raw = ""
    plan: Dict[str, Any] = {}

    def call_llm():
        nonlocal raw
        response = CLIENT.chat.completions.create(
            model=EXPERIMENT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        # Compatible with Foundry Local / OpenAI-like responses
        raw_resp = ""
        try:
            raw_resp = response.choices[0].message.content
        except Exception:
            # fallback for different client shapes
            raw_resp = getattr(response, "content", "") or str(response)
        raw = raw_resp.strip()
        return raw

    # First attempt
    try:
        call_llm()
    except Exception as e:
        logging.exception("LLM call failed: %s", e)
        return None

    # Extract JSON
    json_text = extract_json_text(raw)
    if not json_text:
        logging.warning("No JSON object found in model output. Saving raw output for inspection.")
        debug_raw = os.path.join(OUTPUT_DIR, "debug_plan_raw.txt")
        with open(debug_raw, "w", encoding="utf-8") as fh:
            fh.write(raw)
        # Retry once with a stricter system prompt
        logging.info("Retrying once with a stricter system prompt...")
        try:
            response = CLIENT.chat.completions.create(
                model=EXPERIMENT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only the JSON object matching the schema. No text, no markdown.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.0,
            )
            raw_retry = response.choices[0].message.content.strip()
            json_text = extract_json_text(raw_retry)
            if json_text:
                raw = raw_retry
        except Exception as e:
            logging.exception("Retry LLM call failed: %s", e)

    if not json_text:
        logging.error("Failed to extract JSON from model output. See %s", os.path.join(OUTPUT_DIR, "debug_plan_raw.txt"))
        plan = {"ready_to_train": False, "errors": ["No JSON found in model output. See debug_plan_raw.txt"]}
        save_json(plan, out_path)
        return plan

    # Parse JSON
    try:
        plan = json.loads(json_text)
    except Exception as e:
        logging.exception("JSON parse error: %s", e)
        with open(os.path.join(OUTPUT_DIR, "debug_plan_raw.txt"), "w", encoding="utf-8") as fh:
            fh.write(raw)
        plan = {"ready_to_train": False, "errors": [f"JSON load error: {e}"]}
        save_json(plan, out_path)
        return plan

    # Validate schema
    is_valid, err = validate_plan_schema(plan)
    if not is_valid:
        logging.warning("Schema validation failed: %s", err)
        # Save debug artifacts
        with open(os.path.join(OUTPUT_DIR, "debug_plan_raw.txt"), "w", encoding="utf-8") as fh:
            fh.write(raw)
        with open(os.path.join(OUTPUT_DIR, "debug_plan_parsed.json"), "w", encoding="utf-8") as fh:
            json.dump(plan, fh, indent=2)
        # Retry once with stricter instruction
        logging.info("Retrying once with stricter instruction to return only JSON matching schema...")
        try:
            response = CLIENT.chat.completions.create(
                model=EXPERIMENT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only the JSON object matching the schema exactly. No extra text.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.0,
            )
            raw_retry = response.choices[0].message.content.strip()
            json_text_retry = extract_json_text(raw_retry)
            if json_text_retry:
                try:
                    plan_retry = json.loads(json_text_retry)
                    is_valid_retry, err_retry = validate_plan_schema(plan_retry)
                    if is_valid_retry:
                        plan = plan_retry
                        logging.info("Retry produced a valid plan.")
                    else:
                        logging.warning("Retry still invalid: %s", err_retry)
                except Exception as e:
                    logging.exception("Retry JSON parse error: %s", e)
        except Exception as e:
            logging.exception("Retry LLM call failed: %s", e)

    # Final validation check
    is_valid_final, err_final = validate_plan_schema(plan)
    if not is_valid_final:
        logging.error("Final plan failed validation: %s", err_final)
        # Save final artifacts and return safe plan
        with open(os.path.join(OUTPUT_DIR, "debug_plan_final_raw.txt"), "w", encoding="utf-8") as fh:
            fh.write(raw)
        with open(os.path.join(OUTPUT_DIR, "debug_plan_final_parsed.json"), "w", encoding="utf-8") as fh:
            json.dump(plan, fh, indent=2)
        safe_plan = {
            "assessment": "Failed to produce a valid plan automatically.",
            "assumptions": [],
            "total_examples_target": metrics.get("total_examples_target", 0),
            "train_pct": 70,
            "val_pct": 15,
            "test_pct": 15,
            "flat_cap_pct": 40,
            "flat_oversample": False,
            "min_articles_per_day": 1,
            "exclude_tickers": [],
            "data_gaps": metrics.get("data_gaps", []),
            "recommendations": ["Manual review required: LLM output failed schema validation."],
            "ready_to_train": False,
            "confidence": 0.0,
            "errors": [f"Schema validation failed: {err_final}"],
        }
        save_json(safe_plan, out_path)
        logging.info("Saved safe fallback plan to %s", out_path)
        return safe_plan

    # Save valid plan
    save_json(plan, out_path)
    logging.info("Experiment plan saved to: %s", out_path)

    # Print a concise summary for the user
    print("\n" + "=" * 60)
    print("EXPERIMENT PLAN")
    print("=" * 60)
    print(f"Assessment:        {plan.get('assessment','')}")
    print(f"Ready to train:    {plan.get('ready_to_train')}")
    print(f"Target examples:   {plan.get('total_examples_target')}")
    print(f"Split:             {plan.get('train_pct')}% train / {plan.get('val_pct')}% val / {plan.get('test_pct')}% test")
    print(f"Flat cap:          {plan.get('flat_cap_pct')}%")
    print(f"Oversample non-flat: {plan.get('flat_oversample')}")
    print(f"Min articles/day:  {plan.get('min_articles_per_day')}")
    print(f"Exclude tickers:   {plan.get('exclude_tickers', [])}")
    print("\nData gaps:")
    for g in plan.get("data_gaps", []):
        print(f"  - {g}")
    print("\nRecommendations:")
    for r in plan.get("recommendations", []):
        print(f"  - {r}")
    print(f"\nPlan saved to: {out_path}\n")

    return plan


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an experiment plan from analysis CSVs.")
    parser.add_argument("--out", default="experiment_plan.json", help="Output path for the experiment plan JSON")
    args = parser.parse_args()
    try:
        run_planner(args.out)
    except FileNotFoundError as e:
        logging.error(str(e))
        raise SystemExit(1)
    except Exception as e:
        logging.exception("Unexpected error running planner: %s", e)
        raise SystemExit(2)