r"""
label_sentiment.py — scores articles using local LLM via llm_wrapper API.

Usage:
  python label_sentiment.py --data-dir data/raw --workers 1
  python label_sentiment.py --data-dir data/raw --force   # re-score everything
"""

import os
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from langdetect import detect
from dotenv import load_dotenv

from utils import get_trusted_sources

load_dotenv(dotenv_path="../.env")

CLIENT = OpenAI(
    base_url=f"http://{os.getenv('HOST', '127.0.0.1')}:{os.getenv('PORT', '8091')}/v1",
    api_key="not-needed",
)

SENTIMENT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "Phi-3-mini-128k-instruct-cuda-gpu:2")
TRUSTED_SOURCES = get_trusted_sources()

# Bump this string whenever the prompt changes so you can tell which
# articles need re-scoring after a prompt update.
PROMPT_VERSION = "v2"

# ---------------------------------------------------------------------------
# Instrument descriptions — tells the model how to interpret sentiment
# for each instrument type. Commodity/bond ETFs need the inversion note.
# ---------------------------------------------------------------------------
INSTRUMENT_DESCRIPTIONS = {
    # equities
    "NVDA": "equity – semiconductor/AI chipmaker (Nvidia)",
    "AAPL": "equity – consumer tech (Apple)",
    "MSFT": "equity – enterprise software/cloud (Microsoft)",
    "META": "equity – social media/AI (Meta Platforms)",
    "GOOGL": "equity – search/cloud/AI (Alphabet)",
    "AMZN": "equity – e-commerce/cloud (Amazon)",
    "TSLA": "equity – EV manufacturer (Tesla)",
    "INTC": "equity – semiconductor (Intel)",
    # sector ETFs
    "SOXX": "sector ETF – US semiconductor companies",
    "BOTZ": "sector ETF – global robotics and AI companies",
    "DRIV": "sector ETF – electric vehicles and autonomous driving",
    "SKYY": "sector ETF – cloud computing companies",
    "XLF": "sector ETF – US financial sector",
    "XLK": "sector ETF – US technology sector",
    "XLV": "sector ETF – US healthcare sector",
    # commodity / macro ETFs — note: rising underlying price = ETF goes UP
    "USO": "commodity ETF – tracks crude oil price (rising oil prices = bullish for USO)",
    "TLT": "bond ETF – tracks 20yr US Treasuries (falling interest rates = bullish for TLT)",
    "UUP": "currency ETF – tracks US dollar index (dollar strengthening = bullish for UUP)",
    # broad market
    "SPY": "index ETF – tracks S&P 500 (broad US market proxy)",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

USER_PROMPT = """
You are a quantitative analyst. Output ONLY a single valid JSON object. 
No markdown fences, no explanation, no text before or after the JSON.

Analyze this article's effect on {ticker} ({instrument_description}).

STEP 1 — Relevance: how directly does this article concern {ticker}?
  "high"   = article is primarily about {ticker}
  "medium" = sector or macro news with indirect effect on {ticker}
  "low"    = {ticker} is barely mentioned or article is unrelated

STEP 2 — Sentiment: expected price impact on {ticker} (score 1–5):
  5 = strongly positive (major deal, record beat, key regulatory win)
  4 = mildly positive (solid results, partnership, cost improvement)
  3 = unclear or negligible impact
  2 = mildly negative (miss, leadership risk, competitive loss)
  1 = strongly negative (fraud, major miss, sanctions, product failure)

Important: score price impact on {ticker}, not general news tone.
Example: "oil prices surge on conflict" → sentiment=5 for USO (oil up = USO up).

ARTICLE:
{text}

Respond with this exact JSON and nothing else:
{{"ticker":"{ticker}","relevance":"low|medium|high","sentiment":1|2|3|4|5,"direction":"up|down|flat","summary":"two sentences max","reason":"one sentence: why this score for {ticker} price"}}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except Exception:
        return True  # give benefit of the doubt


def _extract_json(raw: str) -> dict:
    """
    Pull the first {...} block out of raw LLM output.
    Raises ValueError if no valid JSON object is found.
    Does NOT fall back to a default — callers handle that.
    """
    # Strip markdown fences if the model wrapped the output anyway
    if "```" in raw:
        parts = raw.split("```")
        # parts[1] is the content between the first pair of fences
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in output: {raw[:120]!r}")

    return json.loads(raw[start : end + 1])


RELEVANCE_MAP = {"low": 1, "medium": 2, "high": 3}

def _validate_result(result: dict, ticker: str) -> dict:
    sentiment = result.get("sentiment")
    relevance_raw = result.get("relevance")
    direction = result.get("direction", "flat")

    # Sentiment must be int 1–5
    try:
        sentiment = int(sentiment)
    except (TypeError, ValueError):
        raise ValueError(f"Non-integer sentiment: {sentiment!r}")
    if sentiment not in range(1, 6):
        raise ValueError(f"sentiment={sentiment} out of range 1–5")

    # Relevance: accept "low"/"medium"/"high" (new) or int 1–5 (clamp to 1–3)
    if isinstance(relevance_raw, str):
        relevance = RELEVANCE_MAP.get(relevance_raw.lower().strip())
        if relevance is None:
            raise ValueError(f"Unrecognised relevance string: {relevance_raw!r}")
    else:
        try:
            relevance = int(relevance_raw)
        except (TypeError, ValueError):
            raise ValueError(f"Unparseable relevance: {relevance_raw!r}")
        # Clamp: model occasionally copies the 1–5 scale — treat 4–5 as 3
        relevance = min(relevance, 3)
        if relevance < 1:
            raise ValueError(f"relevance={relevance} below minimum 1")

    if direction not in ("up", "down", "flat"):
        direction = "flat"

    return {
        "ticker":           result.get("ticker", ticker),
        "relevance":        relevance,
        "sentiment":        sentiment,
        "direction":        direction,
        "summary":          str(result.get("summary", ""))[:500],
        "sentiment_reason": str(result.get("reason", ""))[:300],
        "prompt_version":   PROMPT_VERSION,
    }

from utils import (
    get_tickers,
    get_benchmarks,
    get_trusted_sources,
    get_private_companies,
)

ALL_TICKERS = get_tickers()
BENCHMARKS = get_benchmarks()
TRUSTED_SOURCES = get_trusted_sources()
PRIVATE_COMPANIES = get_private_companies()


def get_ticker(gdelt_query):
    return ALL_TICKERS.get(gdelt_query)


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------
def score_article(json_path: str, force: bool = False) -> dict | None:
    with open(json_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    # Skip if already scored with the current prompt version (unless --force)
    already_scored = (
        meta.get("sentiment") is not None
        and meta.get("prompt_version") == PROMPT_VERSION
    )
    if already_scored and not force:
        return None

    # Source filter
    if TRUSTED_SOURCES and meta.get("site") not in TRUSTED_SOURCES:
        return None

    # Resolve ticker — required for instrument-aware scoring
    # ticker = meta.get("ticker", "").upper()
    gdelt_query = meta.get("gdelt_query", "")
    ticker = get_ticker(gdelt_query)

    if not ticker:
        print(f"  [SKIP] No ticker in metadata: {json_path}")
        return None

    instrument_description = INSTRUMENT_DESCRIPTIONS.get(
        ticker, f"financial instrument ({ticker})"
    )

    # Load article text which is at same level as json file
    text_path = meta.get("text_path") # this could be article_guid.txt or /data/raw/site/date/article_guid.txt
    text_file_name = Path(text_path).name
    text_file_path = Path(json_path).resolve().parent / text_file_name

    print(f"scoring {text_file_path} for {ticker}...")

    if not text_file_path or not os.path.exists(text_file_path):
        # text_path = text_path.replace("../", "") if text_path else None  # try relative path
        # if not text_path or not os.path.exists(text_path):
        print(f"  [SKIP] Text file missing: {text_file_path}")
        return None

    with open(text_file_path, "r", encoding="utf-8") as fh:
        text = fh.read()[:3000]

    if len(text.split()) < 50:
        return None

    if not is_english(text):
        print(f"  [SKIP] Non-English: {json_path}")
        return None

    # Call the model
    raw = ""
    try:
        user_content = USER_PROMPT.format(
                    ticker=ticker,
                    instrument_description=instrument_description,
                    text=text
                )
        # print(f"  [DEBUG] user_content={user_content}")

        messages=[
            {
                "role": "user",
                "content": user_content,
            },
        ]
        response = CLIENT.chat.completions.create(
            model=SENTIMENT_MODEL,
            messages=messages,
            temperature=0.0,  # deterministic — same article = same score
        )
        raw = response.choices[0].message.content.strip()
        result_raw = _extract_json(raw)
        result = _validate_result(result_raw, ticker)

    except (ValueError, json.JSONDecodeError) as e:
        # Do NOT salvage as neutral — that corrupts training labels.
        # Log it and let the caller decide whether to retry.
        print(f"  [BAD OUTPUT] {json_path}: {e} \n raw={raw}")
        return None
    except Exception as e:
        print(f"ERROR: {e.response.text if hasattr(e, 'response') else str(e)}")
        return None

    # Write back only the new fields — preserve everything else in meta
    meta.update(result)


    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    relevance_flag = " [LOW-REL]" if result["relevance"] == 1 else ""
    print(
        f"  [OK] {meta.get('site',''):<20} {ticker:<5} "
        f"rel={result['relevance']} sent={result['sentiment']} "
        f"dir={result['direction']:<5}{relevance_flag} | "
        f"{result['summary'][:60]}"
    )
    return meta


# ---------------------------------------------------------------------------
# Collection & entry point
# ---------------------------------------------------------------------------
def collect_json_paths(data_dir: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".json"):
                print(f"Found JSON: {os.path.join(root, f)}")
                paths.append(os.path.join(root, f))
    return paths


def main(data_dir: str, workers: int, sleep: float, force: bool) -> None:
    print(f"Collecting article JSON files from {data_dir}...")

    paths = collect_json_paths(data_dir)

    print(f"Found {len(paths)} articles  |  prompt version: {PROMPT_VERSION}")

    if TRUSTED_SOURCES:
        print(f"Source filter: {len(TRUSTED_SOURCES)} trusted sites")
    if force:
        print("--force enabled: re-scoring all articles regardless of existing labels")
    print()

    labeled = skipped = failed = 0

    # NOTE: workers > 1 won't speed up a local GPU model — Phi-3-mini
    # serializes on the GPU regardless. Keep workers=1 unless you have
    # multiple model servers running on different ports.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(score_article, p, force): p for p in paths}
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                skipped += 1
            else:
                labeled += 1
            if sleep:
                time.sleep(sleep)

    print(f"\nDone — labeled: {labeled}  skipped: {skipped}  failed: {failed}")
    print(
        f"Tip: articles with relevance=1 are tagged [LOW-REL] above. "
        f"Filter them out before building your training set."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score articles with local LLM")
    parser.add_argument(
        "--data-dir", default="data/raw", help="Root of article JSON store"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel threads (keep=1 for single GPU)",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="Seconds between requests"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-score already-labeled articles"
    )
    args = parser.parse_args()
    main(args.data_dir, args.workers, args.sleep, args.force)
