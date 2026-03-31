r"""
label_sentiment.py — scores articles using local LLM via llm_wrapper API.

Usage:
  python label_sentiment.py --data-dir data/raw --workers 1
"""
import os
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from langdetect import detect
from dotenv import load_dotenv

from utils import get_trusted_sources

load_dotenv(dotenv_path=".env")

CLIENT = OpenAI(
    base_url=f"http://{os.getenv('HOST', '127.0.0.1')}:{os.getenv('PORT', '8091')}/v1",
    api_key="not-needed"
)

SENTIMENT_MODEL   = os.getenv("DEFAULT_CHAT_MODEL", "phi-3-mini-128k-instruct-cuda-gpu:1")
TRUSTED_SOURCES   = get_trusted_sources()

SYSTEM_PROMPT = (
    "You are a financial market analyst. "
    "You respond ONLY with valid JSON. No markdown, no explanation, no extra text."
)

USER_PROMPT = """Read the article and respond with clean JSON with exactly these keys:

1. summary: a couple of sentences capturing the key market-relevant facts
2. sentiment: market impact score 1-5 where:
     1=strongly negative (war, crisis, crash, sanctions)
     2=negative (rate hikes, recession fears, earnings miss)
     3=neutral (routine news, no clear market impact)
     4=positive (rate cuts, strong earnings, trade deals)
     5=strongly positive (major breakthrough, record growth)
3. reason: one sentence explaining the market impact

ARTICLE TEXT: {text}

Respond ONLY with JSON, no markdown:
{{"summary": "...", "sentiment": <1-5>, "reason": "..."}}"""


def is_english(text):
    try:
        return detect(text) == 'en'
    except Exception:
        return True


def score_article(json_path):
    with open(json_path, 'r', encoding='utf-8') as fh:
        meta = json.load(fh)

    if TRUSTED_SOURCES and meta.get('site') not in TRUSTED_SOURCES:
        return None

    if meta.get('sentiment'):
        return None

    text_path = meta.get('text_path')
    if not text_path or not os.path.exists(text_path):
        print(f"  [SKIP] Text file not found: {text_path}")
        return None

    with open(text_path, 'r', encoding='utf-8') as fh:
        text = fh.read()[:3000]

    if len(text.split()) < 50:
        return None

    if not is_english(text):
        print(f"  [SKIP] Non-English: {json_path}")
        return None

    raw = ""
    try:
        response = CLIENT.chat.completions.create(
            model=SENTIMENT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT.format(text=text)},
            ],
            max_tokens=500,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model wraps output
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # Extract JSON object in case of extra text
        start = raw.find('{')
        end   = raw.rfind('}')
        if start == -1 or end == -1:
            # Model returned plain text — salvage as neutral
            print(f"  [PLAIN TEXT] Salvaging: {json_path}")
            meta['summary']          = raw[:200]
            meta['sentiment']        = 3
            meta['sentiment_reason'] = 'Auto-salvaged from plain text response'
        else:
            raw    = raw[start:end+1]
            result = json.loads(raw)
            meta['summary']          = result.get('summary', '')
            meta['sentiment']        = result.get('sentiment')
            meta['sentiment_reason'] = result.get('reason', '')

        with open(json_path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        print(f"  [OK] {meta.get('site',''):<20} sentiment={meta['sentiment']} | {meta['summary'][:60]}")
        return meta

    except json.JSONDecodeError:
        print(f"  [BAD JSON] {json_path}: {raw[:100]}")
    except Exception as e:
        print(f"  [ERROR] {json_path}: {e}")
    return None


def collect_json_paths(data_dir):
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.json'):
                paths.append(os.path.join(root, f))
    return paths


def main(data_dir, workers, sleep):
    paths = collect_json_paths(data_dir)
    print(f"Found {len(paths)} articles")
    print(f"Trusted sources filter: {len(TRUSTED_SOURCES)} sites\n" if TRUSTED_SOURCES else "No source filter\n")

    labeled, skipped = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(score_article, p): p for p in paths}
        for future in as_completed(futures):
            result = future.result()
            if result:
                labeled += 1
            else:
                skipped += 1
            time.sleep(sleep)

    print(f"\nDone: {labeled} labeled, {skipped} skipped")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw')
    parser.add_argument('--workers',  type=int,   default=1)
    parser.add_argument('--sleep',    type=float, default=0.0)
    args = parser.parse_args()
    main(args.data_dir, args.workers, args.sleep)
