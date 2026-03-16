r"""
label_sentiment.py — scores articles using local LLM via llm_wrapper API
"""
import os
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

CLIENT = OpenAI(
    base_url="http://127.0.0.1:8091/v1",
    api_key="not-needed"
)

SENTIMENT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "Phi-3-mini-128k-instruct-cuda-gpu:1")

SENTIMENT_PROMPT = """You are a financial market analyst. 
Read the article and respond with JSON:

1. summary: max 300 tokens, capture the key market-relevant facts
2. tags: which of these apply: {watchlist}
3. sentiment: market impact score 1-5 where:
     1=strongly negative (war, crisis, crash, sanctions)
     2=negative (rate hikes, recession fears, earnings miss)
     3=neutral (routine news, no clear market impact)
     4=positive (rate cuts, strong earnings, trade deals)
     5=strongly positive (major breakthrough, record growth)
4. reason: one sentence explaining the market impact

Article title: {title}
Article text: {text}

Respond ONLY with JSON, no markdown:
{{"summary": "...", "sentiment": 2, "reason": "..."}}"""


def load_watchlist(config_path="gdelt_config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh) or {}
        wl = cfg.get('watchlist', {})
        all_terms = (
            wl.get('companies', []) +
            wl.get('sectors',   []) +
            wl.get('macro',     [])
        )
        return all_terms
    except Exception:
        return []

def load_trusted_sources(config_path="gdelt_config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh) or {}
        return set(cfg.get('trusted_sources', []))
    except Exception:
        return set()

WATCHLIST = load_watchlist()
TRUSTED_SOURCES = load_trusted_sources()

def score_article(json_path):
    with open(json_path, 'r', encoding='utf-8') as fh:
        meta = json.load(fh)

    if TRUSTED_SOURCES and meta.get('site') not in TRUSTED_SOURCES:
        return None
    
    if meta.get('sentiment'):
        return None

    text_path = meta.get('text_path')
    if not text_path or not os.path.exists(text_path):
        return None

    with open(text_path, 'r', encoding='utf-8') as fh:
        text = fh.read()[:3000]

    # Skip very short articles
    if len(text.split()) < 50:
        return None

    prompt = SENTIMENT_PROMPT.format(
        watchlist=', '.join(WATCHLIST),
        title=meta.get('title', ''),
        text=text
    )

    try:
        response = CLIENT.chat.completions.create(
            model=SENTIMENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,  # bumped slightly for summary
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if Phi wraps output
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)

        meta['summary']          = result.get('summary', '')
        meta['sentiment']        = result.get('sentiment')
        meta['sentiment_reason'] = result.get('reason', '')

        with open(json_path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        print(f"  [OK] {meta['site']} | sentiment={meta['sentiment']} | summary={meta['summary'][:60]}")
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
    print(f"Found {len(paths)} articles to label\n")

    if TRUSTED_SOURCES:
        print(f"Trusted sources filter active: {len(TRUSTED_SOURCES)} sites\n")
    else:
        print("No source filter — processing all articles\n")

    # Local model is single-threaded — workers=1 avoids overwhelming it
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(score_article, p): p for p in paths}
        for future in as_completed(futures):
            future.result()
            time.sleep(sleep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw')
    parser.add_argument('--workers', type=int, default=1)  # default 1 for local model
    parser.add_argument('--sleep', type=float, default=0.0)
    args = parser.parse_args()
    main(args.data_dir, args.workers, args.sleep)