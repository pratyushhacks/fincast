"""
predict_ticker.py — live prediction(s) for one or all supported tickers.

Reuses build_ticker_jsonl.py's functions directly (quality filter, ticker
resolution, LLM dedup, top-5 signal-strength ranking, build_user_content)
so the payload is constructed EXACTLY the way training examples were —
any drift here would be train/serve skew, quietly degrading a model that
actually learned something useful.

Two phases, split mainly so the fine-tuned model only gets loaded ONCE
instead of once per ticker:

  Phase 1 (build): for each ticker, find today's qualifying articles, dedup
  via Foundry Local (same as training-data generation), rank, format into
  the payload text. No fine-tuned model touched yet.

  Phase 2 (infer): load the fine-tuned adapter once, then run generation
  for every built payload in a loop, logging each prediction.

Foundry Local appears to load its model on demand per request rather than
holding it resident continuously, so there's no need to explicitly stop/
start it between phases — the natural sequencing (all dedup calls finish
in phase 1 before phase 2 ever loads our own model) already avoids the two
competing for GPU memory at the same time.

Usage:
  # ad-hoc, single ticker, both phases in one go
  python predict_ticker.py --ticker MSFT

  # nightly batch, one model load for all tickers instead of N:
  python predict_ticker.py --all-tickers --date 2026-07-28
"""
import argparse
import glob
import json
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

import build_ticker_jsonl as btj

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DEFAULT_LOG_PATH = os.getenv("PREDICTIONS_LOG_PATH", "predictions_log.jsonl")


# ---------------------------------------------------------------------------
# Phase 1 — build payload(s). Triggers Foundry Local dedup calls.
# ---------------------------------------------------------------------------
def find_articles_for_ticker_date(data_dir: str, ticker: str, date: str,
                                    min_relevance: int, min_summary_words: int) -> list[dict]:
    """Scan only the given date's folder across every news-source domain
    folder — data_dir/{domain}/{date}/{id}.json — rather than indexing the
    whole tree (that full-tree index is for bulk training-data builds;
    here we only care about one ticker on one day).
    """
    pattern = os.path.join(data_dir, '*', date, '*.json')
    matching_metas = []
    for path in glob.glob(pattern):
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                meta = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        gdelt_query = meta.get('gdelt_query', '')
        resolved_ticker = btj.ALL_TICKERS.get(gdelt_query)
        if not resolved_ticker or resolved_ticker.upper() != ticker.upper():
            continue

        good, _reason = btj.get_article_quality_reason(meta, min_relevance, min_summary_words)
        if not good:
            continue

        matching_metas.append(meta)

    return matching_metas


def build_payload_for_ticker(ticker: str, date: str, data_dir: str,
                              min_relevance: int, min_summary_words: int,
                              max_candidates: int) -> dict | None:
    """Everything needed before inference: find articles, dedup (Foundry
    Local), rank, format. Returns None if there's nothing usable to predict
    from (no qualifying articles, or no price trend yet).
    """
    articles = find_articles_for_ticker_date(data_dir, ticker, date, min_relevance, min_summary_words)
    if not articles:
        return None

    price_record = btj.get_price_record(articles[0])
    if not price_record.get('price_trend'):
        for art in articles:
            pr = btj.get_price_record(art)
            if pr.get('price_trend'):
                price_record = pr
                break

    news = btj.dedupe_all_articles_for_date(date, articles, batch_size=max_candidates or 5)
    news = sorted(news, key=btj.news_signal_strength, reverse=True)[:5]

    record = {'date': date, 'ticker': ticker.upper(), 'news': news, '_price_record': price_record}
    user_content = btj.build_user_content(record)
    if user_content is None:
        return None

    return {
        'ticker': ticker.upper(),
        'date': date,
        'user_content': user_content,
        'base_price': price_record.get('base_price'),
    }


# ---------------------------------------------------------------------------
# Phase 2 — inference. Loads the fine-tuned model once, reused across every
# ticker in the batch.
# ---------------------------------------------------------------------------
def run_inference(model, tokenizer, payload: dict) -> tuple[str | None, str]:
    import torch
    from eval import extract_move_bin

    messages = [
        {"role": "system", "content": btj.SYSTEM_PROMPT},
        {"role": "user", "content": payload['user_content']},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    prediction = extract_move_bin(generated)
    return prediction, generated


def append_prediction_log(log_path: str, entry: dict) -> None:
    """Append-only — every run adds a new line rather than overwriting, so
    rerunning the same ticker/date (e.g. after retraining) doesn't lose the
    earlier prediction. reconcile_predictions.py fills in actual_move_bin/
    actual_change_1d later, once the real outcome has settled.
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_prediction(log_path: str, payload: dict, prediction: str | None, generated: str,
                    model_id: str, adapter_path: str) -> None:
    news_section = payload['user_content'].split("News:\n", 1)[-1]
    num_news_items = len([l for l in news_section.splitlines() if l.strip()])
    append_prediction_log(log_path, {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "ticker": payload['ticker'],
        "date": payload['date'],
        "predicted_move_bin": prediction,
        "raw_model_output": generated,
        "num_news_items": num_news_items,
        "base_price": payload.get('base_price'),
        "model_id": model_id,
        "adapter_path": adapter_path,
        # Filled in later by reconcile_predictions.py once tomorrow's close
        # is known.
        "actual_move_bin": None,
        "actual_change_1d": None,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default=None, help='Single ticker (omit if using --all-tickers)')
    parser.add_argument('--all-tickers', action='store_true',
                         help='Build + predict for every ticker in the watchlist, one model load total')
    parser.add_argument('--date', default=datetime.now().strftime('%Y-%m-%d'),
                         help='Date folder to read, YYYY-MM-DD (default: today)')
    parser.add_argument('--data-dir', default=btj.DEFAULT_DATA_DIR)
    parser.add_argument('--min-relevance', type=int, default=btj.DEFAULT_MIN_RELEVANCE)
    parser.add_argument('--min-summary-words', type=int, default=btj.DEFAULT_MIN_SUMMARY_WORDS)
    parser.add_argument('--max-candidates', type=int, default=0)
    parser.add_argument('--model-id', default=os.getenv("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct"))
    parser.add_argument('--adapter-path', default=os.getenv("ADAPTER_PATH", "C:\\work\\PratyushMishraGitHub\\news_analysis\\fine_tune\\phi3-fincast-qlora-final"))
    parser.add_argument('--log-path', default=DEFAULT_LOG_PATH)
    parser.add_argument('--no-log', action='store_true')
    args = parser.parse_args()

    if not args.ticker and not args.all_tickers:
        parser.error('Either --ticker or --all-tickers is required')

    tickers = sorted(set(btj.ALL_TICKERS.values())) if args.all_tickers else [args.ticker.upper()]

    # --- Phase 1: build every payload first ---
    print(f"Building payload(s) for {len(tickers)} ticker(s) on {args.date} ...")
    payloads = {}
    for t in tickers:
        payload = build_payload_for_ticker(
            t, args.date, args.data_dir, args.min_relevance, args.min_summary_words, args.max_candidates
        )
        if payload is None:
            print(f"  [SKIP] {t}: no qualifying articles or no usable price trend for {args.date}")
            continue
        payloads[t] = payload
        print(f"  [OK] {t}: payload built ({len(payload['user_content'])} chars)")

    if not payloads:
        print("Nothing to predict — no tickers had usable payloads.")
        return

    if not args.all_tickers:
        print("\n--- Payload sent to model (same shape as training data, no label) ---")
        print(payloads[tickers[0]]['user_content'])
        print("---\n")

    # --- Phase 2: load the model ONCE, run inference for every payload ---
    from eval import load_model_and_tokenizer
    print(f"Loading fine-tuned model once for {len(payloads)} prediction(s) ...")
    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, adapter_path=args.adapter_path)
    print(f"Loaded model with MODEL_ID={args.model_id}, ADAPTER_PATH={args.adapter_path}")
    

    for t, payload in payloads.items():
        prediction, generated = run_inference(model, tokenizer, payload)
        print(f"  {t}: {prediction or 'UNPARSEABLE'}  (raw: {generated!r})")
        if not args.no_log:
            log_prediction(args.log_path, payload, prediction, generated, args.model_id, args.adapter_path)

    if not args.no_log:
        print(f"\nLogged {len(payloads)} prediction(s) to {args.log_path}")


if __name__ == '__main__':
    main()