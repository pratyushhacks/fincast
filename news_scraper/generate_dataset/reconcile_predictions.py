"""
reconcile_predictions.py — fills in actual outcomes for previously-logged
predictions, once fetch_prices.py has recomputed change_1d/bucket_1d for
that date (i.e. once the next trading day's close is known).

Run this AFTER fetch_prices.py each night, BEFORE generating tonight's new
predictions — fetch_prices.py just updated yesterday's (now-settled)
articles, so this is exactly the right moment to check which pending
predictions can now be scored.

Reuses build_ticker_jsonl.py's get_price_record + compute_move_bin — the
SAME z-score bucketing your training labels use — so "actual" here means
exactly the same thing "move_bin" means everywhere else in this project,
not a separately-invented definition of correct/incorrect.
"""
import glob
import json
import os

import build_ticker_jsonl as btj

LOG_PATH = os.getenv("PREDICTIONS_LOG_PATH", "predictions_log.jsonl")
DATA_DIR = btj.DEFAULT_DATA_DIR


def find_settled_outcome(ticker: str, date: str, data_dir: str):
    """Re-scan that date's raw articles for this ticker and pull whatever
    price_changes fetch_prices.py has computed by now. Returns
    (actual_move_bin, actual_change_1d), or (None, None) if still
    unsettled — e.g. predicted on a Friday and Monday's close isn't in yet,
    or fetch_prices.py hasn't run for this date again since the prediction
    was logged.
    """
    pattern = os.path.join(data_dir, '*', date, '*.json')
    for path in glob.glob(pattern):
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                meta = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        gdelt_query = meta.get('gdelt_query', '')
        resolved = btj.ALL_TICKERS.get(gdelt_query)
        if not resolved or resolved.upper() != ticker.upper():
            continue

        price_record = btj.get_price_record(meta)
        if price_record.get('change_1d') is not None:
            move_bin = btj.compute_move_bin(price_record)
            return move_bin, price_record['change_1d']

    return None, None


def main():
    if not os.path.exists(LOG_PATH):
        print(f"No prediction log found at {LOG_PATH} — nothing to reconcile.")
        return

    entries = []
    with open(LOG_PATH, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    updated = 0
    still_open = 0
    for entry in entries:
        if entry.get('actual_move_bin') is not None:
            continue  # already reconciled in an earlier run

        actual_move_bin, actual_change_1d = find_settled_outcome(
            entry['ticker'], entry['date'], DATA_DIR
        )
        if actual_move_bin is not None:
            entry['actual_move_bin'] = actual_move_bin
            entry['actual_change_1d'] = actual_change_1d
            entry['correct'] = (actual_move_bin == entry.get('predicted_move_bin'))
            updated += 1
        else:
            still_open += 1

    with open(LOG_PATH, 'w', encoding='utf-8') as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Reconciled {updated} prediction(s) this run. {still_open} still awaiting settlement.")

    reconciled = [e for e in entries if e.get('actual_move_bin') is not None]
    if reconciled:
        correct = sum(1 for e in reconciled if e.get('correct'))
        print(f"Running forward-test accuracy: {correct}/{len(reconciled)} = {correct/len(reconciled)*100:.1f}%")


if __name__ == '__main__':
    main()