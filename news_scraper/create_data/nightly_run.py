r"""
nightly_run.py — run this every night via Task Scheduler / cron

1. Fetches today's articles from GDELT
2. Scores sentiment via Foundry Local
3. Updates stock prices for yesterday's articles (now settled)
4. Reconciles yesterday's predictions against those now-settled prices
   (fills in actual_move_bin / actual_change_1d in predictions_log.jsonl)
5. Builds + logs tonight's predictions for every supported ticker

Schedule: daily at 11pm
  Windows Task Scheduler: python nightly_run.py
  cron: 0 23 * * * /path/to/.venv/Scripts/python nightly_run.py

Usage:
  python nightly_run.py                # default: look back 1 day
  python nightly_run.py --days 7        # e.g. after a missed night or two
"""
import argparse
import subprocess
import sys
from datetime import datetime

PYTHON = sys.executable
TODAY = datetime.now().strftime('%Y-%m-%d')


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Nightly FinCast data + prediction pipeline")
    parser.add_argument('--days', type=int, default=1,
                         help='How many days back fetch_gdelt_news.py should look (default: 1)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    run([PYTHON, "./create_data/fetch_gdelt_news.py", "--days", str(args.days)])
    run([PYTHON, "./create_data/label_sentiment.py", "--data-dir", "./data/raw", "--workers", "1"])
    run([PYTHON, "./create_data/fetch_prices.py", "--data-dir", "./data/raw"])

    # Reconcile predictions made on previous nights — fetch_prices.py just
    # recomputed change_1d/bucket_1d for now-settled dates, so this is the
    # right moment to check which pending predictions can now be scored.
    # Must run BEFORE tonight's new predictions below, not after.
    run([PYTHON, "./generate_dataset/reconcile_predictions.py"])

    if args.days > 1:
        from datetime import timedelta
        anchor = datetime.now()
        for i in range(args.days):
            day = (anchor - timedelta(days=i)).strftime('%Y-%m-%d')
            print(f"\n=== Predicting for date {day} ===")
            run([PYTHON, "./generate_dataset/predict_ticker.py", "--all-tickers", "--date", day])
    else:
        # Build + log tonight's prediction for every supported ticker. This
        # loads the fine-tuned model ONCE for all tickers (not once per
        # ticker) — see predict_ticker.py's module docstring for why the
        # build/infer split exists and why no Foundry Local start/stop
        # juggling is needed around it.
        run([PYTHON, "./generate_dataset/predict_ticker.py", "--all-tickers", "--date", TODAY])