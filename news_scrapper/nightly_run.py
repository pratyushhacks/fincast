r"""
nightly_run.py — run this every night via Task Scheduler / cron

1. Fetches today's articles from GDELT
2. Updates stock prices for yesterday's articles (now settled)

Schedule: daily at 11pm
  Windows Task Scheduler: python nightly_run.py
  cron: 0 23 * * * /path/to/.venv/Scripts/python nightly_run.py
"""
import subprocess
import sys
import time

PYTHON = sys.executable

def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
  # wrapper = subprocess.Popen(
  #   [PYTHON, "..\\llm_wrapper_project\\app.py"],
  #   cwd="..\\llm_wrapper_project"
  # )
  # time.sleep(5)  # wait for it to start

  run([PYTHON, "fetch_gdelt_news.py",  "--days", "1", "--workers", "5"])
  run([PYTHON, "label_sentiment.py",   "--data-dir", "data/raw", "--workers", "1"])
  run([PYTHON, "fetch_prices.py",      "--data-dir", "data/raw"])