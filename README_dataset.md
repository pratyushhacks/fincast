Dataset collection scaffold

Run the RSS-driven scraper to collect raw HTML and per-article metadata into data/raw/{site}/{YYYY-MM-DD}/.

Setup

python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

Run

python scraper.py --feeds feeds.json

Collected files:
- data/raw/{site}/{YYYY-MM-DD}/{id}.html
- data/raw/{site}/{YYYY-MM-DD}/{id}.json
- data/raw/{site}/{YYYY-MM-DD}/{id}.txt
