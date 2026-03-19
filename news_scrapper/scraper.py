r"""
Simple RSS-driven scraper that saves raw HTML and per-article metadata JSON to disk.
Run: python scraper.py
"""
import os
import json
import hashlib
import argparse
from datetime import datetime
from urllib.parse import urlparse

import feedparser
import requests
from bs4 import BeautifulSoup
from readability import Document
from dateutil import parser as dateparser
import tldextract
import yaml

from langdetect import detect

CONFIG_PATH = "config.yaml"
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as fh:
        _CFG = yaml.safe_load(fh) or {}
        COMPANIES = _CFG.get('companies', [])
        INDUSTRIES = _CFG.get('industries', [])
except Exception:
    COMPANIES = []
    INDUSTRIES = []

DATA_DIR = os.path.join('data', 'raw')
SCRAPER_VERSION = "0.1"
REQUEST_HEADERS = {"User-Agent": "news-scraper/0.1"}



def is_english(text):
    try:
        return detect(text[:500]) == 'en'
    except Exception:
        return True  # if detection fails, don't skip



def slug_id(url, publish_date=None):
    key = url + (publish_date or "")
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def domain_from_url(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


def extract_text(html):
    doc = Document(html)
    content_html = doc.summary()
    soup = BeautifulSoup(content_html, "lxml")
    for s in soup(['script', 'style', 'aside', 'noscript']):
        s.decompose()
    text = soup.get_text(separator=' ', strip=True)
    return text, content_html


def save_article(url, feed_entry=None, gdelt_query=None, gdelt_category=None):
    try:
        r = requests.get(url, timeout=15, headers=REQUEST_HEADERS)
        r.raise_for_status()
    except Exception as e:
        print(f"  [SKIP] Failed to fetch {url}: {e}")
        return None

    html = r.text
    text, content_html = extract_text(html)

    # Skip near-empty pages (paywalls, bot checks, etc.)
    if len(text.split()) < 50:
        print(f"  [SKIP] Too short ({len(text.split())} words): {url}")
        return None

    # skip non-English content
    if not is_english(text):
        print(f"  [SKIP] Non-English content: {url}")
        return None
    
    publish_date = None
    if feed_entry:
        dt = feed_entry.get('published') or feed_entry.get('updated')
        if dt:
            try:
                publish_date = dateparser.parse(dt).isoformat()
            except Exception:
                publish_date = None

    article_id = slug_id(url, publish_date)
    domain = domain_from_url(url)
    scrape_date = datetime.utcnow().isoformat()
    date_folder = datetime.utcnow().strftime('%Y-%m-%d')

    out_dir = os.path.join(DATA_DIR, domain, date_folder)
    os.makedirs(out_dir, exist_ok=True)

    html_path = os.path.join(out_dir, f"{article_id}.html")
    json_path = os.path.join(out_dir, f"{article_id}.json")
    text_path = os.path.join(out_dir, f"{article_id}.txt")

    # Skip if already scraped
    if os.path.exists(json_path):
        print(f"  [SKIP] Already exists: {json_path}")
        return None

    # Derive title
    if feed_entry and feed_entry.get('title'):
        title = feed_entry['title']
    else:
        soup_title = BeautifulSoup(html, 'lxml').find('title')
        title = soup_title.string.strip() if soup_title else None

    # Match companies and industries from config
    hay = ' '.join(filter(None, [title, text, domain])).lower()

    matched_companies = []
    for c in COMPANIES:
        name = c.get('name', '')
        domains = c.get('domains', []) or []
        keywords = c.get('keywords', []) or []
        if domain in domains or (name and name.lower() in hay):
            matched_companies.append(name)
            continue
        if any(kw and kw.lower() in hay for kw in keywords):
            matched_companies.append(name)

    matched_industries = []
    for ind in INDUSTRIES:
        iname = ind.get('name', '')
        keywords = ind.get('keywords', []) or []
        if any(kw and kw.lower() in hay for kw in keywords):
            matched_industries.append(iname)

    matched_companies = list(dict.fromkeys(matched_companies))
    matched_industries = list(dict.fromkeys(matched_industries))

    meta = {
        'id': article_id,
        'url': url,
        'site': domain,
        'title': title,
        'publish_date': publish_date,
        'scrape_date': scrape_date,
        'language': None,
        'html_path': html_path,
        'text_path': text_path,
        'word_count': len(text.split()),
        'reading_time_minutes': max(1, int(len(text.split()) / 200)),
        'paywalled': False,
        'dedup_hash': hashlib.sha1(text.encode('utf-8')).hexdigest(),
        'scraper_version': SCRAPER_VERSION,
        'matched_companies': matched_companies,
        'matched_industries': matched_industries,
        'gdelt_query':    gdelt_query,     # e.g. "NVIDIA"
        'gdelt_category': gdelt_category,  # e.g. "company"
    }

    with open(html_path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    with open(text_path, 'w', encoding='utf-8') as fh:
        fh.write(text)

    print(f"  [OK] Saved {url} -> {json_path}")
    return meta


def run(feeds_file):
    with open(feeds_file, 'r', encoding='utf-8') as fh:
        raw = json.load(fh)

    # Support both plain array and {"feeds": [...]} formats
    feeds = raw if isinstance(raw, list) else raw.get('feeds', [])

    for feed_url in feeds:
        print(f"\nParsing feed: {feed_url}")
        d = feedparser.parse(feed_url)
        for entry in d.entries:
            link = entry.get('link')
            if link:
                save_article(link, feed_entry=entry)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feeds', default='feeds.json', help='Path to feeds JSON')
    args = parser.parse_args()
    run(args.feeds)