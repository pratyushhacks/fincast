r"""
scraper.py — fetches articles and saves raw HTML, text and metadata JSON to disk.

Called by fetch_gdelt_news.py (via save_article) and optionally as standalone
RSS scraper (python scraper.py --feeds feeds.json).
"""
import os
import json
import hashlib
import argparse
from datetime import datetime

import feedparser
import requests
from bs4 import BeautifulSoup
from readability import Document
from dateutil import parser as dateparser
import tldextract
from langdetect import detect

DATA_DIR        = os.path.join('data', 'raw')
SCRAPER_VERSION = "0.1"
REQUEST_HEADERS = {"User-Agent": "news-scraper/0.1"}


def is_english(text):
    try:
        return detect(text[:500]) == 'en'
    except Exception:
        return True


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
    return soup.get_text(separator=' ', strip=True), content_html


def save_article(url, feed_entry=None, gdelt_query=None):
    try:
        r = requests.get(url, timeout=15, headers=REQUEST_HEADERS)
        r.raise_for_status()
    except Exception as e:
        print(f"  [SKIP] Failed to fetch {url}: {e}")
        return None

    html = r.text
    text, _ = extract_text(html)

    if len(text.split()) < 50:
        print(f"  [SKIP] Too short ({len(text.split())} words): {url}")
        return None

    if not is_english(text):
        print(f"  [SKIP] Non-English: {url}")
        return None

    publish_date = None
    if feed_entry:
        dt = feed_entry.get('published') or feed_entry.get('updated')
        if dt:
            try:
                publish_date = dateparser.parse(dt).isoformat()
            except Exception:
                publish_date = None

    article_id  = slug_id(url, publish_date)
    domain      = domain_from_url(url)
    scrape_date = datetime.utcnow().isoformat()
    date_folder = datetime.utcnow().strftime('%Y-%m-%d')

    out_dir   = os.path.join(DATA_DIR, domain, date_folder)
    os.makedirs(out_dir, exist_ok=True)

    html_path = os.path.join(out_dir, f"{article_id}.html")
    json_path = os.path.join(out_dir, f"{article_id}.json")
    text_path = os.path.join(out_dir, f"{article_id}.txt")

    if os.path.exists(json_path):
        return None  # already scraped

    if feed_entry and feed_entry.get('title'):
        title = feed_entry['title']
    else:
        soup_title = BeautifulSoup(html, 'lxml').find('title')
        title = soup_title.string.strip() if soup_title else None

    meta = {
        'id':              article_id,
        'url':             url,
        'site':            domain,
        'title':           title,
        'publish_date':    publish_date,
        'scrape_date':     scrape_date,
        'html_path':       html_path,
        'text_path':       text_path,
        'word_count':      len(text.split()),
        'paywalled':       False,
        'dedup_hash':      hashlib.sha1(text.encode('utf-8')).hexdigest(),
        'scraper_version': SCRAPER_VERSION,
        'gdelt_query':     gdelt_query,
    }

    with open(html_path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    with open(text_path, 'w', encoding='utf-8') as fh:
        fh.write(text)

    print(f"  [OK] {url} -> {json_path}")
    return meta


def run(feeds_file):
    with open(feeds_file, 'r', encoding='utf-8') as fh:
        raw = json.load(fh)
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
    parser.add_argument('--feeds', default='feeds.json')
    args = parser.parse_args()
    run(args.feeds)
