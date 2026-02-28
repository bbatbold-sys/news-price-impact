"""
GDELT Yahoo Finance News Scraper
Fetches 2 years of news articles from finance.yahoo.com via GDELT Doc API.
Date range: 2024-02-28 to 2026-02-28
"""

import requests
import json
import csv
import time
import os
from datetime import datetime, timedelta

# Config
DOMAIN = "finance.yahoo.com"
START_DATE = datetime(2024, 2, 28)
END_DATE = datetime(2026, 2, 28)
CHUNK_DAYS = 7          # query window size (weekly chunks to stay under API limits)
MAX_RECORDS = 250       # GDELT API max per request
OUTPUT_CSV = "gdelt_yahoo_finance.csv"
OUTPUT_JSON = "gdelt_yahoo_finance.json"
REQUEST_DELAY = 1.5     # seconds between requests (be polite to API)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

FIELDNAMES = [
    "url", "url_mobile", "title", "seendate", "socialimage",
    "domain", "language", "sourcecountry"
]


def format_gdelt_datetime(dt: datetime) -> str:
    """GDELT expects YYYYMMDDHHmmSS format."""
    return dt.strftime("%Y%m%d%H%M%S")


def fetch_chunk(start: datetime, end: datetime) -> list[dict]:
    """Fetch articles for a single time window."""
    params = {
        "query": f"domain:{DOMAIN}",
        "mode": "artlist",
        "maxrecords": MAX_RECORDS,
        "startdatetime": format_gdelt_datetime(start),
        "enddatetime": format_gdelt_datetime(end),
        "format": "json",
        "sort": "DateDesc",
    }
    try:
        resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        return articles
    except requests.exceptions.HTTPError as e:
        print(f"  HTTP error: {e}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"  Request error: {e}")
        return []
    except json.JSONDecodeError:
        print(f"  JSON decode error for chunk {start} - {end}")
        return []


def main():
    all_articles = []
    seen_urls = set()

    # Generate time chunks
    chunks = []
    current = START_DATE
    while current < END_DATE:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS), END_DATE)
        chunks.append((current, chunk_end))
        current = chunk_end

    total_chunks = len(chunks)
    print(f"Fetching {DOMAIN} news from GDELT")
    print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Total chunks: {total_chunks} ({CHUNK_DAYS}-day windows)\n")

    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        print(f"[{i}/{total_chunks}] {chunk_start.date()} to {chunk_end.date()}", end="  ")
        articles = fetch_chunk(chunk_start, chunk_end)

        new_count = 0
        for article in articles:
            url = article.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_articles.append(article)
                new_count += 1

        print(f"+{new_count} articles (total: {len(all_articles)})")
        time.sleep(REQUEST_DELAY)

    print(f"\nDone. Total unique articles: {len(all_articles)}")

    # Save CSV
    print(f"Saving CSV to {OUTPUT_CSV}")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_articles)

    # Save JSON
    print(f"Saving JSON to {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)

    print("Complete.")


if __name__ == "__main__":
    main()
