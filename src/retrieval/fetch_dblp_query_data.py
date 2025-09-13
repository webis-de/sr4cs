"""
Fetch all DBLP records for a search term, working around DBLP's 10 000-hit limit by slicing the query by publication year.
"""

import os
import json
import time
import random
from datetime import datetime
from collections import OrderedDict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# CONFIG ─ change these and go
# ---------------------------------------------------------------------------
DATE = "03_07_25"  # date for output files
QUERY = "systematic review"  # search term
YEAR_FROM = 2000  # first year (inclusive)
YEAR_TO = datetime.now().year  # last year (inclusive)
DELAY = 2  # seconds between API calls
DATA_DIR = os.path.expanduser("../..//data/raw")
USER_AGENT = "MyDBLPClient/1.1 (mailto:pierre.achkar@uni-leipzig.de)"
RESULTS_PER_PAGE = 1000  # DBLP hard-coded max; leave as-is
# ---------------------------------------------------------------------------

BASE_URL = "https://dblp.org/search/publ/api"

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def make_session() -> requests.Session:
    """Return a requests session with retry and custom UA."""
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def safe_get(session: requests.Session, params: dict):
    """GET with 429 + 5xx handling and exponential back-off."""
    backoff = 1
    while True:
        resp = session.get(BASE_URL, params=params, timeout=30)
        code = resp.status_code

        if code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            print(f"429: sleeping {retry_after}s…")
            time.sleep(retry_after)
            continue

        if 500 <= code < 600:
            print(f"{code}: backing off {backoff}s…")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue

        return resp


# ---------------------------------------------------------------------------
# DBLP helpers
# ---------------------------------------------------------------------------


def fetch_slice(session: requests.Session, query: str, delay: int):
    """Fetch all pages for one query slice (≤10 000 hits)."""
    params = {"q": query, "format": "json", "h": RESULTS_PER_PAGE, "f": 0}
    hits = []

    # first page gives total
    resp = safe_get(session, params)
    if resp.status_code != 200:
        raise RuntimeError(f"Initial request failed: {resp.status_code}")

    data = resp.json()
    total = int(data["result"]["hits"]["@total"])
    hits.extend(data["result"]["hits"].get("hit", []))

    for start in range(RESULTS_PER_PAGE, total, RESULTS_PER_PAGE):
        params["f"] = start
        to_pos = min(start + RESULTS_PER_PAGE, total)
        print(f" {query[:40]:<40} {start:>6}–{to_pos:<6}")
        time.sleep(delay + random.random() * 0.5)
        resp = safe_get(session, params)
        if resp.status_code != 200:
            print(f"  warning: {resp.status_code} at batch {start}. aborting slice.")
            break
        hits.extend(resp.json()["result"]["hits"].get("hit", []))

    return hits


def deduplicate(hits):
    """Remove duplicates by DBLP key, preserving first occurrence."""
    unique = OrderedDict()
    for h in hits:
        unique[h["info"]["key"]] = h
    return list(unique.values())


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------


def fetch_all() -> list:
    session = make_session()
    all_hits = []

    for year in range(YEAR_FROM, YEAR_TO + 1):
        slice_query = f"{QUERY} year:{year}"
        print(f"=== {slice_query} ===")
        slice_hits = fetch_slice(session, slice_query, DELAY)
        print(f" → {len(slice_hits)} hits in {year}")
        all_hits.extend(slice_hits)

    return deduplicate(all_hits)


# ---------------------------------------------------------------------------
# Utility output
# ---------------------------------------------------------------------------


def pretty_sample(hits, k: int = 5):
    for i, hit in enumerate(hits[:k], 1):
        info = hit["info"]
        title = info.get("title", "N/A")
        year = info.get("year", "N/A")
        url = info.get("url", "N/A")
        authors = info.get("authors", {}).get("author", [])
        if isinstance(authors, dict):
            authors = [authors]
        names = ", ".join(a["text"] for a in authors) if authors else "N/A"
        print(f"{i}. {title}\n   {names} — {year}\n   {url}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    hits = fetch_all()
    print(f"\nFetched {len(hits)} unique records.\n")

    out_path = os.path.join(
        DATA_DIR, f"dblp_{QUERY.replace(' ', '_')}_{YEAR_FROM}_{YEAR_TO}_{DATE}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(hits, f, indent=2)
    print(f"Saved to: {out_path}\n")

    print("Sample:")
    pretty_sample(hits)


if __name__ == "__main__":
    main()
