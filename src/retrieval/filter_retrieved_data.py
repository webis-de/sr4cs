"""
Filter the retrieved data to keep only the studies that contain keywords related to systematic reviews, only peer-reviewed articles and open access studies. Converts the filtered data to JSON and CSV.
"""

import json
import csv
import logging
import os

# --- Config ---
DATE = "03_07_25"

RAW_JSON_FILE = os.path.expanduser(
    f"../../data/raw/dblp_systematic_review_2000_2025_{DATE}.json"
)
FILTERED_JSON_FILE = os.path.expanduser(
    f"../../data/filtered/dblp_systematic_review_2000_2025_filtered_{DATE}.json"
)
FILTERED_CSV_FILE = os.path.expanduser(
    f"../../data/filtered/dblp_systematic_review_2000_2025_filtered_{DATE}.csv"
)
LOG_FILE = os.path.expanduser(f"../../logs/filter_retrieved_data_{DATE}.log")

KEYWORDS = [
    "a systematic literature review",
    "systematic review on",
    "systematic review of",
    "systematic literature review on",
    "systematic literature review of",
]

PEER_REVIEWED_TYPES = [
    "Journal Articles",
    "Conference and Workshop Papers",
]

ACCESS_TYPE = "open"

# ensure output directories exist
os.makedirs(os.path.dirname(FILTERED_JSON_FILE), exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    force=True,
)

# --- Load raw data ---
with open(RAW_JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
logging.info(f"Loaded {len(data)} entries from raw JSON.")

# --- Filter by keywords in title ---
filtered = [
    entry
    for entry in data
    if any(kw in entry["info"]["title"].lower() for kw in KEYWORDS)
]
logging.info(f"{len(filtered)} entries match systematic-review keywords.")

# --- Filter to peer-reviewed types ---
filtered = [
    entry for entry in filtered if entry["info"].get("type") in PEER_REVIEWED_TYPES
]
logging.info(f"{len(filtered)} entries are peer-reviewed articles.")

# --- Filter to open access ---
filtered = [entry for entry in filtered if entry["info"].get("access") == ACCESS_TYPE]
logging.info(f"{len(filtered)} entries are open access.")

# --- Filter out entries where doi is empty or whitespace ---
filtered = [
    entry
    for entry in filtered
    if entry["info"].get("doi") and entry["info"]["doi"].strip()
]
logging.info(f"{len(filtered)} entries have a valid DOI.")

# --- Save filtered JSON ---
with open(FILTERED_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=4)
logging.info(f"Filtered JSON saved to {FILTERED_JSON_FILE}.")

# --- Convert to CSV ---
seen_ids = set()
fieldnames = ["id", "title", "venue", "year", "type", "doi", "authors", "access"]

with open(FILTERED_CSV_FILE, "w", newline="", encoding="utf-8") as csv_f:
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    writer.writeheader()

    for entry in filtered:
        eid = entry.get("@id", "")
        if eid in seen_ids:
            logging.warning(f"Duplicate id {eid}, skipping.")
            continue
        seen_ids.add(eid)

        info = entry.get("info", {})
        authors = info.get("authors", {}).get("author", [])
        if not isinstance(authors, list):
            authors = [authors]
        author_names = [
            a.get("text", "") if isinstance(a, dict) else str(a) for a in authors
        ]

        writer.writerow(
            {
                "id": eid,
                "title": info.get("title", ""),
                "venue": info.get("venue", ""),
                "year": info.get("year", ""),
                "type": info.get("type", ""),
                "doi": info.get("doi", ""),
                "authors": "; ".join(author_names),
                "access": info.get("access", ""),
            }
        )

logging.info(f"Filtered CSV saved to {FILTERED_CSV_FILE}.")
logging.info("Filtering and conversion completed successfully.")
