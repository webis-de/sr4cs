"""
Fetch missing abstracts from arXiv and write them back into your CSV.
"""

# -------- config -------------------------------------------------------
SRC_FILE = "../../../data/extracted_information/refs/processed/references_arxiv.csv"
DST_FILE = (
    "../../../data/extracted_information/refs/with_abs/references_arxiv_with_abs.csv"
)
LOG_FILE = "../../../logs/arxiv_fetch.log"
# ----------------------------------------------------------------------

import re, time, xml.etree.ElementTree as ET
from pathlib import Path
import requests, pandas as pd
from tqdm import tqdm
import logging

ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)|([a-z\-]+/\d{7})", re.I)
API = "http://export.arxiv.org/api/query?search_query=id:{}&max_results=1"
HEADERS = {"User-Agent": "arXiv-abstract-fetch/0.1 (mailto:piero@example.com)"}

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def clean_arxiv(raw: str | float) -> str | None:
    """Canonical arXiv ID or None."""
    if pd.isna(raw):
        return None
    m = ID_RE.search(str(raw))
    return m.group(0).lower() if m else None


def fetch_abs(arx_id: str) -> str | None:
    """Return abstract text or None."""
    try:
        r = requests.get(API.format(arx_id), headers=HEADERS, timeout=10)
        if not r.ok:
            return None
        root = ET.fromstring(r.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        summary = root.find(".//a:entry/a:summary", ns)
        return " ".join(summary.text.split()) if summary is not None else None
    except Exception:
        return None


def main():
    df = pd.read_csv(SRC_FILE)

    if "abstract" not in df.columns:
        df["abstract"] = pd.NA

    df["arxiv_id"] = df["arxiv"].apply(clean_arxiv)

    need_mask = df["abstract"].isna() & df["arxiv_id"].notna()
    rows_to_fetch = df[need_mask]

    for idx, row in tqdm(
        rows_to_fetch.iterrows(), total=len(rows_to_fetch), desc="arXiv lookups"
    ):
        arx_id = row["arxiv_id"]
        abs_txt = fetch_abs(arx_id)
        df.at[idx, "abstract"] = abs_txt
        if abs_txt:
            logging.info("FOUND %s", arx_id)
        else:
            logging.warning("NOT_FOUND %s", arx_id)
        time.sleep(0.2)  # polite: ~300 req/h

    df.drop(columns="arxiv_id").to_csv(DST_FILE, index=False)

    missing = df["abstract"].isna().sum()
    logging.info("Complete. %d abstracts still missing", missing)


if __name__ == "__main__":
    main()
