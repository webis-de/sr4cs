"""Script to enrich a CSV of papers with PDF links using DOIs and Unpaywall/Crossref APIs."""

import re
import time
import requests
import pandas as pd
import logging
from tqdm import tqdm

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CSV_INPUT = "../../data/filtered/dblp_systematic_review_2000_2025_filtered_03_07_25.csv"
CSV_OUTPUT = "../../data/full_paper_intermediate/dblp_systematic_review_2000_2025_filtered_link_03_07_25.csv"
EMAIL = "pieer.achkar@imw.fraunhofer.de"
LOG_FILE = "../../logs/get_pdf_link_03_07_25.log"

# ── LOGGING ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
    force=True,
)
logging.info("=== STARTING link extraction run ===")

# ── HELPERS ────────────────────────────────────────────────────────────────────
DOI_RE = re.compile(r"(10\.\d{4,9}/\S+)", re.I)
HEADERS = {"Accept": "application/pdf"}


def extract_doi(doi_url):
    if not isinstance(doi_url, str):
        logging.debug(f"extract_doi: input not a string: {doi_url!r}")
        return None

    m = DOI_RE.search(doi_url)
    if m:
        doi = m.group(1).lower()
        logging.debug(f"extract_doi: extracted '{doi}' from '{doi_url}'")
        return doi
    else:
        logging.warning(f"extract_doi: failed to parse DOI from '{doi_url}'")
        return None


def oa_pdf(doi):
    url = f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}"
    logging.debug(f"{doi} – querying Unpaywall: {url}")
    try:
        r = requests.get(url, timeout=15)
        if r.ok:
            loc = r.json().get("best_oa_location") or {}
            pdf = loc.get("url_for_pdf")
            if pdf:
                logging.info(f"{doi} – Unpaywall returned PDF link")
                return pdf
            else:
                logging.debug(f"{doi} – Unpaywall has no OA PDF")
        else:
            logging.warning(f"{doi} – Unpaywall HTTP {r.status_code}")
    except Exception as e:
        logging.error(f"{doi} – Unpaywall lookup error: {e}")
    return None


def crossref_pdf(doi):
    url = f"https://doi.org/{doi}"
    logging.debug(f"{doi} – querying Crossref via content-negotiation: {url}")
    try:
        r = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=15)
        ctype = r.headers.get("content-type", "")
        if r.ok and ctype.startswith("application/pdf"):
            logging.info(f"{doi} – Crossref returned PDF link")
            return r.url
        else:
            logging.debug(
                f"{doi} – Crossref no PDF (status {r.status_code}, ct={ctype})"
            )
    except Exception as e:
        logging.error(f"{doi} – Crossref lookup error: {e}")
    return None


def find_pdf_link(doi):
    if not doi:
        logging.error("find_pdf_link: missing DOI")
        return None

    # first try OA
    pdf = oa_pdf(doi)
    if pdf:
        return pdf

    # fallback
    return crossref_pdf(doi)


# ── ENRICH CSV WITH PDF LINKS ───────────────────────────────────────────────────
df = pd.read_csv(CSV_INPUT)
logging.info(f"Loaded {len(df)} rows from {CSV_INPUT}")

# extract DOIs
df["doi"] = df["doi"].map(extract_doi)
count_no_doi = df["doi"].isna().sum()
logging.info(f"{count_no_doi} rows without a valid DOI")

# find PDF links with progress bar
tqdm.pandas(desc="Finding PDF links")
df["pdf_link"] = df["doi"].progress_map(find_pdf_link)

found = df["pdf_link"].notna().sum()
missing = len(df) - found
logging.info(f"Found {found} PDF links; {missing} missing")

# save
df.to_csv(CSV_OUTPUT, index=False)
logging.info(f"Wrote enriched CSV to {CSV_OUTPUT}")

logging.info("=== link extraction run complete ===")
print("done.")
