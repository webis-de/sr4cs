"""
Fetch missing abstracts via DOI using Europe PMC API:
Then write them back into the CSV.
"""

# ---------- CONFIG -----------------------------------------------------
SRC_FILE = (
    "../../../data/extracted_information/refs/still_missing/references_split_no_abs.csv"
)
DST_FILE = "../../../data/extracted_information/refs/still_missing/references_split_no_abs_europepmc.csv"
LOG_FILE = "../../../logs/europepmc_fetch.log"
# ----------------------------------------------------------------------

import re, time, requests, logging, urllib.parse as ul, pandas as pd
from tqdm import tqdm

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s;]+", re.I)
EPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
HEADERS = {
    "User-Agent": "europepmc-abstract-fetch/1.0 (mailto:pieer.achkar@imw.fraunhofer.de)"
}

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ----------------------------------------------------------------------
def clean_doi(raw):
    """Extract and clean DOI from raw text"""
    if pd.isna(raw):
        return None
    m = DOI_RE.search(str(raw))
    if not m:
        return None
    doi = m.group(0).rstrip(").,;:")
    if len(doi.split("/", 1)[1]) < 8:  # junk like "10.1007/978-"
        return None
    return doi.lower()


def strip_tags(text):
    """Remove HTML/XML tags from text"""
    return re.sub(r"<[^>]+>", "", text, flags=re.S)


def fetch_europepmc_abs(doi):
    """
    Fetch abstract from Europe PMC using DOI search
    Returns tuple: (abstract_text, pmid) or (None, None)
    """
    try:
        # Europe PMC search by DOI
        params = {
            "query": f'DOI:"{doi}"',
            "format": "json",
            "resultType": "core",
            "pageSize": 1,
            "cursorMark": "*",
        }

        r = requests.get(EPMC_API, params=params, headers=HEADERS, timeout=15)
        if not r.ok:
            return None, None

        data = r.json()
        results = data.get("resultList", {}).get("result", [])

        if not results:
            return None, None

        # Get the first result
        result = results[0]

        # Extract abstract
        abstract = result.get("abstractText", "")
        if abstract:
            abstract = strip_tags(abstract).strip()

        # Extract PMID if available
        pmid = result.get("pmid", "")

        # If no abstract in main result, try to get full text abstract
        if not abstract:
            # Sometimes abstracts are in authorString or other fields
            # Try alternative fields that might contain abstract info
            for field in ["title", "authorString"]:
                if result.get(field):
                    # This is not ideal, but Europe PMC sometimes has limited abstract data
                    pass

        return abstract if abstract else None, pmid if pmid else None

    except Exception as e:
        logging.error(f"Error fetching from Europe PMC for DOI {doi}: {str(e)}")
        return None, None


def fetch_europepmc_fulltext_abs(pmid):
    """
    Attempt to get full abstract using PMID if available
    """
    if not pmid:
        return None

    try:
        # Try to get full text article details
        fulltext_url = (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/article/MED/{pmid}"
        )
        params = {"format": "json"}

        r = requests.get(fulltext_url, params=params, headers=HEADERS, timeout=15)
        if not r.ok:
            return None

        data = r.json()
        result = data.get("result", {})

        abstract = result.get("abstractText", "")
        if abstract:
            return strip_tags(abstract).strip()

        return None

    except Exception as e:
        logging.error(
            f"Error fetching fulltext from Europe PMC for PMID {pmid}: {str(e)}"
        )
        return None


# ----------------------------------------------------------------------
def main():
    """Main execution function"""
    df = pd.read_csv(SRC_FILE)

    # Ensure abstract column exists
    if "abstract" not in df.columns:
        df["abstract"] = pd.NA

    # Add PMID column if it doesn't exist (for tracking)
    if "pmid" not in df.columns:
        df["pmid"] = pd.NA

    # Clean DOIs
    df["doi_id"] = df["doi"].apply(clean_doi)

    # Find rows that need abstracts and have valid DOIs
    need = df["abstract"].isna() & df["doi_id"].notna()
    rows = df[need]

    logging.info(f"Starting Europe PMC abstract fetch for {len(rows)} DOIs")

    found_count = 0
    for idx, row in tqdm(
        rows.iterrows(), total=len(rows), desc="Europe PMC DOI lookups"
    ):
        doi = row["doi_id"]

        # First try: search by DOI
        abs_txt, pmid = fetch_europepmc_abs(doi)

        # Second try: if we got a PMID but no abstract, try fulltext endpoint
        if not abs_txt and pmid:
            abs_txt = fetch_europepmc_fulltext_abs(pmid)

        # Update dataframe
        if abs_txt:
            df.at[idx, "abstract"] = abs_txt
            if pmid:
                df.at[idx, "pmid"] = pmid
            found_count += 1
            logging.info(f"FOUND_EUROPEPMC {doi} (PMID: {pmid or 'N/A'})")
        else:
            logging.warning(f"NO_ABSTRACT {doi}")

        # Be polite to the API
        time.sleep(0.2)  # Europe PMC recommends 200ms between requests

    # Save results
    df.drop(columns=["doi_id"], errors="ignore").to_csv(DST_FILE, index=False)

    still_missing = df["abstract"].isna().sum()
    logging.info(
        f"Finished — Found {found_count} new abstracts, {still_missing} abstracts still missing"
    )
    print(f"Europe PMC fetch complete:")
    print(f"  - Found: {found_count} new abstracts")
    print(f"  - Still missing: {still_missing} abstracts")
    print(f"  - Results saved to: {DST_FILE}")


if __name__ == "__main__":
    main()
