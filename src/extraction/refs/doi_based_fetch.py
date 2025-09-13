"""
Fetch missing abstracts based on DOIs using CrossRef and OpenAlex APIs.
"""

# ---------- CONFIG -----------------------------------------------------
SRC_FILE = "../../../data/extracted_information/refs/with_abs/still_missing/references_split_grobid_filled.csv"
DST_FILE = "../../../data/extracted_information/refs/with_abs/still_missing/references_split_grobid_filled_doi.csv"
LOG_FILE = "../../../logs/doi_fetch_2.log"
# ----------------------------------------------------------------------

import re, time, requests, logging, urllib.parse as ul, pandas as pd
from tqdm import tqdm

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s;]+", re.I)
CR_API = "https://api.crossref.org/works/{}"
OA_API = "https://api.openalex.org/works/doi:{}"
HEADERS = {
    "User-Agent": "doi-abstract-fetch/0.2 (mailto:pieer.achkar@imw.fraunhofer.de)"
}

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ----------------------------------------------------------------------
def clean_doi(raw):
    if pd.isna(raw):
        return None
    m = DOI_RE.search(str(raw))
    if not m:
        return None
    doi = m.group(0).rstrip(").,;:")
    if len(doi.split("/", 1)[1]) < 8:  # junk like “10.1007/978-”
        return None
    return doi.lower()


def strip_tags(text):
    return re.sub(r"<[^>]+>", "", text, flags=re.S)


def fetch_crossref_abs(doi):
    try:
        url = CR_API.format(ul.quote(doi, safe=""))
        r = requests.get(url, headers=HEADERS, timeout=15)
        if not r.ok:
            return None
        abs_raw = r.json()["message"].get("abstract")
        return strip_tags(abs_raw).strip() if abs_raw else None
    except Exception:
        return None


def fetch_openalex_abs(doi):
    try:
        url = OA_API.format(ul.quote(doi, safe=""))
        r = requests.get(url, headers=HEADERS, timeout=15)
        if not r.ok:
            return None
        idx = r.json().get("abstract_inverted_index")
        if not idx:
            return None
        # rebuild text from the inverted index
        words_sorted = sorted(idx.items(), key=lambda kv: kv[1][0])
        return " ".join(w for w, _ in words_sorted)
    except Exception:
        return None


# ----------------------------------------------------------------------
def main():
    df = pd.read_csv(SRC_FILE)
    if "abstract" not in df.columns:
        df["abstract"] = pd.NA
    df["doi_id"] = df["doi"].apply(clean_doi)

    need = df["abstract"].isna() & df["doi_id"].notna()
    rows = df[need]

    for idx, row in tqdm(rows.iterrows(), total=len(rows), desc="DOI lookups"):
        doi = row["doi_id"]
        abs_txt = fetch_crossref_abs(doi)
        origin = "CROSSREF"

        if abs_txt is None:
            abs_txt = fetch_openalex_abs(doi)
            origin = "OPENALEX" if abs_txt else None

        df.at[idx, "abstract"] = abs_txt
        if abs_txt:
            logging.info("FOUND_%s %s", origin, doi)
        else:
            logging.warning("NO_ABSTRACT %s", doi)

        time.sleep(0.1)  # polite

    df.drop(columns="doi_id").to_csv(DST_FILE, index=False)
    logging.info("Finished — %d abstracts still missing", df["abstract"].isna().sum())


if __name__ == "__main__":
    main()
