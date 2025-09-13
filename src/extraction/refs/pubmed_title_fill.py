"""
Fill missing abstracts with title-only look-ups against PubMed (NCBI
E-utilities).
"""

# ───────────── CONFIG ────────────────────────────────────────────────
SRC_FILE = "../../../data/extracted_information/refs/with_abs/still_missing/references_split_no_abs.csv"
DST_FILE = "../../../data/extracted_information/refs/with_abs/still_missing/references_split_pubmed_filled.csv"
LOG_FILE = "../../../logs/pubmed_title_fetch.log"

PAUSE_SEC_NO_KEY = 0.34  # 3 requests
PAUSE_SEC_KEYED = 0.11  # 10 req / s when API key set
CKPT_EVERY = 200
FUZZ_OK = 92  # Jaro–Winkler

EMAIL = "pieer.achkar@uni-leipzig.de"
API_KEY = "..."
# ──────────────────────────────────────────────────────────────────────

import ast, logging, os, re, textwrap, time, urllib.parse as ul
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

try:
    from rapidfuzz.distance import JaroWinkler

    _sim = lambda a, b: JaroWinkler.normalized_similarity(a, b) * 100
except ImportError:  # crude fallback
    _sim = lambda a, b: 100.0 if a == b else 0.0

API_KEY = API_KEY or os.getenv("NCBI_API_KEY")
PAUSE_SEC = PAUSE_SEC_KEYED if API_KEY else PAUSE_SEC_NO_KEY

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("pubmed")

# ── helpers ───────────────────────────────────────────────────────────


def norm(txt: str) -> str:
    return re.sub(r"\W+", " ", txt.lower()).strip()


def clean_title(raw) -> str | None:
    if pd.isna(raw):
        return None
    txt = str(raw).strip()
    if txt.startswith("[") and txt.endswith("]"):
        try:
            lst = ast.literal_eval(txt)
            if isinstance(lst, list) and lst:
                txt = str(lst[0])
        except Exception:
            pass
    txt = re.sub(r"\s+", " ", txt.strip(" '\""))
    return txt if txt else None


# ── HTTP session with retry/back-off ──────────────────────────────────

S = requests.Session()
S.headers.update({"User-Agent": f"pubmed_title_fill/0.1 ({EMAIL})"})
S.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=6,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
    ),
)


def esearch_ids(title: str) -> List[str]:
    term = f'"{title}"[Title]'
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": 5,
        "api_key": API_KEY,
    }
    r = S.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params=params,
        timeout=20,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    return [id_el.text for id_el in root.findall(".//Id")]


def efetch_article(pmid: str) -> tuple[str, str] | None:
    params = {"db": "pubmed", "id": pmid, "retmode": "xml", "api_key": API_KEY}
    r = S.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params=params,
        timeout=20,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    art = root.find(".//Article")
    if art is None:
        return None
    tit = art.findtext(".//ArticleTitle", default="")
    abs_el = art.find(".//Abstract/AbstractText")
    abs_txt = "".join(abs_el.itertext()).strip() if abs_el is not None else ""
    return tit, abs_txt


def fetch(title: str) -> str | None:
    try:
        for pmid in esearch_ids(title):
            fetched = efetch_article(pmid)
            if not fetched:
                continue
            tit, abs_txt = fetched
            if abs_txt and _sim(norm(title), norm(tit)) >= FUZZ_OK:
                return abs_txt
    except Exception as exc:
        log.error("REQ_FAIL %s – %s", title[:80], exc)
    return None


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    log.info("Loading %s", SRC_FILE)
    df = pd.read_csv(SRC_FILE)
    if "abstract" not in df.columns:
        df["abstract"] = pd.NA

    targets: List[Tuple[int, str]] = [
        (i, clean_title(t))
        for i, (t, a) in enumerate(zip(df["title_norm"], df["abstract"]))
        if pd.isna(a) and clean_title(t)
    ]
    log.info("Need %d PubMed look-ups", len(targets))

    done = 0
    for idx, title in tqdm(targets, desc="PubMed look-ups"):
        abs_txt = fetch(title)
        if abs_txt:
            df.at[idx, "abstract"] = abs_txt
            log.info("FOUND %s", title[:100])
        time.sleep(PAUSE_SEC)
        done += 1
        if done % CKPT_EVERY == 0:
            df.to_csv(DST_FILE, index=False)
            log.info("Checkpoint at %d / %d", done, len(targets))

    df.to_csv(DST_FILE, index=False)
    remaining = df["abstract"].isna().sum()
    log.info("Finished – %d abstracts still missing", remaining)
    print(f"Done → {DST_FILE}. {remaining} abstracts still missing.")


if __name__ == "__main__":
    main()
