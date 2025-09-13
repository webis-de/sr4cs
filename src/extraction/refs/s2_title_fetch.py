"""
Fetch missing abstracts from Semantic Scholar by title search.
"""

# ───────────── CONFIG ────────────────────────────────────────────────
SRC_FILE = "../../../data/extracted_information/refs/with_abs/still_missing/references_split_no_abs_part_1.csv"
DST_FILE = "../../../data/extracted_information/refs/with_abs/still_missing/references_split_no_abs_part_1_s2_filled.csv"
LOG_FILE = "../../../logs/s2_title_fetch_part_1.log"

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,year,abstract"
PAUSE_SEC = 3.1  # 100 requests / 5 min
CKPT_EVERY = 200  # rows
FUZZ_OK = 92  # Jaro–Winkler %

# ──────────────────────────────────────────────────────────────────────

import ast, logging, re, textwrap, time, urllib.parse as ul
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

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("s2")

# ── helpers ───────────────────────────────────────────────────────────


def norm(txt: str) -> str:
    return re.sub(r"\W+", " ", txt.lower()).strip()


def clean_title(raw) -> str | None:
    """Return a plain title string, or None if empty."""
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

SESSION = requests.Session()
SESSION.mount(
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


def fetch(title: str) -> str | None:
    """Return abstract string or None for one title."""
    params = {"query": title, "limit": 1, "fields": FIELDS}
    try:
        r = SESSION.get(S2_URL, params=params, timeout=20)
        if r.status_code == 429:
            retry_for = int(r.headers.get("retry-after", "15"))
            log.warning("429 – sleeping %s s", retry_for)
            time.sleep(retry_for)
            return fetch(title)  # one retry after wait
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None
        hit = data[0]
        if _sim(norm(title), norm(hit.get("title", ""))) >= FUZZ_OK:
            return hit.get("abstract")
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
        for i, (t, a) in enumerate(zip(df["title"], df["abstract"]))
        if pd.isna(a) and clean_title(t)
    ]
    log.info("Need %d look-ups", len(targets))

    done = 0
    for idx, title in tqdm(targets, desc="S2 look-ups"):
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
