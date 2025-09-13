"""
Title-based abstract fetcher (Crossref → OpenAlex → Microsoft Academic)
with YEAR ±1 and AUTHOR fuzzy guards.
"""

# ---------- CONFIG -----------------------------------------------------
SRC_FILE = (
    "../../../data/extracted_information/refs/with_abs/references_title_with_abs.csv"
)
DST_FILE = (
    "../../../data/extracted_information/refs/with_abs/references_title_with_abs_v2.csv"
)
LOG_FILE = "../../../logs/title_fetch_3.log"
CHECKPOINT_EVERY = 250  # rows
MAX_WORKERS = 10  # threads – stay polite
YEAR_TOLERANCE = 1  # ±1 year accepted
FUZZ_TITLE_MIN = 95  # rapid-fuzz ratio threshold for title equality
FUZZ_SURNAME_MIN = 85  # partial ratio threshold for surname equality
SKIP_BEFORE = 100150
# ----------------------------------------------------------------------

import ast
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests
from rapidfuzz import fuzz
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

LOG_LEVEL = os.getenv("TITLE_FETCH_LOGLEVEL", "INFO").upper()
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(threadName)s::%(name)s – %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("Logger initialised at level %s", LOG_LEVEL)

CR_API = "https://api.crossref.org/works"
OA_API = "https://api.openalex.org/works"
MA_API = "https://api.labs.cognitive.microsoft.com/academic/v1.0/evaluate"

HEADERS = {
    "User-Agent": (
        "title-abstract-fetch/0.9 "
        "(mailto:pieer.achkar@imw.fraunhofer.de; orcid:0000-0000-0000-0000)"
    )
}

MS_KEY = os.getenv("MS_ACAD_KEY")
HEADERS_MS = {"Ocp-Apim-Subscription-Key": MS_KEY, **HEADERS} if MS_KEY else None

_session: Optional[requests.Session] = None


def get_session() -> requests.Session:
    """Singleton HTTP session with retries."""
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        s.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=5,
                    backoff_factor=1.0,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET"],
                ),
            ),
        )
        _session = s
    return _session


# ---------------- helpers -----------------


def norm(txt: str) -> str:
    return re.sub(r"\W+", " ", str(txt)).strip().lower()


def first_surname(author_field) -> Optional[str]:
    """Extract first author's surname from Crossref CSV field (JSON list-of-dicts)."""
    if pd.isna(author_field):
        return None
    try:
        data = (
            ast.literal_eval(author_field)
            if isinstance(author_field, str)
            else author_field
        )
        if isinstance(data, list) and data:
            return norm(data[0].get("family", ""))
    except Exception:
        pass
    return None


def year_from(date_field) -> Optional[int]:
    m = re.search(r"\d{4}", str(date_field))
    return int(m.group()) if m else None


def strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html, flags=re.S)


def clean_titles(raw_title: str) -> List[str]:
    """Normalise ‘title’ column: can be a JSON list or plain string."""
    raw_title = str(raw_title).strip()
    if not raw_title:
        return []
    if raw_title.startswith("[") and raw_title.endswith("]"):
        try:
            items = ast.literal_eval(raw_title)
            return [str(t).strip(" \"'\n") for t in items if t]
        except Exception:
            pass
    return [raw_title.rstrip("… .")]


# ------------- fuzzy-match helpers ----------------


def title_matches(a: str, b: str) -> bool:
    return fuzz.ratio(norm(a), norm(b)) >= FUZZ_TITLE_MIN


def year_matches(a: Optional[int], b: Optional[int]) -> bool:
    return (a is None or b is None) or abs(a - b) <= YEAR_TOLERANCE


def surname_matches(a: Optional[str], b: Optional[str]) -> bool:
    return (not a or not b) or fuzz.partial_ratio(a, b) >= FUZZ_SURNAME_MIN


# ---------- new safe-surname extraction helpers -----------------


def surname_from_openalex(rec) -> Optional[str]:
    auth = rec.get("authorships") or []
    if not auth:
        return None
    display = (auth[0].get("author") or {}).get("display_name", "")
    parts = display.split()
    return norm(parts[-1]) if parts else None


def surname_from_ms(ent) -> Optional[str]:
    aa = ent.get("AA") or []
    if not aa:
        return None
    display = aa[0].get("AuN", "")
    parts = display.split()
    return norm(parts[-1]) if parts else None


# -------------------------------------------------

FetcherResult = Tuple[Optional[str], bool, str]


def timed_request(src: str, url: str, **kw) -> Optional[requests.Response]:
    t0 = time.perf_counter()
    try:
        r = get_session().get(url, **kw)
        r.raise_for_status()
        logger.debug("%s OK %.2fs", src, time.perf_counter() - t0)
        return r
    except requests.exceptions.RequestException as exc:
        logger.error("REQUEST_FAIL_%s %s – %s", src.upper(), url, exc)
        return None


# ---------------- individual source look-ups ----------------------


def crossref_by_title(
    title: str, year: Optional[int], surname: Optional[str]
) -> FetcherResult:
    res = timed_request(
        "Crossref", CR_API, params={"query.title": title, "rows": 5}, timeout=(4, 20)
    )
    if res is None:
        return None, False, "Crossref"
    for it in res.json()["message"].get("items", []):
        if not title_matches(title, it.get("title", [""])[0]):
            continue
        if not year_matches(
            year, it.get("issued", {}).get("date-parts", [[None]])[0][0]
        ):
            continue
        # Crossref already uses helper first_surname on CSV side; here we extract on the fly
        cr_surname = (
            norm(it.get("author", [{}])[0].get("family", ""))
            if it.get("author")
            else None
        )
        if not surname_matches(surname, cr_surname):
            continue
        raw = it.get("abstract")
        return strip_tags(raw).strip() if raw else "", True, "Crossref"
    return None, False, "Crossref"


def openalex_by_title(
    title: str, year: Optional[int], surname: Optional[str]
) -> FetcherResult:
    res = timed_request(
        "OpenAlex", OA_API, params={"search": title, "per-page": 5}, timeout=(4, 20)
    )
    if res is None:
        return None, False, "OpenAlex"
    for rec in res.json().get("results", []):
        if not title_matches(title, rec.get("display_name", "")):
            continue
        if not year_matches(year, rec.get("publication_year")):
            continue
        if not surname_matches(surname, surname_from_openalex(rec)):
            continue
        idx = rec.get("abstract_inverted_index")
        if idx:
            words = sorted(idx.items(), key=lambda kv: kv[1][0])
            return " ".join(w for w, _ in words), True, "OpenAlex"
        return "", True, "OpenAlex"
    return None, False, "OpenAlex"


def microsoft_by_title(
    title: str, year: Optional[int], surname: Optional[str]
) -> FetcherResult:
    if not HEADERS_MS:
        return None, False, "MSAcademic"
    expr = f"Ti='{title.replace('\\', ' ').replace("'", ' ')}'"
    res = timed_request(
        "MSAcademic",
        MA_API,
        params={
            "expr": expr,
            "model": "latest",
            "count": 5,
            "attributes": "Ti,Y,AA.AuN,IA",
        },
        headers=HEADERS_MS,
        timeout=(4, 20),
    )
    if res is None:
        return None, False, "MSAcademic"
    for ent in res.json().get("entities", []):
        if not title_matches(title, ent.get("Ti", "")):
            continue
        if not year_matches(year, ent.get("Y")):
            continue
        if not surname_matches(surname, surname_from_ms(ent)):
            continue
        idx = ent.get("IA", {}).get("InvertedIndex")
        if idx:
            words = sorted(idx.items(), key=lambda kv: kv[1][0])
            return " ".join(w for w, _ in words), True, "MSAcademic"
        return "", True, "MSAcademic"
    return None, False, "MSAcademic"


# ------------- checkpoint helper ----------------


def save_checkpoint(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Checkpoint saved to %s (%d rows)", path, len(df))


# ------------- row processor -------------------


def process_row(
    payload: Tuple[int, str, Optional[str], Optional[str]],
) -> Tuple[int, Optional[str]]:
    idx, title_raw, date_raw, author_raw = payload
    titles = clean_titles(title_raw)
    if not titles:
        logger.info("ROW %d – NO_TITLE", idx)
        return idx, None

    year = year_from(date_raw)
    surname = first_surname(author_raw)

    for t in titles:
        matched_any = False
        for fetcher in (crossref_by_title, openalex_by_title, microsoft_by_title):
            abs_txt, matched, source = fetcher(t, year, surname)
            if matched:
                matched_any = True
                if abs_txt:  # got text: we're done
                    logger.info("ROW %d – FOUND via %s", idx, source)
                    return idx, abs_txt
                logger.debug(
                    "ROW %d – empty abstract from %s, trying next source…", idx, source
                )
        if matched_any:
            logger.warning("ROW %d – MATCH_NO_ABS (all sources empty)", idx)
            return idx, None

    logger.warning("ROW %d – NOT_FOUND", idx)
    return idx, None


# ------------- main ---------------------------


def main():
    t0 = time.perf_counter()
    logger.info("Loading %s …", SRC_FILE)
    df = pd.read_csv(SRC_FILE)
    if "abstract" not in df.columns:
        logger.error("No 'abstract' column – aborting")
        return

    targets = [
        (i, r.title, getattr(r, "date", None), getattr(r, "author", None))
        for i, r in enumerate(df.itertuples())
        if i >= SKIP_BEFORE and pd.isna(r.abstract)
    ]
    logger.info(
        "%d rows need lookup (starting at %d, df len %d)",
        len(targets),
        SKIP_BEFORE,
        len(df),
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fut_map = {pool.submit(process_row, p): p[0] for p in targets}
        for fut in tqdm(
            as_completed(fut_map), total=len(fut_map), desc="Title lookups"
        ):
            idx, abs_txt = fut.result()
            if abs_txt is not None:
                df.at[idx, "abstract"] = abs_txt
            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                save_checkpoint(df, DST_FILE)

    save_checkpoint(df, DST_FILE)
    missing = df["abstract"].isna().sum()
    logger.info(
        "Finished – %d missing • runtime %.2f min",
        missing,
        (time.perf_counter() - t0) / 60,
    )
    print(f"Done. {missing} abstracts still missing.")


if __name__ == "__main__":
    main()
