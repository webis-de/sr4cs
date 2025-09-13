"""
Fetch missing abstracts via DBLP title search (loose only; author/year gating).
Then write them back into the CSV.
"""

import os, re, json, time, random, logging
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from tqdm import tqdm  # progress bar

# =========================
# CONFIG
# =========================
INPUT_CSV = (
    "../../../data/extracted_information/refs/still_missing/references_split_part2.csv"
)
OUTPUT_CSV = None
TITLE_COL = "title_norm"
DATE_COL = "date"
AUTHOR_COL = "author"
SAVE_EVERY = 20
DELAY_SEC = 2.0
MAX_RETRIES = 3
LOG_FILE = "../../../logs/dblp_title.log"

# Matching parameters
MAX_HITS = 80  # max DBLP hits to consider
CONF_THRESH = 0.80  # base title threshold (token-set ratio driven)
CONF_STRONG = 0.90  # accept even without author/year if very high
YEAR_TOL = 2  # ± tolerance when year available
SHORT_TITLE_K = 4  # token threshold considered "short"

# Option 1: allow a book review if it's the ONLY hit and the title match is very strong
ALLOW_BOOK_REVIEWS_IF_ONLY_HIT = True
BOOK_REVIEW_STRONG_CONF = 0.95

UA = "Academic Research Tool/1.0 (polite bot; contact: research use)"
HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# =========================
# Logging (FILE ONLY)
# =========================
os.makedirs(os.path.dirname(LOG_FILE) if LOG_FILE else "./logs", exist_ok=True)
LOG_FILE = LOG_FILE or os.path.join(
    "./logs", f"dblp_title_fill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")],
    # NOTE: no StreamHandler → logs go to file only
)
log = logging.getLogger("dblp_fill")

# =========================
# HTTP with retries
# =========================
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get_with_retries(
    url: str, params: dict = None, timeout: int = 25
) -> Optional[requests.Response]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(
                url, params=params, timeout=timeout, allow_redirects=True
            )
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = (2 ** (attempt - 1)) * DELAY_SEC + random.uniform(1, 3)
                log.warning(
                    f"429 from {url} — retry {attempt}/{MAX_RETRIES} after {wait:.1f}s"
                )
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = (2 ** (attempt - 1)) * 2 + random.uniform(1, 2)
                log.warning(
                    f"{resp.status_code} from {url} — retry {attempt}/{MAX_RETRIES} after {wait:.1f}s"
                )
                time.sleep(wait)
                continue
            log.info(f"Request failed {resp.status_code} for {url}")
            return None
        except requests.RequestException as e:
            wait = (2 ** (attempt - 1)) * 1.5
            log.warning(
                f"Request error {e} — retry {attempt}/{MAX_RETRIES} after {wait:.1f}s"
            )
            time.sleep(wait)
    return None


# =========================
# Utils
# =========================
STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "from",
    "and",
    "or",
    "is",
    "are",
    "be",
    "as",
}


def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip())
    s = re.sub(r"[^\w\s]", "", s.lower()).strip()
    return s


def _tokens(s: str) -> List[str]:
    return [t for t in _clean(s).split() if t and t not in STOP]


def parse_first_author_family(author_field: Any) -> Optional[str]:
    if author_field is None:
        return None
    s = str(author_field).strip()
    if not s:
        return None
    s2 = s.replace("’", "'").replace("“", '"').replace("”", '"')
    try_variants = [s2, s2.replace("'", '"')]
    for cand in try_variants:
        try:
            data = json.loads(cand)
            if isinstance(data, list) and data:
                first = data[0]
                fam = first.get("family") if isinstance(first, dict) else None
                if fam:
                    return str(fam).lower()
        except Exception:
            pass
    m = re.search(r"['\"]family['\"]\s*:\s*['\"]([^'\"]+)['\"]", s2)
    return m.group(1).strip().lower() if m else None


def extract_year(date_field: Any) -> Optional[int]:
    if date_field is None:
        return None
    s = str(date_field)
    m = re.match(r"(\d{4})", s)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None


def is_book_review_str(s: str) -> bool:
    return bool(re.search(r"\bbook\s+review\b", s, re.I))


# =========================
# DBLP search (loose only)
# =========================
_title_cache: Dict[str, Optional[dict]] = {}


def dblp_loose_query(title: str) -> Optional[List[dict]]:
    q = title  # as-is
    url = "https://dblp.org/search/publ/api"
    params = {"q": q, "format": "json", "h": 1000, "c": 0}
    log.info(f"🔎 DBLP LOOSE query: {q}")
    time.sleep(DELAY_SEC + random.uniform(0, 0.3))
    resp = get_with_retries(url, params=params)
    if not resp:
        log.info("❌ DBLP request failed")
        return None
    try:
        data = resp.json()
    except Exception:
        log.info("❌ DBLP invalid JSON")
        return None
    hits = data.get("result", {}).get("hits", {}).get("hit", [])
    if not isinstance(hits, list):
        hits = [hits] if hits else []
    log.info(f"📊 DBLP hits: {len(hits)}")
    return hits


def parse_dblp_authors(info: dict) -> List[str]:
    out = []
    authors = info.get("authors")
    if isinstance(authors, dict):
        a = authors.get("author", [])
        if not isinstance(a, list):
            a = [a] if a else []
        for item in a:
            name = item.get("text") if isinstance(item, dict) else None
            if name:
                out.append(str(name).lower())
    return out


def extract_doi_oa_arxiv(
    ee_field,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not ee_field:
        return None, None, None
    if isinstance(ee_field, str):
        links = [ee_field]
    elif isinstance(ee_field, list):
        links = [str(x) for x in ee_field if x]
    else:
        links = []
    doi = arxiv_id = best_pdf = None
    for link in links:
        low = link.lower()
        m = re.search(r"10\.\d{4,9}/[^\s]+", link)
        if not doi and m:
            doi = m.group(0)
        if "doi.org/" in low and not doi:
            m2 = re.search(r"doi\.org/(.+)$", low)
            if m2:
                doi = m2.group(1)
        if "arxiv.org" in low and not arxiv_id:
            m3 = re.search(
                r"arxiv\.org/(abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:\.pdf)?", low
            )
            if m3:
                arxiv_id = m3.group(2)
        if (low.endswith(".pdf") or "arxiv.org" in low) and not best_pdf:
            best_pdf = (
                f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                if ("arxiv.org" in low and arxiv_id)
                else link
            )
    return doi, arxiv_id, best_pdf


def title_conf(q: str, cand: str) -> float:
    q2, c2 = _clean(q), _clean(cand)
    if not q2 or not c2:
        return 0.0
    # Rely mainly on token-set ratio for loose matching
    return fuzz.token_set_ratio(q2, c2) / 100.0


def choose_best_hit(
    title: str,
    hits: List[dict],
    row_year: Optional[int],
    first_author_family: Optional[str],
) -> Optional[dict]:
    best = None
    best_score = 0.0
    q_tokens = _tokens(title)
    is_short = len(q_tokens) < SHORT_TITLE_K

    for i, h in enumerate(hits[:MAX_HITS], start=1):
        info = h.get("info", {}) or {}
        dblp_title = info.get("title", "") or ""
        dblp_year = info.get("year")
        dblp_auths = parse_dblp_authors(info)

        score = title_conf(title, dblp_title)

        # Author/year checks
        yr_ok = True
        au_ok = True

        if row_year and str(dblp_year).isdigit():
            yr_ok = abs(int(dblp_year) - row_year) <= YEAR_TOL

        if first_author_family and dblp_auths:
            au_ok = any(first_author_family in a for a in dblp_auths)

        # For short/generic titles, require both signals if available
        if is_short:
            if row_year:
                if not yr_ok:
                    pass
            if first_author_family:
                if not au_ok:
                    pass

        flags = []
        book_rev = is_book_review_str(dblp_title)
        if book_rev:
            flags.append("book_review")
        if not yr_ok and row_year is not None:
            flags.append(f"year_off:{dblp_year}")
        if not au_ok and first_author_family:
            flags.append("author_mismatch")

        # Option 1 logic: allow book review if it's the only hit and very strong
        reject = False
        if flags:
            if (
                book_rev
                and ALLOW_BOOK_REVIEWS_IF_ONLY_HIT
                and len(hits) <= 1
                and score >= BOOK_REVIEW_STRONG_CONF
            ):
                log.info(
                    f"  [{i}] {dblp_title[:100]} — conf={score:.2f} — "
                    f"ALLOW (book_review, only hit, strong match)"
                )
            else:
                reject = True

        if reject:
            log.info(
                f"  [{i}] {dblp_title[:100]} — conf={score:.2f} — REJECT ({', '.join(flags)})"
            )
            continue
        else:
            log.info(f"  [{i}] {dblp_title[:100]} — conf={score:.2f}")

        # Acceptance logic:
        accept = False
        if is_short:
            need_au = first_author_family is not None
            need_yr = row_year is not None
            conds = []
            if need_au:
                conds.append(au_ok)
            if need_yr:
                conds.append(yr_ok)
            base_ok = score >= max(CONF_THRESH, 0.75)
            accept = base_ok and all(conds) if conds else base_ok
        else:
            if score >= CONF_STRONG:
                accept = True
            else:
                conds = [score >= CONF_THRESH]
                if first_author_family is not None:
                    conds.append(au_ok)
                if row_year is not None:
                    conds.append(yr_ok)
                accept = all(conds)

        if accept and score > best_score:
            dblp_doi, arxiv_id, best_pdf = extract_doi_oa_arxiv(info.get("ee"))
            best = {
                "dblp_title": dblp_title,
                "dblp_doi": dblp_doi,
                "dblp_arxiv_id": arxiv_id,
                "dblp_best_oa_location_pdf": best_pdf,
                "venue": info.get("venue"),
                "year": info.get("year"),
                "ee": (
                    info.get("ee")
                    if isinstance(info.get("ee"), list)
                    else [info.get("ee")] if info.get("ee") else []
                ),
                "dblp_authors": dblp_auths,
            }
            best_score = score

    if best:
        log.info(f"✅ Chosen hit: {best['dblp_title']}")
        if best["dblp_doi"]:
            log.info(f"   DOI: {best['dblp_doi']}")
        if best["dblp_arxiv_id"]:
            log.info(f"   arXiv: {best['dblp_arxiv_id']}")
        if best["dblp_best_oa_location_pdf"]:
            log.info(f"   OA PDF: {best['dblp_best_oa_location_pdf']}")
    else:
        log.info("❌ No sufficiently confident DBLP hit")
    return best


def search_dblp_by_title_loose(
    title: str, row_year: Optional[int], first_author_family: Optional[str]
) -> Optional[dict]:
    if not title or str(title).strip() == "" or str(title).lower() == "nan":
        return None

    key = f"{str(title).strip().lower()}|{row_year}|{first_author_family or ''}"
    if key in _title_cache:
        log.info(f"💾 Cache hit for: {title[:80]}...")
        return _title_cache[key]

    hits = dblp_loose_query(title)
    hit = choose_best_hit(title, hits or [], row_year, first_author_family)
    _title_cache[key] = hit
    return hit


# =========================
# Abstract scraping
# =========================
META_NAMES = [
    ("name", "description"),
    ("property", "og:description"),
    ("name", "DC.Description"),
    ("name", "citation_abstract"),
]
CSS_SELECTORS = [
    "#abstract",
    ".abstract",
    ".abstract-text",
    ".abstractSection",
    "[data-testid='abstract']",
    "[role='doc-abstract']",
    "[data-type='abstract']",
    ".c-article-section__content",
    ".chapter-abstract",
    ".hlFld-Abstract",
    ".article-section__content",
    "#Abs1",
    "#Abs1-content",
    "section.abstract",
    "#abstracts",
    ".abstract-content",
    ".simple-para",
]


def clean_abstract(text: str) -> str:
    if not text:
        return ""
    t = re.sub(
        r"^\s*(abstract|summary|synopsis|overview)\s*:?\s*", "", text, flags=re.I
    )
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) >= 100 else ""


def fetch_html(url: str) -> Optional[BeautifulSoup]:
    time.sleep(0.8 + random.uniform(0, 0.4))
    resp = get_with_retries(url, None, 25)
    if not resp:
        return None
    return BeautifulSoup(resp.content, "html.parser")


def try_extract_abstract(page_url: str) -> Tuple[Optional[str], str]:
    soup = fetch_html(page_url)
    if not soup:
        return None, "no_response"

    # JSON-LD (Elsevier, etc.)
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, dict) and "description" in data:
                txt = clean_abstract(data["description"])
                if txt:
                    return txt, "jsonld_description"
        except Exception:
            pass

    # meta tags
    for k, v in META_NAMES:
        m = soup.find("meta", {k: v})
        if m and m.get("content"):
            txt = clean_abstract(m["content"])
            if txt:
                return txt, f"meta_{v}"

    # CSS selectors
    for sel in CSS_SELECTORS:
        el = soup.select_one(sel)
        if el:
            txt = clean_abstract(el.get_text(" ", strip=True))
            if txt:
                return txt, f"css_{sel}"

    # fallback: first substantial <p>
    for p in soup.find_all("p"):
        txt = clean_abstract(p.get_text(" ", strip=True))
        if txt:
            return txt, "p_fallback"

    return None, "not_found"


def attempt_publisher_abstract(links: List[str]) -> Tuple[Optional[str], Optional[str]]:
    for link in links or []:
        if not link:
            continue
        log.info(f"   → Try publisher: {link}")
        abs_text, note = try_extract_abstract(link)
        if abs_text:
            log.info(f"      ✅ abstract found via {note}, len={len(abs_text)}")
            return abs_text, link
        else:
            log.info(f"      ❌ no abstract ({note})")
    return None, None


# =========================
# DataFrame I/O
# =========================
def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "abstract",
        "dblp_title",
        "dblp_doi",
        "dblp_arxiv_id",
        "dblp_best_oa_location_pdf",
        "dblp_processed",
    ]:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    log.info(f"💾 Saved: {path}")


# =========================
# Main
# =========================
def main():
    log.info("=" * 80)
    log.info(
        "Start: DBLP Title → DOI/arXiv/OA → Abstract (loose-only + author/year gating)"
    )
    log.info(f"Input: {INPUT_CSV}")
    log.info(
        f"Save every: {SAVE_EVERY} rows | Delay: {DELAY_SEC}s | Retries: {MAX_RETRIES}"
    )
    log.info("=" * 80)

    out_path = OUTPUT_CSV or INPUT_CSV.replace(".csv", "_with_dblp_abstracts.csv")

    # backup once
    df = pd.read_csv(INPUT_CSV)
    backup = INPUT_CSV.replace(
        ".csv", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    df.to_csv(backup, index=False)
    log.info(f"Backup created: {backup}")

    df = ensure_cols(df)

    if TITLE_COL not in df.columns:
        log.error(f"Missing title column: {TITLE_COL}")
        log.info(f"Available columns: {list(df.columns)}")
        return
    if DATE_COL not in df.columns:
        log.warning("Note: date column missing; year filtering will be weaker.")
    if AUTHOR_COL not in df.columns:
        log.warning("Note: author column missing; author filtering will be skipped.")

    # resume: skip already processed
    mask_unprocessed = df["dblp_processed"].isna() | (df["dblp_processed"] == False)
    mask_need_title = df[TITLE_COL].notna() & (
        df[TITLE_COL].astype(str).str.strip() != ""
    )
    work_idx = df.index[mask_unprocessed & mask_need_title].tolist()

    log.info(f"Total rows: {len(df)} | To process now: {len(work_idx)}")

    processed_since_save = 0
    abstracts_found = 0

    # tqdm progress bar (logs go to file only; console shows just the bar)
    with tqdm(
        total=len(work_idx), desc="🔍 DBLP (loose)", unit="row", dynamic_ncols=True
    ) as pbar:
        for idx in work_idx:
            title = str(df.at[idx, TITLE_COL])

            # Extract year, first author (if available)
            row_year = (
                extract_year(df.at[idx, DATE_COL]) if DATE_COL in df.columns else None
            )
            first_author_family = (
                parse_first_author_family(df.at[idx, AUTHOR_COL])
                if AUTHOR_COL in df.columns
                else None
            )

            log.info("-" * 70)
            log.info(f"Row {idx} | Title: {title[:140]}")
            if row_year:
                log.info(f"   Year: {row_year}")
            if first_author_family:
                log.info(f"   First author (family): {first_author_family}")

            # DBLP search — loose only
            hit = search_dblp_by_title_loose(title, row_year, first_author_family)

            if hit:
                # store metadata regardless of abstract success
                df.at[idx, "dblp_title"] = hit.get("dblp_title")
                df.at[idx, "dblp_doi"] = hit.get("dblp_doi")
                df.at[idx, "dblp_arxiv_id"] = hit.get("dblp_arxiv_id")
                df.at[idx, "dblp_best_oa_location_pdf"] = hit.get(
                    "dblp_best_oa_location_pdf"
                )

                # try publisher pages
                abstract, _ = attempt_publisher_abstract(hit.get("ee", []))
                if abstract:
                    df.at[idx, "abstract"] = abstract
                    abstracts_found += 1
                    log.info("✅ SUCCESS: abstract captured")
                else:
                    log.info(
                        "❌ Found DBLP match but no abstract on any publisher page"
                    )
            else:
                log.info("❌ No DBLP match; skipping abstract attempt")

            # mark processed
            df.at[idx, "dblp_processed"] = True
            processed_since_save += 1

            # periodic save
            if processed_since_save >= SAVE_EVERY:
                save_df(df, out_path)
                processed_since_save = 0

            # update progress bar
            pbar.set_postfix(abs=abstracts_found)
            pbar.update(1)

            # politeness
            time.sleep(DELAY_SEC)

    # final save
    save_df(df, out_path)

    # summary
    tot_abs = df["abstract"].notna().sum()
    tot_doi = df["dblp_doi"].notna().sum()
    tot_arx = df["dblp_arxiv_id"].notna().sum()
    tot_pdf = df["dblp_best_oa_location_pdf"].notna().sum()
    log.info("=" * 80)
    log.info("DONE.")
    log.info(f"Abstracts total: {tot_abs}")
    log.info(f"DOIs total:     {tot_doi}")
    log.info(f"arXiv IDs total:{tot_arx}")
    log.info(f"OA PDFs total:  {tot_pdf}")
    log.info(f"Output: {out_path}")
    log.info(f"Log:    {LOG_FILE}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
