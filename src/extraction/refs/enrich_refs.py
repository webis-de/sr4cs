"""
Enrich references with abstracts from various sources (Crossref, OpenAlex,
Europe-PMC, arXiv) using DOIs or titles for lookup.
"""

import csv, re, time, logging, difflib, xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────────
IN_CSV = "../../../data/extracted_information/refs/processed/references_dedup.csv"
OUT_CSV = "../../../data/extracted_information/refs/processed/refs_enriched.csv"
LOG_FILE = "/Users/pierreachkar/Documents/projects/slr_ds/logs/enrich_refs.log"

USERMAIL = "pieer.achkar@imw.fraunhofer.de"  # real e-mail → polite UA
SLEEP = 0.20  # seconds between HTTP calls
SIM_TH = 0.90  # title-similarity threshold
SAVE_EVERY = 50  # checkpoint interval
# ────────────────────────────────────────────────────────────────────────────────

HEADERS = {"User-Agent": f"abstract-enricher/0.1 (mailto:{USERMAIL})"}
doi_rx = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")
arxiv_id_rx = re.compile(r"(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(v\d+)?$", re.I)


# ─── HELPERS ────────────────────────────────────────────────────────────────────
def strip_html(jats: str | None) -> Optional[str]:
    return BeautifulSoup(jats, "lxml").get_text(" ", strip=True) if jats else None


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def norm_title(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        raw = " ".join(raw)
    raw = re.sub(r"^[\[\('\" ]+|[\]\)'\" ]+$", "", str(raw))
    return " ".join(raw.split())


# ─── SOURCE QUERIES (Crossref, OpenAlex, Europe-PMC, arXiv) ────────────────────
def cr_by_doi(doi: str):
    try:
        m = requests.get(
            f"https://api.crossref.org/works/{doi}", headers=HEADERS, timeout=15
        ).json()["message"]
        return strip_html(m.get("abstract")), (m.get("title") or [""])[0]
    except Exception:
        return None, None


def cr_by_title(title: str):
    try:
        it = requests.get(
            "https://api.crossref.org/works",
            headers=HEADERS,
            params={"query.bibliographic": title, "rows": 1},
            timeout=15,
        ).json()["message"]["items"][0]
        return strip_html(it.get("abstract")), (it.get("title") or [""])[0]
    except Exception:
        return None, None


def oa_by_doi(doi: str):
    try:
        j = requests.get(
            f"https://api.openalex.org/works/https://doi.org/{doi}",
            headers=HEADERS,
            timeout=15,
        ).json()
        idx = j.get("abstract_inverted_index")
        return (" ".join(idx) if idx else None), j.get("title")
    except Exception:
        return None, None


def oa_by_title(title: str):
    try:
        j = requests.get(
            "https://api.openalex.org/works",
            headers=HEADERS,
            params={"filter": f'title.search:"{title}"', "per-page": 1},
            timeout=15,
        ).json()
        it = (j["results"] or [None])[0]
        if not it:
            return None, None
        idx = it.get("abstract_inverted_index")
        return (" ".join(idx) if idx else None), it.get("title")
    except Exception:
        return None, None


def ep_hit(expr: str):
    try:
        j = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            headers=HEADERS,
            params={"query": expr, "format": "json", "pageSize": 1},
            timeout=15,
        ).json()
        return (j.get("resultList", {}).get("result") or [None])[0]
    except Exception:
        return None


def epmc_by_doi(doi: str):
    h = ep_hit(f'DOI:"{doi}"')
    return (h or {}).get("abstractText"), (h or {}).get("title")


def epmc_by_title(title: str):
    h = ep_hit(f'TITLE:"{title}"')
    return (h or {}).get("abstractText"), (h or {}).get("title")


def ax_get(arx_id: str) -> Optional[str]:
    try:
        root = ET.fromstring(
            requests.get(
                f"https://export.arxiv.org/api/query?search_query=id:{arx_id}&max_results=1",
                headers=HEADERS,
                timeout=15,
            ).text
        )
        return (
            (root.findtext(".//{http://www.w3.org/2005/Atom}summary") or "")
            .strip()
            .replace("\n", " ")
        )
    except Exception:
        return None


def ax_by_title(title: str):
    try:
        root = ET.fromstring(
            requests.get(
                f"https://export.arxiv.org/api/query?search_query=all:{'+'.join(title.split())}&max_results=1",
                headers=HEADERS,
                timeout=15,
            ).text
        )
        e = root.find("{http://www.w3.org/2005/Atom}entry")
        if e is None:
            return None, None
        abs_txt = e.findtext("{http://www.w3.org/2005/Atom}summary")
        return abs_txt.strip().replace("\n", " "), e.findtext(
            "{http://www.w3.org/2005/Atom}title"
        )
    except Exception:
        return None, None


# ─── ENRICH ONE ROW ────────────────────────────────────────────────────────────
def enrich(row: Dict[str, str], log) -> str:
    if row.get("abstract"):
        row.setdefault("abstract_src", "given")
        return "already"

    doi = (row.get("doi") or "").strip().lower()
    title = norm_title(row.get("title"))

    # DOI-based ----------------------------------------------------------
    if doi:
        # arXiv DOI alias
        if doi.startswith("10.48550/"):
            abs_txt = ax_get(doi.split("/", 1)[1])
            log.debug("  · arxiv_doi | %s | %s", "hit" if abs_txt else "miss", [title])
            if abs_txt:
                row.update(abstract=abs_txt, abstract_src="arxiv_doi")
                log.info("hit via arxiv_doi")
                return "arxiv_doi"

        for tag, fn in [
            ("cr_doi", cr_by_doi),
            ("oa_doi", oa_by_doi),
            ("ep_doi", epmc_by_doi),
        ]:
            abs_txt, t = fn(doi)
            hit = bool(abs_txt and t and sim(title or t, t) >= SIM_TH)
            log.debug("  · %-7s | %s | %s", tag, "hit" if hit else "miss", [title])
            if hit:
                row.update(abstract=abs_txt, abstract_src=tag)
                log.info("hit via %s", tag)
                return tag
            time.sleep(SLEEP)

    # explicit arXiv id?
    for fld in ("arxiv_id", "id"):
        arx = (row.get(fld) or "").strip()
        if arxiv_id_rx.match(arx):
            abs_txt = ax_get(arx)
            log.debug("  · arxiv_id  | %s | %s", "hit" if abs_txt else "miss", [title])
            if abs_txt:
                row.update(abstract=abs_txt, abstract_src="arxiv_id")
                log.info("hit via arxiv_id")
                return "arxiv_id"

    # Title-based --------------------------------------------------------
    if not title:
        return "no_title"

    for tag, fn in [
        ("cr_tit", cr_by_title),
        ("oa_tit", oa_by_title),
        ("ep_tit", epmc_by_title),
        ("ax_tit", ax_by_title),
    ]:
        abs_txt, t = fn(title)
        hit = bool(abs_txt and t and sim(title, t) >= SIM_TH)
        log.debug("  · %-7s | %s | %s", tag, "hit" if hit else "miss", [title])
        if hit:
            row.update(abstract=abs_txt, abstract_src=tag)
            log.info("hit via %s", tag)
            return tag
        time.sleep(SLEEP)

    return "miss"


# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        filename=LOG_FILE,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        level=logging.DEBUG,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    log = logging.getLogger()

    log.info("=== enrichment started ===")

    with IN_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log.info("Loaded %s rows", f"{len(rows):,}")

    already = sum(bool(r.get("abstract")) for r in rows)
    added = 0
    counts = {}

    # checkpoint writer --------------------------------------------------
    def dump(done: int):
        header = list(set(rows[0].keys()) | {"abstract_src"})
        with OUT_CSV.open("w", newline="", encoding="utf-8") as fo:
            wr = csv.DictWriter(fo, fieldnames=header)
            wr.writeheader()
            wr.writerows(rows)
        log.info("checkpoint @ %s / %s", f"{done:,}", f"{len(rows):,}")

    # process ------------------------------------------------------------
    for idx, row in enumerate(tqdm(rows, desc="enrich", unit="ref")):
        tag = enrich(row, log)
        counts[tag] = counts.get(tag, 0) + 1
        if tag not in ("already", "miss", "no_title"):
            added += 1
        if (idx + 1) % SAVE_EVERY == 0:
            dump(idx + 1)

    if "abstract" not in rows[0]:
        for r in rows:
            r.setdefault("abstract", "")
    dump(len(rows))  # final write

    total = len(rows)
    log.info("=== run finished ===")
    log.info("    total rows      : %s", f"{total:,}")
    log.info("    had abstract    : %s", f"{already:,}")
    log.info("    newly enriched  : %s", f"{added:,}")
    log.info("    still missing   : %s", f"{total - already - added:,}")
    log.info(
        "    by source: %s", ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    )


if __name__ == "__main__":
    main()
