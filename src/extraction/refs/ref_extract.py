"""
High-recall reference extraction pipeline: Grobid → AnyStyle → (optional) Crossref
enrichment.
"""

from pathlib import Path
import json, csv, tempfile, subprocess, requests, logging, time, re, sys
from bs4 import BeautifulSoup
import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
USE_ANYSTYLE = True  # False → skip AnyStyle
CROSSREF_ENRICH = False  # False → skip Crossref calls
GROBID_TIMEOUT = 600  # seconds per PDF
GROBID_RETRIES = 3
SAVE_EVERY = 5  # flush every N new PDFs

BASE_DIR = Path("../../../data/full_paper_final/pdfs")
OUT_DIR = Path("../../../data/extracted_information/refs")
LOG_FILE = Path("../../../logs/ref_extract_03_07_25.log")

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
CROSSREF_API = "https://api.crossref.org/works"
COURTESY_EMAIL = "pieer.achkar@imw.fraunhofer.de"

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUT_DIR / "raw/refs_raw.json"
OUTPUT_CSV = OUT_DIR / "raw/refs_raw.csv"
# ────────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
    force=True,
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

doi_re = re.compile(r"\b10\.\d{4,9}/\S+\b", re.I)


# ─── HELPERS ────────────────────────────────────────────────────────────────────
def clean_abstract(jats_xml: str | None) -> str | None:
    if not jats_xml:
        return None
    return BeautifulSoup(jats_xml, "lxml").get_text(" ", strip=True)


def write_csv_from_json(json_list: list[dict], path: Path) -> None:
    """Dump the current cache to CSV – every unique key becomes a column."""
    rows: list[dict] = []
    for entry in json_list:
        pid = entry["id"]
        for ref in entry["references"]:
            rows.append({"id": pid, **ref})
    header = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, header)
        w.writeheader()
        w.writerows(rows)


def flush(json_cache: list[dict]) -> None:
    OUTPUT_JSON.write_text(json.dumps(json_cache, indent=2), "utf-8")
    write_csv_from_json(json_cache, OUTPUT_CSV)
    logging.info(f"Checkpoint saved  –  {len(json_cache)} PDFs processed so far")


# ─── PIPELINE PRIMITIVES ────────────────────────────────────────────────────────
def grobid_raw_refs(pdf: Path) -> list[str]:
    """Extract raw reference strings from a PDF using Grobid."""
    for attempt in range(1, GROBID_RETRIES + 1):
        try:
            logging.info(f"Grobid ↑  {pdf.name}  (try {attempt})")
            with pdf.open("rb") as fh:
                r = requests.post(
                    GROBID_URL,
                    files={"input": (pdf.name, fh)},
                    data={"includeRawCitations": "1", "consolidateCitations": "0"},
                    timeout=GROBID_TIMEOUT,
                )
            r.raise_for_status()
            break
        except requests.exceptions.ReadTimeout:
            logging.warning(f"{pdf.name}: timeout ({GROBID_TIMEOUT}s) – try {attempt}")
            if attempt == GROBID_RETRIES:
                logging.error(f"{pdf.name}: skipped after {attempt} timeouts")
                return []
            time.sleep(5)

    soup = BeautifulSoup(r.text, "lxml-xml")
    ref_div = soup.find("div", {"type": "references"})
    if not ref_div:
        logging.warning(f"{pdf.name}: no <div type='references'> found")
        return []

    refs = [n.text.strip() for n in ref_div.find_all("note", {"type": "raw_reference"})]
    logging.info(f"Grobid ↓  {pdf.name}: {len(refs)} refs")
    return refs


def anystyle_parse(refs: list[str], tag: str) -> list[dict]:
    """Run AnyStyle over a list of reference strings – returns list[dict]."""
    if not refs:
        return []
    logging.info(f"AnyStyle ↑  {tag}  ({len(refs)} refs)")
    with tempfile.TemporaryDirectory() as tmp:
        txt = Path(tmp) / "refs.txt"
        txt.write_text("\n".join(refs), "utf-8")
        proc = subprocess.run(
            ["anystyle", "--stdout", "-f", "json", "parse", str(txt)],
            check=True,
            capture_output=True,
            text=True,
        )
    parsed = json.loads(proc.stdout)
    logging.info(f"AnyStyle ↓  {tag}: {len(parsed)} parsed")
    return parsed


def crossref_lookup(ref_str: str) -> dict:
    """Return cleaned Crossref metadata or {}."""
    headers = {"User-Agent": f"ref-extractor/0.1 (mailto:{COURTESY_EMAIL})"}
    doi_match = doi_re.search(ref_str)

    try:
        if doi_match:  # 1) direct DOI lookup
            data = requests.get(
                f"{CROSSREF_API}/{doi_match.group()}", headers=headers, timeout=15
            ).json()["message"]
        else:  # 2) free-text search
            params = {"query.bibliographic": ref_str, "rows": 1}
            items = requests.get(
                CROSSREF_API, headers=headers, params=params, timeout=15
            ).json()["message"]["items"]
            if not items:
                return {}
            data = items[0]
    except Exception:
        return {}

    if isinstance(data, list) and data:
        data = data[0]

    return {
        "doi": data.get("DOI"),
        "title": (data.get("title") or [""])[0],
        "author": "; ".join(
            f"{a.get('family','')} {a.get('given','')}".strip()
            for a in data.get("author", [])
        ),
        "year": (data.get("issued", {}).get("date-parts", [[None]])[0][0]),
        "journal": (data.get("container-title") or [""])[0],
        "abstract": clean_abstract(data.get("abstract")),
        "match_score": data.get("score"),
    }


# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # resume from previous run (if any)
    if OUTPUT_JSON.exists():
        json_cache = json.loads(OUTPUT_JSON.read_text("utf-8"))
        processed = {e["id"] for e in json_cache}
        logging.info(f"Loaded checkpoint – {len(processed)} PDFs already done")
    else:
        json_cache, processed = [], set()

    pdfs = list(BASE_DIR.glob("*.pdf"))
    new_counter = 0

    for pdf in tqdm.tqdm(pdfs, desc="PDFs", unit="file"):
        pid = pdf.stem
        if pid in processed:
            continue

        # 1) raw reference strings
        raw_refs = grobid_raw_refs(pdf)

        # 2) parsed metadata  (+ keep the raw string)
        if USE_ANYSTYLE:
            parsed = anystyle_parse(raw_refs, pdf.name)
            refs_struct = [{"raw": raw, **p} for raw, p in zip(raw_refs, parsed)]
        else:
            refs_struct = [{"raw": raw} for raw in raw_refs]

        # 3) optional Crossref enrichment
        if CROSSREF_ENRICH and refs_struct:
            for obj in refs_struct:
                meta = crossref_lookup(obj["raw"])
                if meta:
                    obj.update(meta)
                time.sleep(0.25)  # be polite to Crossref

        # 4) store in cache
        json_cache.append({"id": pid, "references": refs_struct})
        processed.add(pid)
        new_counter += 1

        if new_counter % SAVE_EVERY == 0:
            flush(json_cache)

    flush(json_cache)  # final flush
    logging.info(
        f"All done – {len(processed)} PDFs, "
        f"{sum(len(e['references']) for e in json_cache)} refs extracted"
    )


if __name__ == "__main__":
    main()
