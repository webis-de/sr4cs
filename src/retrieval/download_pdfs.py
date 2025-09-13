"""Script to download PDFs from a list of URLs in a CSV file and log the results."""

import pathlib, logging, time, datetime, requests, pandas as pd
from urllib.parse import urlparse
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_INPUT = "../../data/full_paper_intermediate/dblp_systematic_review_2000_2025_filtered_link_03_07_25.csv"
PDF_DIR = pathlib.Path("../../data/full_paper_intermediate/pdfs")
LOG_FILE = "../../logs/download_pdfs_03_07_25.log"

PDF_DIR.mkdir(parents=True, exist_ok=True)

# ── LOGGING ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
    force=True,
)
logging.info("=== STARTING PDF download run ===")

# ── SESSION WITH DEFAULT HEADERS ───────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream;q=0.8,*/*;q=0.5",
    }
)


# ── DOWNLOAD ROUTINE ───────────────────────────────────────────────────────────
def download_file(url: str, dest: pathlib.Path):
    # fix stale IEEE staging URLs
    if "xplorestaging.ieee.org" in url:
        url = url.replace("xplorestaging", "ieeexplore")

    # special handling for MDPI
    is_mdpi = "mdpi.com" in urlparse(url).netloc

    def _try(u, extra_headers=None):
        hdrs = SESSION.headers.copy()
        if extra_headers:
            hdrs.update(extra_headers)
        return SESSION.get(
            u, stream=True, timeout=60, headers=hdrs, allow_redirects=True
        )

    r = _try(url, {"Referer": url.split("/pdf")[0]} if is_mdpi else None)

    # MDPI plan B: strip ?version=… and retry on 403
    if is_mdpi and r.status_code == 403 and "?" in url:
        r.close()
        clean_url = url.split("?", 1)[0]
        r = _try(clean_url, {"Referer": clean_url.split("/pdf")[0]})

    # generic 406 retry with stricter Accept
    if r.status_code == 406:
        r.close()
        r = _try(url, {"Accept": "application/pdf"})

    r.raise_for_status()
    if r.headers.get("content-type", "").startswith("text/html"):
        raise requests.HTTPError("HTML page received (likely paywalled)")

    with open(dest, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)


# ── MAIN ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_INPUT)
if "id" not in df.columns:
    df.insert(0, "id", df.index.astype(str))
logging.info(f"Loaded {len(df)} rows – using column 'id' for filenames")

# Pre-count existing files
existing = {p.stem for p in PDF_DIR.glob("*.pdf")}
logging.info(f"{len(existing)} PDFs already present and will be skipped")

missing_rows = []  # collect failures for CSV

for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading PDFs"):
    row_id = str(row["id"])
    pdf_link = row.get("pdf_link")

    if not isinstance(pdf_link, str) or not pdf_link:
        logging.error(f"{row_id} – no_link")
        missing_rows.append({"id": row_id, "pdf_link": pdf_link})
        continue

    outpath = PDF_DIR / f"{row_id}.pdf"
    if row_id in existing or outpath.exists():
        continue  # already downloaded earlier

    try:
        download_file(pdf_link, outpath)
        logging.info(f"{row_id} – downloaded")
        time.sleep(1)  # politeness
    except Exception as e:
        logging.error(f"{row_id} – download_error: {e}")
        missing_rows.append({"id": row_id, "pdf_link": pdf_link})

# ── WRITE CSV OF FAILURES ──────────────────────────────────────────────────────
if missing_rows:
    miss_df = pd.DataFrame(missing_rows, columns=["id", "pdf_link"])
    stamp = datetime.date.today().strftime("%Y_%m_%d")
    miss_csv = PDF_DIR.parent / f"missing_pdfs_{stamp}.csv"
    miss_df.to_csv(miss_csv, index=False)
    logging.info(f"Missing list written to {miss_csv} ({len(miss_df)} entries)")
else:
    logging.info("All PDFs downloaded successfully – no missing list generated")

logging.info("=== PDF download run complete ===")
print("done.")
