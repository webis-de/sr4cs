"""Prepare the final set of PDFs and a filtered CSV by removing outliers."""

import pathlib
import shutil
import pandas as pd
import logging
import datetime
import csv

# ── CONFIG ────────────────────────────────────────────────────────────────────
INTER_PDF_DIR = pathlib.Path("../../data/full_paper_intermediate/pdfs")
OUTLIER_LIST = pathlib.Path(
    "../../data/full_paper_intermediate/outlier_ids_03_07_25.csv"
)
OUT_DIR = pathlib.Path("../../data/full_paper_final")
FINAL_PDF_DIR = OUT_DIR / "pdfs"
CSV_INPUT = pathlib.Path(
    "../../data/full_paper_intermediate/"
    "dblp_systematic_review_2000_2025_filtered_link_03_07_25.csv"
)
FINAL_CSV = OUT_DIR / "dblp_systematic_review_2000_2025_filtered_final_03_07_25.csv"
LOGGING_FILE = "../../logs/prepare_final_set_03_07_25.log"

# ── SETUP ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", filename=LOGGING_FILE
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_PDF_DIR.mkdir(parents=True, exist_ok=True)

# ── LOAD OUTLIER IDS ──────────────────────────────────────────────────────────
outliers = pd.read_csv(OUTLIER_LIST, header=0).iloc[:, 0].astype(str).tolist()
outliers_set = set(outliers)
logging.info(f"Loaded {len(outliers_set)} outlier IDs")

# ── COPY NON-OUTLIER PDFs ───────────────────────────────────────────────────────
copied, skipped = 0, 0
for pdf in INTER_PDF_DIR.glob("*.pdf"):
    pid = pdf.stem
    dest = FINAL_PDF_DIR / pdf.name
    if pid in outliers_set:
        skipped += 1
        logging.debug(f"Skipping outlier {pid}")
    else:
        shutil.copy2(pdf, dest)
        copied += 1

logging.info(f"Copied {copied} PDFs to {FINAL_PDF_DIR} (skipped {skipped} outliers)")

# ── BUILD FINAL ID SET ─────────────────────────────────────────────────────────
final_ids = {p.stem for p in FINAL_PDF_DIR.glob("*.pdf")}
logging.info(f"Final PDF set contains {len(final_ids)} files")

# ── LOAD INTERMEDIATE CSV ──────────────────────────────────────────────────────
df = pd.read_csv(CSV_INPUT, dtype={"id": str})
csv_ids = set(df["id"].astype(str))
before = len(df)
logging.info(f"Intermediate CSV has {before} rows")

# ── DETECT MISMATCHES ──────────────────────────────────────────────────────────
today = datetime.date.today().strftime("%Y_%m_%d")

# PDFs with no matching CSV row
extra_pdfs = sorted(final_ids - csv_ids)
if extra_pdfs:
    logging.warning("Found %d PDFs with no CSV row", len(extra_pdfs))
    mismatch_csv = OUT_DIR / f"pdfs_not_in_csv_{today}.csv"
    with open(mismatch_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("id",), *[(pid,) for pid in extra_pdfs]])
    logging.warning("Wrote extra-PDF list to %s", mismatch_csv)

# CSV rows with no PDF file
missing_pdfs = sorted(csv_ids - final_ids)
if missing_pdfs:
    logging.warning("Found %d CSV rows with no PDF file", len(missing_pdfs))
    missing_csv = OUT_DIR / f"csv_rows_no_pdf_{today}.csv"
    with open(missing_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("id",), *[(cid,) for cid in missing_pdfs]])
    logging.warning("Wrote missing-CSV-row list to %s", missing_csv)

# ── FILTER CSV TO FINAL SET ────────────────────────────────────────────────────
df_final = df[df["id"].isin(final_ids)]
after = len(df_final)
dropped = before - after
logging.info(f"Dropped {dropped} rows; {after} remain in final CSV")

# -- SANITY CHECKS ───────────────────────────────────────────────────────────
# Make sure that the list of IDs in the final CSV matches the final PDF set
final_csv_ids = set(df_final["id"].astype(str))
if final_csv_ids != final_ids:
    logging.error("Mismatch between final CSV IDs and final PDF set!")
    logging.error(f"Final CSV IDs: {len(final_csv_ids)}")
    logging.error(f"Final PDF IDs: {len(final_ids)}")
    raise ValueError("Final CSV IDs do not match final PDF set!")

# ── SAVE FINAL CSV ────────────────────────────────────────────────────────────
df_final.to_csv(FINAL_CSV, index=False)
logging.info(f"Wrote filtered CSV to {FINAL_CSV}")
