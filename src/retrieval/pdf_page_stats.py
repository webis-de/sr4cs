"""Analyze the page counts of a directory of PDFs, log statistics, plot distributions, and export outlier IDs and unreadable files."""

import pathlib, datetime, csv, warnings, logging
from statistics import median, quantiles
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pypdf import PdfReader
from pypdf.errors import PdfReadWarning

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATE = "03_07_25"
PDF_DIR = pathlib.Path("../../slr_ds/data/full_paper_intermediate/pdfs")
OUT_DIR = pathlib.Path("../../slr_ds/data/full_paper_intermediate")

HIST_PNG = OUT_DIR / "pages_hist.png"
BOX_PNG = OUT_DIR / "pages_box.png"
LOG_FILE = f"../../slr_ds/logs/pdf_page_stats_{DATE}.log"


# ── LOGGING ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# ── SUPPRESS pypdf NOISE ───────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=PdfReadWarning)
logging.getLogger("pypdf").setLevel(logging.ERROR)

# ── READ PAGE COUNTS ──────────────────────────────────────────────────────────
records, bad_files = [], []  # (id, pages)

for pdf in tqdm(list(PDF_DIR.glob("*.pdf")), desc="Scanning PDFs", leave=False):
    pid = pdf.stem
    try:
        records.append((pid, len(PdfReader(pdf).pages)))
    except Exception as err:
        bad_files.append((pid, pdf.name, str(err)))

if not records:
    logging.error("No readable PDFs – nothing to analyse.")
    raise SystemExit(1)

page_counts = [p for _, p in records]

# ── TUKEY STATS & OUTLIERS ────────────────────────────────────────────────────
q1, q3 = quantiles(page_counts, n=4)[0], quantiles(page_counts, n=4)[2]
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outliers = [(pid, p) for pid, p in records if p < lower_fence or p > upper_fence]
outlier_ids = [pid for pid, _ in outliers]

logging.info(
    "Files OK=%d | min=%d | Q1=%.1f | median=%.1f | Q3=%.1f | "
    "max=%d | IQR=%.1f | lower_fence=%.1f | upper_fence=%.1f | "
    "Tukey_outliers=%d",
    len(records),
    min(page_counts),
    q1,
    median(page_counts),
    q3,
    max(page_counts),
    iqr,
    lower_fence,
    upper_fence,
    len(outliers),
)
if bad_files:
    logging.warning("Unreadable PDFs: %d (written to CSV)", len(bad_files))

# ── PLOTS (SEABORN) ───────────────────────────────────────────────────────────
sns.set(style="whitegrid", font_scale=1.1)

# histogram
plt.figure(figsize=(8, 5))
sns.histplot(page_counts, bins=range(0, max(page_counts) + 5, 5), kde=False)
plt.title("PDF page-count distribution")
plt.xlabel("pages")
plt.ylabel("frequency")
plt.tight_layout()
plt.savefig(HIST_PNG, dpi=300)
logging.info("Histogram saved to %s", HIST_PNG)

# plain box-plot (vertical)
plt.figure(figsize=(8, 5))
sns.boxplot(y=page_counts, width=0.4, showfliers=False)
plt.ylabel("pages")
plt.title("PDF page-count box-plot")
# add the numbers on the plot like median, Q1, Q3, Min, Max
plt.text(
    0.5,
    median(page_counts),
    f"Median: {median(page_counts):.1f}",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=10,
    color="black",
)
plt.text(
    0.5,
    q1,
    f"Q1: {q1:.1f}",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=10,
    color="blue",
)
plt.text(
    0.5,
    q3,
    f"Q3: {q3:.1f}",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=10,
    color="red",
)
plt.tight_layout()
plt.savefig(BOX_PNG, dpi=300)
logging.info("Box-plot saved to %s", BOX_PNG)

# ── CSV EXPORTS ───────────────────────────────────────────────────────────────
if outlier_ids:
    out_csv = OUT_DIR / f"outlier_ids_{DATE}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("id",), *[(oid,) for oid in outlier_ids]])
    logging.info("Outlier ID list written to %s", out_csv)

if bad_files:
    bad_csv = OUT_DIR / f"bad_pdfs_{DATE}.csv"
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("id", "filename", "error"), *bad_files])
    logging.info("Unreadable list written to %s", bad_csv)
