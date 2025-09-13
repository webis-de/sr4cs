"""
Deduplicate references and build mapping table between SLRs and unique references.
"""

import pandas as pd
import unicodedata
import re
from pathlib import Path
import logging

# ─── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path("../../../data/extracted_information/refs")
SRC_CSV = BASE_DIR / "raw/refs_raw.csv"
DEDUP_CSV = BASE_DIR / "processed/references_dedup.csv"
MAP_CSV = BASE_DIR / "processed/slr2ref_map.csv"
LOGGING_FILE = "../../../logs/refs_mapping.log"

# ─── LOGGING ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOGGING_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
    force=True,
)
logging.getLogger().addHandler(logging.StreamHandler())

# ─── 1. LOAD & BASIC CLEANUP ───────────────────────────────────────────────────
df = pd.read_csv(SRC_CSV)
logging.info(f"Loaded {len(df):,} raw references from {SRC_CSV.name}")

# ensure missing are really NaN
df = df.replace({"": pd.NA, " ": pd.NA})

# keep only rows that have either a title OR a DOI, and whose raw string is “long enough”
MIN_RAW_LEN = 30
mask_valid = (df["doi"].notna() | df["title"].notna()) & (
    df["raw"].str.len() >= MIN_RAW_LEN
)

df_valid = df.loc[mask_valid].copy()
logging.info(f"Filtered to {len(df_valid):,} valid references")


# ─── 2. NORMALISE FIELDS FOR DEDUPING ──────────────────────────────────────────
def normalise_title(t: str | float) -> str:
    if pd.isna(t):
        return ""
    t = unicodedata.normalize("NFKD", str(t)).lower()  # normalize unicode
    t = re.sub(r"[^a-z0-9]+", " ", t)  # keep alphanum only
    return re.sub(r"\s+", " ", t).strip()


df_valid["title_norm"] = df_valid["title"].apply(normalise_title)
logging.info("Normalised titles for deduplication")

# ─── 3. DEDUPLICATE ────────────────────────────────────────────────────────────
# priority order:
#   1) DOI   (most reliable)
#   2) normalised title
dedup_cols = ["doi", "title_norm"]

df_unique = (
    df_valid.sort_values(dedup_cols)  # deterministic
    .drop_duplicates(subset=dedup_cols, keep="first")
    .reset_index(drop=True)
)

df_unique.insert(0, "ref_id", range(1, len(df_unique) + 1))
logging.info(f"Deduplicated to {len(df_unique):,} unique references")

# ─── 4. BUILD MAPPING TABLE ────────────────────────────────────────────────────
# Merge back to get ref_id for every (id, reference) pair
df_map = (
    df_valid.merge(df_unique[["ref_id"] + dedup_cols], on=dedup_cols, how="left")
    .loc[:, ["id", "ref_id"]]
    .sort_values(["id", "ref_id"])
    .reset_index(drop=True)
)

logging.info(f"Created mapping table with {len(df_map):,} SLR ↔ reference links")
# ─── 5. SAVE ───────────────────────────────────────────────────────────────────
DEDUP_CSV.write_text("")
MAP_CSV.write_text("")

df_unique.to_csv(DEDUP_CSV, index=False)
df_map.to_csv(MAP_CSV, index=False)

logging.info(f"Saved deduplicated references to {DEDUP_CSV.name}")
logging.info(f"Saved SLR ↔ reference mapping to {MAP_CSV.name}")
logging.info("Done processing references")
