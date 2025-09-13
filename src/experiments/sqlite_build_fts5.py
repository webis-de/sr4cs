"""Build SQLite FTS5 index from Parquet references (literal only, no stemmer)."""

import sqlite3
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Iterable, Tuple, List

# =========================
# Config
# =========================
DB_PATH = "../../data/final/exp/sqlite/refs.db"
PARQUET_PATH = "../../data/final/refs.parquet"

# Prefix lengths for fast wildcard prefixes (pollution* etc.)
LITERAL_PREFIX_NGRAMS = "2 3 4 5 6 7 8 9 10"
BATCH = 20_000  # insert batch size


# =========================
# Helpers
# =========================
def as_str(x) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (np.integer,)):
        return str(int(x))
    if isinstance(x, float) and float(x).is_integer():
        return str(int(x))
    return str(x)


def batched(iterable: Iterable[Tuple], batch_size: int) -> Iterable[List[Tuple]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def normalize_text(s: str) -> str:
    return s.lower().strip()


def augment_wdg_variants(s: str) -> str:
    """
    Approximate ES word_delimiter_graph with:
      - preserve_original (keep base)
      - split on - / _ into tokens (spaced form)
      - catenate_all (glued form: remove delimiters)
      - split on letter<->digit boundaries in the spaced variant
      - do NOT split on case change (camelCase kept), like ES split_on_case_change=false
    """
    if not s:
        return s

    base = s

    # Replace delimiters with spaces
    spaced = (
        base.replace("-", " ")
        .replace("‐", " ")
        .replace("‒", " ")
        .replace("–", " ")
        .replace("—", " ")
        .replace("/", " ")
        .replace("_", " ")
    )

    # Split between letter<->digit boundaries
    spaced = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", spaced)

    # Glued form (remove delimiters)
    glued = (
        base.replace("-", "")
        .replace("‐", "")
        .replace("‒", "")
        .replace("–", "")
        .replace("—", "")
        .replace("/", "")
        .replace("_", "")
    )

    variants = {base, spaced, glued}
    return "\n".join(v for v in variants if v)


# =========================
# Build index (literal only, ES-like)
# =========================
def build_index(parquet_path: str, db_path: str) -> None:
    df = pd.read_parquet(parquet_path)

    # Required columns
    required_any = {"ref_id", "abstract"}
    missing_any = required_any - set(df.columns)
    if missing_any:
        raise KeyError(f"Missing required columns in Parquet: {missing_any}")

    if "title_norm" not in df.columns:
        if "title" in df.columns:
            df["title_norm"] = df["title"].fillna("").astype(str)
            print("Info: 'title_norm' not found. Using 'title'.")
        else:
            raise KeyError("Missing 'title_norm' and 'title' in Parquet.")

    # Normalize
    df["ref_id"] = df["ref_id"].map(as_str)
    df["title_norm"] = df["title_norm"].fillna("").astype(str).map(normalize_text)
    df["abstract"] = df["abstract"].fillna("").astype(str).map(normalize_text)

    df = df.reset_index(drop=True)

    # Augment tokens
    df["title_aug"] = df["title_norm"].map(augment_wdg_variants)
    df["abstract_aug"] = df["abstract"].map(augment_wdg_variants)

    n_rows = len(df)
    n_unique = df["ref_id"].nunique(dropna=False)
    print(f"Rows to index: {n_rows:,} (unique ref_id: {n_unique:,})")

    # SQLite
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Pragmas
    for pragma in [
        "PRAGMA journal_mode = WAL;",
        "PRAGMA synchronous = NORMAL;",
        "PRAGMA temp_store = MEMORY;",
        "PRAGMA mmap_size = 30000000000;",
        "PRAGMA cache_size = -200000;",
    ]:
        try:
            cur.execute(pragma)
        except Exception as e:
            print(f"Warning: {pragma.strip()} -> {e}")

    # Create literal-only FTS5 table
    cur.execute(
        f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS refs_fts_lit USING fts5(
      ref_id UNINDEXED,
      title,
      abstract,
      tokenize = 'unicode61 remove_diacritics 2',
      prefix = '{LITERAL_PREFIX_NGRAMS}'
    );
    """
    )
    cur.execute("DELETE FROM refs_fts_lit;")
    con.commit()

    print(
        f"Loading refs_fts_lit (unicode61, prefix={LITERAL_PREFIX_NGRAMS}, diacritics-fold)…"
    )
    rows = df[["ref_id", "title_aug", "abstract_aug"]].itertuples(
        index=False, name=None
    )
    cur.execute("BEGIN;")
    for chunk in batched(rows, BATCH):
        cur.executemany(
            "INSERT INTO refs_fts_lit (ref_id, title, abstract) VALUES (?, ?, ?)", chunk
        )
    con.commit()

    try:
        cur.execute("PRAGMA optimize;")
    except Exception as e:
        print(f"Warning: PRAGMA optimize -> {e}")
    con.commit()
    con.close()

    print(
        "Built FTS5 index:\n"
        f" - refs_fts_lit  (unicode61, prefix={LITERAL_PREFIX_NGRAMS}, diacritics-fold, ES-like, no stemmer)\n"
        f"DB → {Path(db_path).resolve()}"
    )


if __name__ == "__main__":
    build_index(PARQUET_PATH, DB_PATH)
