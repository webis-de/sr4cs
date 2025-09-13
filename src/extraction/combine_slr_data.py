"""
Combine multiple JSON files containing SLR search data into a single Parquet dataset.
Each JSON file is validated and normalized to ensure consistent schema.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =========================
# Config (edit as needed)
# =========================
INPUT_DIR = Path("../../data_tmp/extracted_information/search_data/nanonet")
OUTPUT_PATH = Path(
    "../../data_tmp/extracted_information/search_data/nanonet/slr_dataset.parquet"
)

# =========================
# Helpers
# =========================

LIST_FIELDS = [
    "databases",
    "search_strings_boolean",
    "inclusion_criteria",
    "exclusion_criteria",
    "research_questions",
    "language_restrictions",
]

STR_FIELDS = [
    "topic",
    "objective",
]

BOOL_FIELDS = [
    "snowballing",
]

REQUIRED_FIELDS = [
    "id",
    "databases",
    "search_strings_boolean",
    "year_range",
    "language_restrictions",
    "inclusion_criteria",
    "exclusion_criteria",
    "topic",
    "objective",
    "research_questions",
    "snowballing",
]


def to_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        # Sometimes "id" might be "00123" or float 123.0 -> still fine
        try:
            return int(float(v))
        except Exception:
            return None


def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v in (None, "", "null"):
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        sv = v.strip().lower()
        if sv in {"true", "yes", "y", "1"}:
            return True
        if sv in {"false", "no", "n", "0"}:
            return False
    # Fallback: best effort cast
    return bool(v)


def to_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def to_list_of_str(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return ["" if x is None else str(x) for x in v]
    # If a scalar sneaks in, wrap it
    return [str(v)]


def normalize_year_range(v: Any) -> str:
    # Ensure consistent string type. If list, join with "; "
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "; ".join(str(x) for x in v)
    return str(v)


def load_jsons(input_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    files = sorted(input_dir.glob("*.json"))
    out: List[Tuple[Path, Dict[str, Any]]] = []
    for f in files:
        try:
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            out.append((f, data))
        except Exception as e:
            print(f"[ERROR] Failed to read {f}: {e}")
    return out


def validate_and_normalize(
    raw: Dict[str, Any], src: Path
) -> Tuple[Dict[str, Any], List[str]]:
    issues: List[str] = []
    rec: Dict[str, Any] = {}

    # ID
    rec["id"] = to_int(raw.get("id"))
    if rec["id"] is None:
        issues.append("id is missing or not int-castable")

    # List fields
    for k in LIST_FIELDS:
        rec[k] = to_list_of_str(raw.get(k))
        # sanity: lists but not strings inside?
        if not isinstance(rec[k], list):
            issues.append(f"{k} not normalized to list")
        else:
            # ensure all str
            if not all(isinstance(x, str) for x in rec[k]):
                issues.append(f"{k} contains non-string after normalization")

    # String fields
    for k in STR_FIELDS:
        rec[k] = to_str(raw.get(k))

    # year_range
    rec["year_range"] = normalize_year_range(raw.get("year_range"))
    if not isinstance(rec["year_range"], str):
        issues.append("year_range not string after normalization")

    # Boolean fields
    for k in BOOL_FIELDS:
        rec[k] = to_bool(raw.get(k))

    # Missing required checks
    for k in REQUIRED_FIELDS:
        if k not in rec:
            issues.append(f"missing required field: {k}")

    # Optional: flag obviously empty search strings
    if len(rec["search_strings_boolean"]) == 0:
        issues.append("search_strings_boolean is empty")

    # Optional: languages empty
    if len(rec["language_restrictions"]) == 0:
        issues.append("language_restrictions is empty")

    # Attach source path for debugging if needed
    rec["_source_file"] = str(src)

    return rec, issues


def dataframe_with_schema(records: List[Dict[str, Any]]) -> pa.Table:
    df = pd.DataFrame(records)

    # Enforce pandas dtypes
    df["id"] = df["id"].astype("int64", errors="raise")
    df["topic"] = df["topic"].astype("string")
    df["objective"] = df["objective"].astype("string")
    df["year_range"] = df["year_range"].astype("string")
    df["snowballing"] = df["snowballing"].astype("bool")

    # Ensure list columns are actually lists (not numpy arrays)
    for col in LIST_FIELDS:
        df[col] = df[col].apply(
            lambda x: list(x) if isinstance(x, (list, tuple)) else []
        )

    # Arrow schema: keep lists-of-string
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("databases", pa.list_(pa.string())),
            pa.field("search_strings_boolean", pa.list_(pa.string())),
            pa.field("year_range", pa.string()),
            pa.field("language_restrictions", pa.list_(pa.string())),
            pa.field("inclusion_criteria", pa.list_(pa.string())),
            pa.field("exclusion_criteria", pa.list_(pa.string())),
            pa.field("topic", pa.string()),
            pa.field("objective", pa.string()),
            pa.field("research_questions", pa.list_(pa.string())),
            pa.field("snowballing", pa.bool_()),
            pa.field("_source_file", pa.string()),
        ]
    )

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=True)
    return table


# =========================
# Main
# =========================

if __name__ == "__main__":
    print(f"[INFO] Scanning: {INPUT_DIR.resolve()}")
    loaded = load_jsons(INPUT_DIR)
    print(f"[INFO] Found {len(loaded)} JSON files")

    normalized: List[Dict[str, Any]] = []
    all_issues: List[Tuple[str, List[str]]] = []

    for src, raw in loaded:
        rec, issues = validate_and_normalize(raw, src)
        normalized.append(rec)
        if issues:
            all_issues.append((str(src), issues))

    # Build Arrow table
    table = dataframe_with_schema(normalized)

    # Quick stats
    df_stats = table.to_pandas()
    non_empty_queries = (df_stats["search_strings_boolean"].apply(len) > 0).sum()
    total = len(df_stats)

    print("----- Schema & Stats -----")
    print(df_stats.dtypes)
    print(
        f"Rows with non-empty search_strings_boolean: {non_empty_queries} out of {total}"
    )

    # Issue summary (compact)
    if all_issues:
        print("----- Validation issues (first 20 files) -----")
        for i, (src, issues) in enumerate(all_issues[:20], 1):
            joined = "; ".join(issues)
            print(f"{i:02d}. {src}: {joined}")
        if len(all_issues) > 20:
            print(f"... and {len(all_issues) - 20} more files with issues")

    # Ensure output dir exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write Parquet
    pq.write_table(table, OUTPUT_PATH)
    print(f"[OK] Wrote Parquet -> {OUTPUT_PATH.resolve()}")
