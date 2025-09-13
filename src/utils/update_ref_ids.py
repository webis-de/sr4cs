"""
Script to update 'ref_id' lists in a JSON dataset based on valid IDs from a Parquet file.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Optional

import pandas as pd

# ===================== CONFIG =====================
INPUT_JSON_PATH = Path("../..//data/final/cleaned/slr_dataset_final.json")
REFS_PARQUET_PATH = Path("../..//data/final/refs.parquet")
OUTPUT_JSON_PATH = Path("../..//data/final/cleaned/slr_dataset_final.json")
WRITE_REMOVALS_LOG = True
REMOVALS_LOG_PATH = Path("removed_ref_ids_report.jsonl")
# ==================================================


def _norm_id(x: Any) -> Optional[int]:
    """Coerce to int if possible; else None."""
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def load_valid_reference_ids(parquet_path: Path) -> Set[int]:
    """
    Load a set of valid reference IDs from the parquet.
    Prefer 'id' column; fallback to 'ref_id'.
    If both exist, use 'id'.
    """
    df = pd.read_parquet(parquet_path)
    cols = set(df.columns.str.lower())

    chosen_col = None
    if "ref_id" in cols:
        chosen_col = [c for c in df.columns if c.lower() == "ref_id"][0]
        print("Column 'ref_id' found in refs.parquet.")
    else:
        raise ValueError(
            f"refs.parquet must have an 'id' or 'ref_id' column. Found: {list(df.columns)}"
        )

    valid = set()
    for v in df[chosen_col].tolist():
        iv = _norm_id(v)
        if iv is not None:
            valid.add(iv)

    if not valid:
        raise ValueError(f"No valid integer IDs found in column '{chosen_col}'.")

    print(
        f"Using '{chosen_col}' from refs.parquet as the reference ID column. "
        f"Unique IDs loaded: {len(valid)}"
    )
    return valid


def clean_json_refs(json_path: Path, valid_ids: Set[int]) -> Dict[str, Any]:
    """
    Load JSON, filter each object's ref_id list by 'valid_ids',
    update num_refs, and return {'data': new_list, 'removals': log}.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    removals_log: List[Dict[str, Any]] = []
    updated = 0
    total_removed = 0

    for obj in data:
        rid = str(obj.get("id", "")).strip()

        # Only proceed if there is a ref_id list
        ref_list = obj.get("ref_id", None)
        if not isinstance(ref_list, list):
            # Nothing to clean
            continue

        # Coerce all entries to ints if possible; keep original order
        coerced: List[int] = []
        for x in ref_list:
            iv = _norm_id(x)
            if iv is not None:
                coerced.append(iv)

        # Filter by valid set
        cleaned: List[int] = [x for x in coerced if x in valid_ids]

        # Count removals
        removed_ids = [x for x in coerced if x not in valid_ids]
        if removed_ids:
            removals_log.append(
                {
                    "sr_id": rid,
                    "removed_count": len(removed_ids),
                    "removed_ids": removed_ids,
                }
            )
            total_removed += len(removed_ids)

        # Update object only if changed (but updating num_refs anyway is fine)
        obj["ref_id"] = cleaned
        obj["num_refs"] = len(cleaned)
        updated += 1

    return {
        "data": data,
        "removals": removals_log,
        "updated_count": updated,
        "total_removed": total_removed,
    }


def main():
    valid_ids = load_valid_reference_ids(REFS_PARQUET_PATH)
    result = clean_json_refs(INPUT_JSON_PATH, valid_ids)

    # Write cleaned JSON
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(result["data"], f, ensure_ascii=False, indent=2)

    # Optional: write removals log (one JSON object per line)
    if WRITE_REMOVALS_LOG:
        with REMOVALS_LOG_PATH.open("w", encoding="utf-8") as f:
            for row in result["removals"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"Done. SRs processed: {len(result['data'])} | "
        f"Lists updated: {result['updated_count']} | "
        f"Total ref_ids removed: {result['total_removed']}"
    )
    print(f"Wrote cleaned JSON: {OUTPUT_JSON_PATH.resolve()}")
    if WRITE_REMOVALS_LOG:
        print(f"Wrote removals log: {REMOVALS_LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()
