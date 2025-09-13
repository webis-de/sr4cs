import json
import shutil
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# =========================
# CONFIG — edit these paths
# =========================
DF_FILE = Path("../../data/final/slr_dataset_final.parquet")  # .parquet or .csv
DF_ID_COLUMN = "id"  # column name containing IDs
JSON_DIR = Path("../../data/final/nanonet")  # contains files like 123.json
OUTPUT_COMBINED_JSON = Path("../../data/final/nanonet/combined.json")
OUTPUT_COMBINED_JSONL = Path("../../data/final/nanonet/combined.jsonl")

# Safety: move deleted files into TRASH_DIR instead of permanent deletion.
SAFE_DELETE_TO_TRASH = True
TRASH_DIR = JSON_DIR / "_trash_removed_jsons"

# If your IDs in the dataframe are numeric but filenames are strings, keep this True.
COERCE_IDS_TO_STR = True


def load_dataframe_ids(df_path: Path, id_col: str, as_str: bool = True) -> set:
    if not df_path.exists():
        raise FileNotFoundError(f"Dataframe file not found: {df_path}")

    if df_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(df_path)
    elif df_path.suffix.lower() == ".csv":
        df = pd.read_csv(df_path)
    else:
        raise ValueError("Unsupported dataframe format. Use .parquet or .csv")

    if id_col not in df.columns:
        raise KeyError(
            f"Column '{id_col}' not found in dataframe. Columns: {list(df.columns)}"
        )

    ids = df[id_col].dropna().unique().tolist()
    if as_str:
        # Normalize to plain strings without surrounding spaces
        ids = [str(x).strip() for x in ids]
    return set(ids)


def list_json_files(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"JSON directory not found: {directory}")
    return sorted([p for p in directory.glob("*.json") if p.is_file()])


def safe_remove(file_path: Path, use_trash: bool, trash_dir: Path) -> None:
    if use_trash:
        trash_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), trash_dir / file_path.name)
    else:
        file_path.unlink(missing_ok=True)


def prune_jsons(
    json_dir: Path, allowed_ids: set, use_trash: bool, trash_dir: Path
) -> Dict[str, int]:
    files = list_json_files(json_dir)
    removed = 0
    kept = 0
    for f in files:
        stem = f.stem.strip()  # filename without .json
        if stem not in allowed_ids:
            safe_remove(f, use_trash, trash_dir)
            removed += 1
        else:
            kept += 1
    return {"removed": removed, "kept": kept, "total": removed + kept}


def load_all_jsons(json_dir: Path, allowed_ids: set) -> List[Dict[str, Any]]:
    combined = []
    files = list_json_files(json_dir)
    bad = 0
    for f in files:
        stem = f.stem.strip()
        # Only combine files whose stem is in allowed_ids
        if stem not in allowed_ids:
            continue
        try:
            with f.open("r", encoding="utf-8") as fh:
                obj = json.load(fh)
                combined.append(obj)
        except Exception:
            bad += 1
            # Skip corrupted JSON; continue
    if bad:
        print(f"[WARN] Skipped {bad} corrupted/invalid JSON files.")
    return combined


def write_outputs(objs: List[Dict[str, Any]], out_json: Path, out_jsonl: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    # Standard JSON array
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(objs, f, ensure_ascii=False, indent=2)
    # JSONL for streaming pipelines
    with out_jsonl.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def main():
    allowed_ids = load_dataframe_ids(DF_FILE, DF_ID_COLUMN, as_str=COERCE_IDS_TO_STR)
    print(f"[INFO] Loaded {len(allowed_ids)} unique IDs from dataframe.")

    # Optional: quick peek at directory population
    total_before = len(list_json_files(JSON_DIR))
    print(f"[INFO] JSON files in directory before prune: {total_before}")

    stats = prune_jsons(JSON_DIR, allowed_ids, SAFE_DELETE_TO_TRASH, TRASH_DIR)
    print(
        f"[INFO] Prune done: kept={stats['kept']}, removed={stats['removed']}, total_seen={stats['total']}"
    )

    # Combine remaining JSONs
    combined = load_all_jsons(JSON_DIR, allowed_ids)
    print(f"[INFO] Loaded {len(combined)} JSON objects for combination.")

    write_outputs(combined, OUTPUT_COMBINED_JSON, OUTPUT_COMBINED_JSONL)
    print(f"[OK] Wrote: {OUTPUT_COMBINED_JSON}")
    print(f"[OK] Wrote: {OUTPUT_COMBINED_JSONL}")


if __name__ == "__main__":
    main()
