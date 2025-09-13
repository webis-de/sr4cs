"""Script to delete specific entries from JSON and Parquet dataset files based on a list of IDs."""

import json
import pandas as pd
from pathlib import Path

# ========= CONFIG =========
JSON_FILES = [
    "../../data/final/slr_dataset_final.json",
    "../../data/final/slr_dataset_with_sqlite.json",
]
PARQUET_FILE = "../../data/final/slr_dataset_final.parquet"
OUTPUT_DIR = "../../data/final/cleaned"
DROP_IDS = {
    "1920346",
    "276688",
    "2778589",
    "310787",
    "3222481",
    "3513691",
    "3928218",
    "4241710",
    "920289",
}
# ==========================

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ---- Clean JSON files ----
for json_path in JSON_FILES:
    with open(json_path, "r") as f:
        data = json.load(f)

    cleaned = [entry for entry in data if str(entry["id"]) not in DROP_IDS]

    out_path = Path(OUTPUT_DIR) / Path(json_path).name
    with open(out_path, "w") as f:
        json.dump(cleaned, f, indent=2)

    print(f"Cleaned {json_path} → {out_path} (kept {len(cleaned)} entries)")

# ---- Clean Parquet file ----
df = pd.read_parquet(PARQUET_FILE)

df_cleaned = df[~df["id"].astype(str).isin(DROP_IDS)]

out_parquet = Path(OUTPUT_DIR) / Path(PARQUET_FILE).name
df_cleaned.to_parquet(out_parquet, index=False)

print(f"Cleaned {PARQUET_FILE} → {out_parquet} (kept {len(df_cleaned)} rows)")
