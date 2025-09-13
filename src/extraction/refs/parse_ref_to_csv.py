"""
Parse extracted references from JSON files and combine them into a single CSV file.
Each reference is assigned a unique ref_id, and the corresponding slr_id is extracted from the file name.
"""

import os
import json
import csv
from pathlib import Path

JSON_DIR = Path("../../../data/extracted_information/references")
CSV_OUTPUT_FILE = JSON_DIR / "combined_references.csv"


def extract_slr_id(file_name):
    """
    Extract the slr_id from the file name.
    Expected format: "<slr_id>_..._references.json"
    """
    # Remove the extension and split by underscore.
    stem = Path(file_name).stem
    return stem.split("_")[0]


def build_csv_rows(json_dir):
    all_rows = []
    ref_counter = 1
    # List all JSON files in the directory
    json_files = list(json_dir.glob("*_references.json"))

    for json_file in json_files:
        slr_id = extract_slr_id(json_file.name)
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                references = json.load(f)
            except Exception as e:
                print(f"Could not load {json_file}: {e}")
                continue

        # Process each reference
        for ref in references:
            # Build authors as a semicolon separated list if available
            authors_list = ref.get("authors", [])
            authors_formatted = "; ".join(
                " ".join(
                    filter(None, [a.get("firstname", ""), a.get("lastname", "")])
                ).strip()
                for a in authors_list
                if a
            )

            row = {
                "ref_id": ref_counter,
                "slr_id": slr_id,
                "doi": ref.get("doi", ""),
                "title": ref.get("title", ""),
                "year": ref.get("year", ""),
                "journal": ref.get("journal", ""),
                "authors": authors_formatted,
            }
            all_rows.append(row)
            ref_counter += 1
    return all_rows


def save_csv(rows, output_file):
    fieldnames = ["ref_id", "slr_id", "doi", "title", "year", "journal", "authors"]
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Combined CSV saved to {output_file}")


def main():
    rows = build_csv_rows(JSON_DIR)
    save_csv(rows, CSV_OUTPUT_FILE)
    print(f"Processed {len(rows)} references.")


if __name__ == "__main__":
    main()
