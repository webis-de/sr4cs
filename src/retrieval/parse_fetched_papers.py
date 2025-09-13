"""
Load a JSON of fetched paper metadata and emit a CSV with the key fields,
including authors parsed from `z_authors` entries (using raw_author_name).
"""

import json
import csv
import os

# ───────────────────────── CONFIGURATION ─────────────────────────
INPUT_FILE = "../../data/full_paper_intermediate/dblp_systematic_review_2000_2025_filtered_03_07_25_fetched.json"
OUTPUT_FILE = "../../data/full_paper_intermediate/dblp_systematic_review_2000_2025_filtered_03_07_25_parsed.csv"
# ───────────────────────────────────────────────────────────────────


def process_entry(entry: dict) -> dict:
    """Extract fields and normalize authors from the entry."""
    # Basic metadata
    id = entry.get("id") or "None"
    doi = entry.get("doi") or "None"
    doi_url = entry.get("doi_url") or "None"
    title = entry.get("title") or "None"
    genre = entry.get("genre") or "None"
    year = str(entry.get("year")) if entry.get("year") is not None else "None"
    journal_name = entry.get("journal_name") or "None"
    journal_issns = entry.get("journal_issns") or "None"
    publisher = entry.get("publisher") or "None"
    is_oa = str(entry.get("is_oa")) if entry.get("is_oa") is not None else "None"

    # OA URLs
    oa_locs = entry.get("oa_locations", [])
    if not oa_locs:
        oa_urls = ["None"]
        oa_pdfs = ["None"]
        oa_landings = ["None"]
    else:
        oa_urls = [loc.get("url") or "None" for loc in oa_locs]
        oa_pdfs = [loc.get("url_for_pdf") or "None" for loc in oa_locs]
        oa_landings = [loc.get("url_for_landing_page") or "None" for loc in oa_locs]

    # Authors
    z_authors = entry.get("z_authors", [])
    if not z_authors:
        names = ["None"]
        positions = ["None"]
        affiliations = ["None"]
    else:
        names = []
        positions = []
        affiliations = []
        for auth in z_authors:
            raw_name = auth.get("raw_author_name") or "None"
            names.append(raw_name)

            pos = auth.get("author_position") or "None"
            positions.append(pos)

            raw_affs = auth.get("raw_affiliation_strings", [])
            if raw_affs:
                # join multiple affiliations with ","
                affil_str = ",".join(raw_affs)
            else:
                affil_str = "None"
            affiliations.append(affil_str)

    # Join lists into semicolon-separated strings
    def join_or_none(lst):
        return ";".join(lst) if lst else "None"

    return {
        "doi": doi,
        "doi_url": doi_url,
        "title": title,
        "genre": genre,
        "year": year,
        "journal_name": journal_name,
        "journal_issns": journal_issns,
        "publisher": publisher,
        "is_oa": is_oa,
        "oa_urls": join_or_none(oa_urls),
        "oa_urls_for_pdf": join_or_none(oa_pdfs),
        "oa_urls_for_landing_page": join_or_none(oa_landings),
        "author_names": join_or_none(names),
        "author_positions": join_or_none(positions),
        "author_affiliations": join_or_none(affiliations),
    }


def main():
    # load JSON
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        papers = json.load(f)

    # transform
    rows = [process_entry(p) for p in papers]

    # write CSV
    fieldnames = [
        "doi",
        "doi_url",
        "title",
        "genre",
        "year",
        "journal_name",
        "journal_issns",
        "publisher",
        "is_oa",
        "oa_urls",
        "oa_urls_for_pdf",
        "oa_urls_for_landing_page",
        "author_names",
        "author_positions",
        "author_affiliations",
    ]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Processed {len(rows)} entries.")
    print(f"CSV saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
