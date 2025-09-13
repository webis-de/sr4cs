"""Script to count the number of pages in each PDF file in a directory and output the results to a text file."""

import os
import sys
from PyPDF2 import PdfReader


def count_pages(path):
    reader = PdfReader(path)
    return len(reader.pages)


def main(directory, out_path):
    files = []
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.lower().endswith(".pdf"):
            try:
                pages = count_pages(entry.path)
            except Exception as e:
                # skip unreadable files
                continue
            files.append((entry.name, pages))

    if not files:
        print("No PDF files found.")
        return

    files.sort(key=lambda x: x[1])
    with open(out_path, "w", encoding="utf-8") as f:
        for name, pages in files:
            f.write(f"{name}: {pages} pages\n")

    print(f"Wrote {len(files)} entries to {out_path}")


if __name__ == "__main__":
    dir_path = "../../data/full_paper_final/pdfs"
    out_file = "../../data/full_paper_final/pdf_page_counts.txt"
    main(dir_path, out_file)
