"""Script to sample PDFs along with their corresponding OCR and smoldocling files for evaluation purposes."""

import os
import random
import shutil


def sample_draws(
    full_pdf_dir, rolm_ocr_dir, smoldocling_dir, output_dir, draws=5, sample_size=20
):
    # Gather PDF bases that have both .txt and .md parsed files
    pdf_files = [f for f in os.listdir(full_pdf_dir) if f.lower().endswith(".pdf")]
    bases = [os.path.splitext(f)[0] for f in pdf_files]

    # Filter to those with both OCR and smoldocling files
    valid_bases = []
    for base in bases:
        txt_path = os.path.join(rolm_ocr_dir, base + ".txt")
        md_path = os.path.join(smoldocling_dir, base + ".md")
        if os.path.isfile(txt_path) and os.path.isfile(md_path):
            valid_bases.append(base)

    if len(valid_bases) < sample_size:
        raise ValueError(
            f"Not enough complete sets (PDF+TXT+MD) in input directories (found {len(valid_bases)})"
        )

    for draw_idx in range(1, draws + 1):
        draw_name = f"draw_{draw_idx}"
        draw_path = os.path.join(output_dir, draw_name)
        # Create subdirectories
        for sub in ("full_pdf", "rolm_ocr", "smoldocling"):
            os.makedirs(os.path.join(draw_path, sub), exist_ok=True)

        # Sample bases without replacement within this draw
        sampled_bases = random.sample(valid_bases, sample_size)
        for base in sampled_bases:
            pdf_name = base + ".pdf"
            txt_name = base + ".txt"
            md_name = base + ".md"

            # Copy PDF
            shutil.copy2(
                os.path.join(full_pdf_dir, pdf_name),
                os.path.join(draw_path, "full_pdf", pdf_name),
            )

            # Copy OCR .txt
            shutil.copy2(
                os.path.join(rolm_ocr_dir, txt_name),
                os.path.join(draw_path, "rolm_ocr", txt_name),
            )

            # Copy smoldocling .md
            shutil.copy2(
                os.path.join(smoldocling_dir, md_name),
                os.path.join(draw_path, "smoldocling", md_name),
            )

        print(f"Completed {draw_name} with {sample_size} items.")


if __name__ == "__main__":
    # Define your directories here or integrate argparse as you like
    full_pdf_dir = "../../data/full_paper_final/pdfs"
    rolm_ocr_dir = "../../data/extracted_information/rolm_ocr"
    smoldocling_dir = "../../data/extracted_information/md_smoldocling"
    output_dir = "../../data/sample_pdf_eval"
    # Optionally set draws and sample_size
    sample_draws(
        full_pdf_dir, rolm_ocr_dir, smoldocling_dir, output_dir, draws=5, sample_size=20
    )
