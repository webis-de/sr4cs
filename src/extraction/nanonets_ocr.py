"""Extract text from PDFs using Nanonets OCR model via OpenAI API."""

from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
import base64, io, os, glob, logging, re

# ---------- config ----------
OPENAI_API_KEY = "123"
OPENAI_BASE_URL = "http://gammaweb08.medien.uni-weimar.de:8000/v1"
MODEL_NAME = "nanonets/Nanonets-OCR-s"

PDF_DIR = "../../data/full_paper_final/pdfs"
OUTPUT_DIR = "../../data/full_paper_final/parsed"
LOG_PATH = "../../logs/nanonets_dir_vllm.log"
DPI = 200

# ---------- logging ----------
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger().addHandler(logging.StreamHandler())  # console too

# ---------- OpenAI client ----------
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ---------- helpers ----------
PAGE_TAG_RE = re.compile(r"<!-- Page (\d+)/(\d+) -->")


def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ocr_page(
    img_b64: str, page_no: int, total_pages: int, max_tokens: int = 4096
) -> str:
    prompt = (
        "Extract the text from the above document as if you were reading it naturally. "
        "Return the tables in html format. Return the equations in LaTeX representation. "
        "If there is an image in the document and image caption is not present, add a small description "
        "of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. "
        "Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. "
        f"<page_number>{page_no}/{total_pages}</page_number>. "
        "Prefer using ☐ and ☑ for check boxes."
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def md_complete(path: str) -> bool:
    """
    True if `path` exists and its last <!-- Page x/y --> marker
    indicates x == y (fully processed).
    """
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as f:
        text = f.read()
    matches = PAGE_TAG_RE.findall(text)
    if not matches:
        return False
    last_x, last_y = map(int, matches[-1])
    return last_x == last_y


def pdf_to_markdown(pdf_path: str, output_path: str, dpi: int = 200):
    pages = convert_from_path(pdf_path, dpi=dpi)
    total = len(pages)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for i, page in enumerate(
            tqdm(pages, desc=os.path.basename(pdf_path), leave=False), start=1
        ):
            out.write(f"\n\n<!-- Page {i}/{total} -->\n")
            out.write(ocr_page(encode_image(page), i, total, max_tokens=15000))

    logging.info("Converted %s (%d pages)", pdf_path, total)


def process_directory(pdf_dir: str, output_dir: str, dpi: int = 200):
    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_files:
        logging.warning("No PDFs in %s", pdf_dir)
        return

    logging.info("Found %d PDFs", len(pdf_files))
    for pdf in tqdm(pdf_files, desc="PDFs", unit="file"):
        print(f"Processing {pdf}...")
        md_path = os.path.join(
            output_dir, os.path.splitext(os.path.basename(pdf))[0] + ".md"
        )
        if md_complete(md_path):
            logging.info("✓ Skipping already-done %s", pdf)
            continue
        try:
            logging.info("→ Processing %s", pdf)
            pdf_to_markdown(pdf, md_path, dpi=dpi)
        except Exception as e:
            logging.exception("✗ Failed on %s: %s", pdf, e)


# ---------- run ----------
if __name__ == "__main__":
    process_directory(PDF_DIR, OUTPUT_DIR, dpi=DPI)
