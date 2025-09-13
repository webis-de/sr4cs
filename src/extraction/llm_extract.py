"""
Extract structured search information from systematic reviews using Azure OpenAI GPT-4.1-Mini.
"""

import os
import glob
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import tiktoken
from openai import AzureOpenAI

# =========================
# Config 
# =========================
INPUT_DIR = "../../data/full_paper_final/parsed"
OUTPUT_DIR = "../../data/extracted_information/search_data/nanonet"
LOG_FILE = "../../logs/llm_extraction_nanonet.log"

AZURE_ENDPOINT = "https://ai-dsiplayground101757747291.cognitiveservices.azure.com/"
API_KEY = os.getenv("GPT4_1_MINI_API_KEY")
API_VERSION = "2024-12-01-preview"
DEPLOYMENT_NAME = "gpt-4.1-mini"

# Model/context knobs
MAX_INPUT_TOKENS = 1_000_000
COMPLETION_TOKENS_BUDGET = 2_000
TRUNCATION_SAFETY = 0.95
TEMPERATURE = 0.0
TOK_CHUNK_CHARS = 100_000

# =========================
# Init
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a",
)

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT, api_key=API_KEY, api_version=API_VERSION
)

ENC = tiktoken.get_encoding("cl100k_base")


# =========================
# Helpers
# =========================
def count_tokens(text: str) -> int:
    """Robust token count using chunked encoding to avoid tiktoken stack overflows."""
    total = 0
    n = len(text)
    for i in range(0, n, TOK_CHUNK_CHARS):
        chunk = text[i : i + TOK_CHUNK_CHARS]
        try:
            total += len(ENC.encode(chunk))
        except Exception as e:
            logging.warning(
                f"tiktoken failed on chunk [{i}:{i+TOK_CHUNK_CHARS}] ({len(chunk)} chars): {e}. "
                f"Using ~4 chars/token estimate for this chunk."
            )
            total += max(1, len(chunk) // 4)
    return total


def get_file_id(path: str) -> str:
    """Extract file id from filename prefix before first underscore or dot."""
    name = os.path.basename(path)
    fid = name.split("_")[0]
    return fid.split(".")[0]


def already_done(file_id: str) -> bool:
    return os.path.exists(os.path.join(OUTPUT_DIR, f"{file_id}.json"))


def truncate_to_fit(text: str, system_prompt: str, user_template_no_text: str) -> str:
    """
    Truncate the SLR text so that system + user(template+text) <= MAX_INPUT_TOKENS - COMPLETION_TOKENS_BUDGET.
    """
    base_tokens = count_tokens(system_prompt) + count_tokens(user_template_no_text)
    budget = MAX_INPUT_TOKENS - COMPLETION_TOKENS_BUDGET - base_tokens
    if budget <= 0:
        raise ValueError(
            "No room left for content. Reduce prompt lengths or increase MAX_INPUT_TOKENS."
        )

    text_tokens = count_tokens(text)
    if text_tokens <= budget:
        return text

    # Slice by character ratio and try to cut at a period boundary.
    ratio = (budget / max(1, text_tokens)) * TRUNCATION_SAFETY
    cut_chars = max(1, int(len(text) * ratio))
    candidate = text[:cut_chars]
    last_period = candidate.rfind(".")
    if last_period >= int(
        cut_chars * 0.7
    ):  # keep most of the slice, cut at a sentence end if close
        candidate = candidate[: last_period + 1]

    # Ensure final fits (rarely needed)
    while count_tokens(candidate) > budget and len(candidate) > 1000:
        candidate = candidate[: int(len(candidate) * 0.95)]

    return candidate


def parse_json_strict(raw: str, filename: str) -> Dict[str, Any]:
    """
    Minimal parsing: expect clean JSON; fallback to grabbing the first {...} block.
    """
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # fallback: extract first JSON object block
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception as e:
                logging.error(
                    f"[{filename}] JSON fallback failed: {e}\nSnippet: {snippet[:1000]}"
                )
        raise


# =========================
# Prompts (kept EXACTLY the same as you posted)
# =========================
SYSTEM_PROMPT = (
    "You are an expert in systematic-review methodology. "
    "From each SLR full-text, extract verbatim information and output ONLY valid JSON "
    'with these keys (use [] or "" if the element is absent): '
    "databases, search_strings_boolean, year_range, language_restrictions, "
    "inclusion_criteria, exclusion_criteria, topic, objective, research_questions. "
    "Return ONLY the JSON object with no additional text, tags, or formatting."
)

USER_TEMPLATE = """<context>
You will receive the full text of a systematic review (SR).
It may list databases searched, Boolean queries used, time windows, language limits, inclusion/exclusion criteria,
the topic/objective, and whether the study performed snowballing/citation chasing.
</context>

<instructions>
Extract the following key information exactly as they appear. Avoid paraphrasing or inference beyond boolean detection for snowballing.
If a field is not explicitly present, use [] for lists or "" for strings. For snowballing, return true if the SR reports
backward/forward citation chasing, screening references of included studies, using seed papers for expansion, hand-searching
reference lists, or explicitly mentions "snowballing" (or equivalent). Otherwise, return false.

Required fields:
- databases (list) // The databases searched 
- search_strings_boolean (list) // The exact boolean search strings used
- year_range (string) // The years covered by the search 
- language_restrictions (list) // Languages included in the search
- inclusion_criteria (list) // The inclusion criteria as a list
- exclusion_criteria (list) // The exclusion criteria as a list 
- topic (string) // The main topic of the systematic review (if the name is systematic review of X, extract X)
- objective (string) // The main objective or research goal of the systematic review
- research_questions (list) // The specific research questions the review aims to answer
- snowballing (boolean) // true if the review performed snowballing/citation chasing, false otherwise

<output_format>
Return ONLY valid JSON in exactly this structure (no extra text):

{
  "databases": [], // list of strings (e.g., ["PubMed", "Scopus"])
  "search_strings_boolean": [], // list of strings (e.g., ["(cancer OR tumor) AND (therapy OR treatment)"])
  "year_range": "", // string (e.g., "2000-2023")
  "language_restrictions": [], // list of strings (e.g., ["English", "French"])
  "inclusion_criteria": [], // list of strings (e.g., ["studies on adults", "randomized controlled trials"])
  "exclusion_criteria": [], // list of strings (e.g., ["case reports", "non-peer-reviewed articles"])
  "topic": "", // string (e.g., "The effectiveness of immunotherapy in treating melanoma")
  "objective": "", // string (e.g., "To evaluate the efficacy and safety of immunotherapy for melanoma patients")
  "research_questions": [], // list of strings (e.g., ["What is the response rate of immunotherapy in melanoma?", "What are the common side effects?"])
  "snowballing": false // boolean (true or false)
}
</output_format>

<text>
{TEXT}
</text>
"""


# =========================
# Core
# =========================
def process_one(path: str) -> Dict[str, Any]:
    file_id = get_file_id(path)
    with open(path, "r", encoding="utf-8") as f:
        slr_text = f.read().strip()

    user_no_text = USER_TEMPLATE.replace("{TEXT}", "")
    truncated_text = truncate_to_fit(slr_text, SYSTEM_PROMPT, user_no_text)
    user_prompt = USER_TEMPLATE.replace("{TEXT}", truncated_text)

    # quick visibility in logs
    total_prompt_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(user_prompt)
    logging.info(f"Processing {os.path.basename(path)} | tokens={total_prompt_tokens}")

    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=COMPLETION_TOKENS_BUDGET,
        temperature=TEMPERATURE,
        top_p=1.0,
    )

    raw = resp.choices[0].message.content or ""
    data = parse_json_strict(raw, os.path.basename(path))

    # ensure the boolean key exists
    if "snowballing" not in data or not isinstance(data["snowballing"], bool):
        data["snowballing"] = False  # default to false if missing or malformed

    return {"id": file_id, **data}


def main():
    patterns = [str(Path(INPUT_DIR) / "*.txt"), str(Path(INPUT_DIR) / "*.md")]
    files: List[str] = sum((glob.glob(p) for p in patterns), [])
    if not files:
        print(f"No .txt or .md files in {INPUT_DIR}")
        return

    to_process = []
    for p in files:
        fid = get_file_id(p)
        if already_done(fid):
            logging.info(f"Skip {os.path.basename(p)} (already processed)")
        else:
            to_process.append(p)

    print(
        f"Found {len(files)} files; processing {len(to_process)}, skipping {len(files) - len(to_process)}"
    )
    # print the ids of the files needing processing
    if to_process:
        print("Files to process:")
        for p in to_process:
            print(f" - {os.path.basename(p)}")

    for p in tqdm(to_process, desc="Processing"):
        try:
            out = process_one(p)
            out_path = os.path.join(OUTPUT_DIR, f"{out['id']}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved: {out_path}")
        except Exception as e:
            logging.error(f"Error processing {p}: {e}")
            print(f"Error {p}: {e}")

    print(f"Done. Output → {OUTPUT_DIR}\nLogs   → {LOG_FILE}")


if __name__ == "__main__":
    main()
