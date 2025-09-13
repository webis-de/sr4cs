"""Transform bibliographic Boolean queries to SQLite FTS5 MATCH syntax using Azure OpenAI."""

import os
import json
import time
import logging
from typing import List
from pathlib import Path

from tqdm import tqdm
from openai import AzureOpenAI
from openai import APIError, RateLimitError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

# ========================
# Config
# ========================
AZURE_ENDPOINT = "https://ai-dsiplayground101757747291.cognitiveservices.azure.com/"
API_KEY = os.getenv("GPT4_1_MINI_API_KEY")
API_VERSION = "2024-12-01-preview"
DEPLOYMENT_NAME = "gpt-4.1-mini"

COMBINED_JSON = Path("../../data/final/sr4cs.json")
OUTPUT_JSON = Path("../../data/final/sr4cs_with_sql.json")
LOG_FILE = Path("../../logs/sqlite_refine.log")

CHECKPOINT_EVERY = 20
MAX_COMPLETION_TOKENS = 16000
TEMPERATURE = 0.0
RETRIES = 4
BACKOFF_BASE = 2.0
SKIP_DONE = True

# ========================
# Logging
# ========================
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("sqlite_refiner")

# ========================
# Azure client
# ========================
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION,
)

SYSTEM_PROMPT = "You are a precise query rewriter. Convert bibliographic Boolean strings into valid SQLite FTS5 MATCH queries for a virtual table with ONLY two indexed columns: title and abstract. Return only the rewritten queries; one FTS5 query per input line, in order. Your output must be executable by FTS5."

USER_PROMPT_TEMPLATE = r"""
    <task>
    <context>
    Translate Boolean queries from different bibliographic databases into SQLite FTS5 MATCH syntax for a virtual table with only two columns: title, abstract.
    </context>

    <instructions>
    <hard-constraints>
    <item>Preserve Boolean logic and parentheses exactly (AND/OR/NOT in UPPERCASE).</item>
    <item>Use ONLY fields title: and abstract:. Never use title:(...) or abstract:(...). Repeat the field prefix for EVERY token that needs it.</item>
    <item>Field mapping:
    - INTITLE | TITLE | TI -> title:...
    - INABSTRACT | ABSTRACT | AB -> abstract:...
    - TITLE-ABS-KEY or unknown fields (TOPIC:, KEYS:, KEYWORDS:, KW:, SU:, DE:) -> (title:... OR abstract:...)</item>
    <item>If a token/phrase has NO field specified, or if nothing is specified, wrap it as (title:TERM_OR_PHRASE OR abstract:TERM_OR_PHRASE). Default to searching both title and abstract if no field is set.</item>
    <item>Remove unsupported metadata/filters (years, language/source limits, document types, LIMIT-TO, PUBYEAR, SRCTYPE, etc.).</item>
    <item>Use ASCII straight quotes only. Replace “smart quotes“ with ".</item>
    <item>Avoid putting wildcards inside quotes. If the input has a quoted phrase with a wildcard (e.g., "success factor*"), rewrite as same-column tokens with prefixes, e.g., (title:success AND title:factor*) (and the abstract analogue).</item>
    <item>Hyphens: unquoted hyphens are NOT operators in FTS5. For hyphenated terms (e.g., data-driven), output safe variants: title:"data driven" OR (title:data AND title:driven). Keep the abstract analogue too. If the input has a quoted hyphenated phrase, also include the spaced variant.</item>
    <item>Slashes and similar shorthand (e.g., cardiomyopathy/ies): rewrite to a safe stem/prefix (cardiomyopath*), or explicit OR if clearer.</item>
    <item>NEAR/PHRASE PROXIMITY: FTS5 does not allow NEAR with column-qualified terms. If the input uses NEAR, replace it with same-column co-occurrence using AND (e.g., (title:a AND title:b)) or drop column qualifiers for the NEAR clause if proximity must be preserved. Prefer same-column AND.</item>
    <item>NOT: Use valid FTS5 negation. Example: A AND NOT (B OR C). Wrap multi-term exclusions in parentheses.</item>
    <item>Lowercase all terms; keep Boolean operators uppercase.</item>
    <item>Output RAW FTS5 queries only, one per input line, with no extra text or code fences.</item>
    </hard-constraints>
    <few-shot-examples>
        <example>
            <input>TI:(cancer AND immunotherapy) AND NOT (AB:review)</input>
            <output>(title:cancer AND title:immunotherapy) AND NOT (abstract:review)</output>
        </example>
        <example>
            <input>"data-driven" OR cardiomyopathy/ies</input>
            <output>(title:"data driven" OR (title:data AND title:driven) OR abstract:"data driven" OR (abstract:data AND abstract:driven)) OR (title:cardiomyopath* OR abstract:cardiomyopath*)</output>
        </example>
        <example>
            <input>TOPIC:(machine learning OR AI) AND PUBYEAR > 2020</input>
            <output>((title:machine OR abstract:machine) AND (title:learning OR abstract:learning) OR (title:ai OR abstract:ai))</output>
        </example>
        <example>
            <input>success factor*</input>
            <output>(title:success OR abstract:success) AND (title:factor* OR abstract:factor*)</output>
        </example>
    </few-shot-examples>
    </instructions>

    <io>
    <input>
    <<YOUR_LINES>>
    </input>
    <output>Return only the rewritten FTS5 queries, one per line.</output>
    </io>
    </task>
""".strip()


def run_chat(system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    out = (resp.choices[0].message.content or "").strip()
    if out.startswith("```"):  # strip accidental fences
        lines = out.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        out = "\n".join(lines).strip()
    return out


def refine_queries(queries: List[str]) -> List[str]:
    user_prompt = USER_PROMPT_TEMPLATE.replace("<<YOUR_LINES>>", "\n".join(queries))
    for attempt in range(1, RETRIES + 1):
        try:
            raw = run_chat(SYSTEM_PROMPT, user_prompt)
            refined_list = [line.strip() for line in raw.splitlines() if line.strip()]
            return refined_list
        except (RateLimitError, APITimeoutError, APIError, Exception) as e:
            wait = BACKOFF_BASE ** (attempt - 1)
            logger.warning(
                f"LLM call failed (attempt {attempt}/{RETRIES}): {type(e).__name__}: {e}. Backing off {wait:.1f}s"
            )
            time.sleep(wait)
    logger.error("Exhausted retries; returning empty refinement.")
    return []


if __name__ == "__main__":
    with COMBINED_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} entries from {COMBINED_JSON}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    processed_since_ckpt = 0
    total_processed = 0

    with tqdm(total=len(data), desc="Refining", unit="entry") as pbar:
        for idx, entry in enumerate(data):
            entry_id = entry.get("id")
            queries = entry.get("search_strings_boolean", []) or []

            if (
                SKIP_DONE
                and isinstance(entry.get("sqlite_refined_queries"), list)
                and entry["sqlite_refined_queries"]
            ):
                logger.info(
                    f"[SKIP] id={entry_id} already has sqlite_refined_queries ({len(entry['sqlite_refined_queries'])})"
                )
                pbar.update(1)
                continue

            logger.info(f"Processing id={entry_id} with {len(queries)} queries")
            if queries:
                refined = refine_queries(queries)
            else:
                refined = []

            entry["sqlite_refined_queries"] = refined
            logger.info(f"Done id={entry_id}: refined={len(refined)}")

            processed_since_ckpt += 1
            total_processed += 1
            pbar.update(1)

            # checkpoint
            if processed_since_ckpt >= CHECKPOINT_EVERY:
                with OUTPUT_JSON.open("w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(
                    f"Checkpoint saved after {total_processed} processed → {OUTPUT_JSON}"
                )
                processed_since_ckpt = 0

    # final save
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Final saved {len(data)} entries → {OUTPUT_JSON}")
    print(f"[OK] Saved with refined queries → {OUTPUT_JSON}")
    print(f"[LOG] See details in {LOG_FILE}")
