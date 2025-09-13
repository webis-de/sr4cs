# SR4CS — Systematic Review Test Collection for Computer Science

### Overview

SR4CS is a benchmark dataset of systematic reviews in Computer Science. This repository provides the full codebase for dataset construction and baseline experiments.
- The dataset itself (JSON + reference pool in multiple formats) is hosted on Zenodo: LINK

This repository contains the pipelines for:
- Retrieving and filtering candidate SRs from DBLP.
- Parsing PDFs into structured text.
- Extracting search methodology fields with LLMs.
- Extracting and enriching references (metadata + abstracts).
- Building final JSON/Parquet/SQLite/Elasticsearch datasets.
- Translating Boolean Queries to SQL Match Syntax.
- Running baseline retrieval experiments (SQLite FTS5 and Elasticsearch).

---

### Repository Layout
- src/retrieval/ — Fetch, filter, and prepare SR candidates + PDFs.
- src/extraction/ — OCR/LLM field extraction and reference parsing + enrichment.
- src/utils/ — Assembly and dataset hygiene (ID updates, metadata integration).
- src/experiments/ — Retrieval baselines (SQLite, Elasticsearch).

---

### Installation
- Python 3.10+
- Install deps:
    `pip install -r requirements.txt`

- External tools/services:
  - Grobid (reference extraction, requires local service).
  - AnyStyle (citation parsing).
  - Azure OpenAI (for LLM field/query extraction).
  - OCR model (Nanonets-OCR-s via VLLM).

---