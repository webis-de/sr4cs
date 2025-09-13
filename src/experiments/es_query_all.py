"""Evaluate Elasticsearch retrieval for all SRs and log results."""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Iterable, Set

import pandas as pd
from tqdm import tqdm
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

load_dotenv()

# ========= Config =========
ES_URL = "https://elasticsearch.srv.webis.de"
ES_INDEX = "pierre-sr-boolq"

# read from .env to avoid hardcoding in the script
API_KEY_ENCODED = os.getenv("API_KEY_ENCODED")


JSON_IN = "../../data/final/sr4cs_with_sql.json"
PARQUET_OUT = "../../data/final/exp/es_eval_scores.parquet"
LOG_FILE = "../../logs/es_eval.log"

SCROLL_SIZE = 1000
SCROLL_TTL = "1m"

# Logging verbosity toggles
LOG_PER_QUERY_IDS = False  # set True to also log IDs returned per individual query
# ==========================

# --- Logging setup ---
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("es_eval")


def es_client() -> Elasticsearch:
    return Elasticsearch(ES_URL, api_key=API_KEY_ENCODED, request_timeout=120)


# --- ID normalization ---
def norm_id(x) -> str:
    """Normalize IDs"""
    if x is None:
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        return s[:-2]
    try:
        f = float(s)
        i = int(f)
        return str(i) if f == i else s
    except:
        return s


def sorted_ids(iterable) -> list:
    """Return sorted, de-duplicated, normalized list of ids."""
    return sorted({norm_id(x) for x in iterable if norm_id(x)})


def search_all_ids(es: Elasticsearch, index: str, raw_query: str) -> List[str]:
    log.info(f"Running query: {raw_query}")
    if not raw_query:
        return []
    body = {
        "query": {
            "query_string": {
                "query": raw_query,  # <-- sent as-is
                "fields": ["title", "abstract"],
                "default_operator": "AND",
                "analyze_wildcard": True,
                "lenient": True,
            }
        },
        "_source": ["ref_id"],
    }
    res = es.search(index=index, body=body, size=SCROLL_SIZE, scroll=SCROLL_TTL)
    sid = res.get("_scroll_id")
    hits = res.get("hits", {}).get("hits", [])
    out: List[str] = []

    def add(hs):
        for h in hs:
            src = h.get("_source") or {}
            rid = src.get("ref_id")
            out.append(norm_id(rid) if rid is not None else norm_id(h.get("_id")))

    add(hits)

    while True:
        if not sid or not hits:
            break
        res = es.scroll(scroll_id=sid, scroll=SCROLL_TTL)
        sid = res.get("_scroll_id")
        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            break
        add(hits)

    try:
        if sid:
            es.clear_scroll(scroll_id=sid)
    except Exception as e:
        log.warning(f"clear_scroll failed: {e}")

    log.info(f"Query returned {len(out)} hits")
    return out


def to_set(ids: Iterable) -> Set[str]:
    return {norm_id(x) for x in ids if norm_id(x)}


def evaluate(retrieved_ids: Iterable, truth_ids: Iterable) -> Dict[str, Any]:
    R, T = to_set(retrieved_ids), to_set(truth_ids)
    tp_ids, fp_ids, fn_ids = R & T, R - T, T - R
    tp, fp, fn = len(tp_ids), len(fp_ids), len(fn_ids)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    def f_beta(p, r, b):
        if p == 0 and r == 0:
            return 0.0
        b2 = b * b
        denom = b2 * p + r
        return (1 + b2) * p * r / denom if denom else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f_beta(precision, recall, 1.0),
        "f3": f_beta(precision, recall, 3.0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_ids": sorted(tp_ids),
        "fp_ids": sorted(fp_ids),
        "fn_ids": sorted(fn_ids),
        "retrieved_count": len(R),
        "truth_count": len(T),
    }


if __name__ == "__main__":
    log.info("=== Run started ===")
    es = es_client()
    info = es.info()
    log.info(
        f"Connected to ES: {info.get('name')} v{info.get('version', {}).get('number')}"
    )

    with open(JSON_IN, "r", encoding="utf-8") as f:
        entries: List[Dict[str, Any]] = json.load(f)
    log.info(f"Loaded {len(entries)} SR entries from {JSON_IN}")

    rows = []
    t0 = time.time()

    for e in tqdm(entries, desc="Evaluating", unit="sr"):
        sr_id = str(e.get("id", "")).strip()
        queries: List[str] = e.get("sqlite_refined_queries") or []
        truth_ids: List[Any] = e.get("ref_id") or []

        retrieved_union: Set[str] = set()
        for qi, q in enumerate(queries):
            if not q:
                continue
            try:
                ids = search_all_ids(es, ES_INDEX, q)
                retrieved_union.update(ids)
                log.info(f"[SR {sr_id}] Query {qi+1} returned {len(ids)} docs")
                if LOG_PER_QUERY_IDS:
                    log.info(
                        f"[SR {sr_id}] Query {qi+1} retrieved_ids={sorted_ids(ids)}"
                    )
            except Exception as ex:
                log.error(f"[SR {sr_id}] Query {qi+1} failed: {ex}")

        # Log normalized truth and retrieved unions for inspection
        truth_sorted = sorted_ids(truth_ids)
        retrieved_sorted = sorted_ids(retrieved_union)
        log.info(f"[SR {sr_id}] truth_ids={truth_sorted}")
        log.info(f"[SR {sr_id}] retrieved_ids_union={retrieved_sorted}")

        metrics = evaluate(retrieved_sorted, truth_sorted)
        log.info(
            f"[SR {sr_id}] P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, F3={metrics['f3']:.4f}, "
            f"retrieved={metrics['retrieved_count']}, truth={metrics['truth_count']}"
        )
        rows.append({"id": sr_id, "n_queries": len(queries), **metrics})

    df = pd.DataFrame(rows)
    Path(PARQUET_OUT).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_OUT, index=False)
    log.info(f"Wrote scores to {PARQUET_OUT}")
    log.info(f"=== Run finished in {time.time()-t0:.1f}s ===")
    print(f"[OK] wrote {PARQUET_OUT}")
