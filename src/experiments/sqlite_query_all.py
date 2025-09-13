"""Evaluate refined SQLite queries against SQLite FTS5 and compute retrieval metrics."""

import json
import re
import sqlite3
import traceback
from pathlib import Path
from typing import List, Tuple, Iterable, Set, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# =============== Config ===============
DB_PATH = "../../data/final/exp/sqlite/refs.db"
DEBUG_SQL = False

JSON_WITH_REFINED_IN = "../../data/final/sr4cs.json"
JSON_WITH_REFINED_OUT = "../../data/final/sr4cs.json"
OUTPUT_SCORES = "../../data/final/exp/sqlite_eval_scores.parquet"
LOG_FILE = "../../logs/sqlite_eval.log"

# Deterministic repair only (no LLM)
CHECKPOINT_EVERY_SR = 50
CHECKPOINT_ON_FIXES = 10

# =============== Logging ===============
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("sqlite_eval")


# =============== SQLite ===============
def _connect_ro(db_path: str, debug: bool = DEBUG_SQL) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        con.execute("PRAGMA query_only = 1;")
    except sqlite3.OperationalError:
        pass
    if debug:
        con.set_trace_callback(lambda s: print(f"[TRACE] {s}"))
    return con


def _select_sql(table: str, with_text: bool, use_bm25: bool) -> str:
    cols = "ref_id, title, abstract" if with_text else "ref_id"
    order = f" ORDER BY bm25({table})" if use_bm25 else ""
    return f"SELECT {cols} FROM {table} WHERE {table} MATCH ?{order};"


def _fts_search_lit(raw_query: str, with_text: bool) -> List[Tuple]:
    con = _connect_ro(DB_PATH, debug=DEBUG_SQL)
    cur = con.cursor()

    sql = _select_sql("refs_fts_lit", with_text, use_bm25=True)
    if DEBUG_SQL:
        print(f"\n[QUERY] table=refs_fts_lit (bm25 attempt)")
        print(f"[MATCH] {raw_query}")
        print(f"[SQL  ] {sql.strip()}")
        print(f"[PARAM] {(raw_query,)}")
    try:
        rows = cur.execute(sql, (raw_query,)).fetchall()
        con.close()
        return rows
    except sqlite3.OperationalError as e:
        if DEBUG_SQL:
            print(f"[INFO ] bm25 ORDER BY not available: {e}")
            print("[INFO ] Falling back to no ORDER BY")

    sql_fallback = _select_sql("refs_fts_lit", with_text, use_bm25=False)
    if DEBUG_SQL:
        print(f"[SQL  ] {sql_fallback.strip()}")
        print(f"[PARAM] {(raw_query,)}")
    rows = cur.execute(sql_fallback, (raw_query,)).fetchall()
    con.close()
    return rows


def fts_search(raw_query: str, with_text: bool = True) -> Dict[str, Any]:
    """Search only the literal table (ES-like)."""
    rows_lit = _fts_search_lit(raw_query, with_text=with_text)
    return {
        "rows": rows_lit,
        "count_lit": len(rows_lit),
    }


def fts_search_ids(raw_query: str) -> List[str]:
    ids_lit = [str(r[0]) for r in _fts_search_lit(raw_query, with_text=False)]
    return ids_lit


# =============== Metrics ===============
def _to_str_set(ids: Iterable) -> Set[str]:
    return {str(x).strip() for x in ids if str(x).strip()}


def evaluate(retrieved_ids: Iterable, truth_ids: Iterable) -> dict:
    R = _to_str_set(retrieved_ids)
    T = _to_str_set(truth_ids)

    tp_ids = R & T
    fp_ids = R - T
    fn_ids = T - R

    tp, fp, fn = len(tp_ids), len(fp_ids), len(fn_ids)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    def f_beta(p: float, r: float, beta: float) -> float:
        if p == 0.0 and r == 0.0:
            return 0.0
        b2 = beta * beta
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


# =============== Ground truth from JSON ===============
def normalize_ref_ids(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val]
    return [str(val).strip()]


def build_truth_map_from_entries(entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    truth: Dict[str, List[str]] = {}
    for e in entries:
        sr_id = str(e.get("id", "")).strip()
        truth[sr_id] = normalize_ref_ids(e.get("ref_id"))
    return truth


# =============== Minimal sanitizer ===============
def _needs_quotes(token: str) -> bool:
    return bool(re.search(r'[^a-z0-9*"]', token))


def _quote_fielded_terms(s: str) -> str:
    # allow only title:/abstract:
    def _q(m: re.Match) -> str:
        fld, term = m.group(1), m.group(2)
        if term.startswith('"') and term.endswith('"') and len(term) >= 2:
            return f"{fld}:{term}"
        if _needs_quotes(term):
            return f'{fld}:"{term}"'
        return f"{fld}:{term}"

    return re.sub(rf"\b(title|abstract):([^\s()]+)", _q, s, flags=re.IGNORECASE)


def _remove_near(s: str) -> str:
    # FTS5 has NEAR, but your ES mirror likely has no cross-field column; safest: map NEAR->AND
    s = re.sub(r"\bNEAR\s*/\s*\d+\b", "AND", s, flags=re.IGNORECASE)
    s = re.sub(r"\bNEAR\b", "AND", s, flags=re.IGNORECASE)
    return s


def _drop_not_clauses(s: str) -> str:
    # Optional: drop NOT to avoid surprises; keep if you want strict NOT behavior.
    s = re.sub(r"\bAND\s+NOT\s*\((?:[^()]*|\([^()]*\))*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bNOT\s*\((?:[^()]*|\([^()]*\))*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r'\bAND\s+NOT\s+(?:"[^"]+"|\([^)]+\)|\S+)', "", s, flags=re.IGNORECASE)
    s = re.sub(r'\bNOT\s+(?:"[^"]+"|\([^)]+\)|\S+)', "", s, flags=re.IGNORECASE)
    return s


def _balance_quotes(s: str) -> str:
    if s.count('"') % 2 == 1:
        s = s.rstrip()
        s = s[:-1] if s.endswith('"') else s + '"'
    return s


def _balance_parens(s: str) -> str:
    opens = s.count("(")
    closes = s.count(")")
    if opens > closes:
        s = s + (")" * (opens - closes))
    elif closes > opens:
        surplus = closes - opens
        for _ in range(surplus):
            pos = s.rfind(")")
            if pos >= 0:
                s = s[:pos] + s[pos + 1 :]
    return s


def _strip_trailing_boolean(s: str) -> str:
    return re.sub(r"\s*(?:OR|AND)\s*\)*\s*$", "", s, flags=re.IGNORECASE)


def normalize_query(q: str) -> str:
    s = q.strip()
    s = _remove_near(s)
    s = _drop_not_clauses(s)
    s = _quote_fielded_terms(s)
    s = _balance_quotes(s)
    s = _balance_parens(s)
    s = _strip_trailing_boolean(s)
    s = " ".join(s.split())
    return s


def quick_sanitize(q: str) -> str:
    return normalize_query(q)


# =============== Deterministic recovery ===============
def run_query_no_llm(raw_q: str) -> Dict[str, Any]:
    """
    Try raw → quick_sanitize → normalize_query.
    No LLM involved.
    """
    # 1) raw
    try:
        return fts_search(raw_q, with_text=True)
    except sqlite3.OperationalError as e:
        log.error(f"RAW failed: {e}")

    # 2) deterministic sanitize
    q1 = quick_sanitize(raw_q)
    if q1 != raw_q:
        log.info(f"Applying quick_sanitize → {q1[:240]}{' …' if len(q1)>240 else ''}")
        try:
            return fts_search(q1, with_text=True)
        except sqlite3.OperationalError as e:
            log.error(f"SANITIZED failed: {e}")

    # 3) last-resort normalize and try once
    q_last = normalize_query(raw_q)
    log.info(f"Last-resort normalize → {q_last[:240]}{' …' if len(q_last)>240 else ''}")
    res = fts_search(q_last, with_text=True)  # let error raise to caller
    return res


# =============== Main ===============
if __name__ == "__main__":
    log.info("=== Run started (no LLM) ===")
    log.info(f"DB_PATH={DB_PATH}")
    log.info(f"JSON_WITH_REFINED_IN={JSON_WITH_REFINED_IN}")
    log.info(f"JSON_WITH_REFINED_OUT={JSON_WITH_REFINED_OUT}")
    log.info(f"OUTPUT_SCORES={OUTPUT_SCORES}")

    # Load JSON (entries contain sqlite_refined_queries and ref_id list)
    with open(JSON_WITH_REFINED_IN, "r", encoding="utf-8") as f:
        entries: List[Dict[str, Any]] = json.load(f)
    log.info(f"Loaded {len(entries)} SR entries from JSON.")

    # Build ground truth map directly from JSON
    truth_map = build_truth_map_from_entries(entries)
    log.info(f"Built ground-truth map for {len(truth_map)} SR ids from JSON.")

    # Ensure OUT dir exists
    Path(JSON_WITH_REFINED_OUT).parent.mkdir(parents=True, exist_ok=True)

    fixed_count = 0  # counts deterministic fixes applied (sanitized/normalized)
    rows_for_df = []

    for i_entry, entry in enumerate(tqdm(entries, desc="Evaluating", unit="sr"), 1):
        sr_id = str(entry.get("id", "")).strip()
        refined_queries: List[str] = entry.get("sqlite_refined_queries") or []

        try:
            log.info(f"[SR {sr_id}] Start | n_queries={len(refined_queries)}")
            retrieved_ids_union: Set[str] = set()

            for qi, q in enumerate(refined_queries):
                q = (q or "").strip()
                if not q:
                    log.info(f"[SR {sr_id}] Query {qi+1}: empty → skip")
                    continue

                q_preview = q if len(q) <= 300 else (q[:300] + " …")
                log.info(f"[SR {sr_id}] Running Query {qi+1}: {q_preview}")

                try:
                    res = fts_search(q, with_text=True)
                except sqlite3.OperationalError:
                    # Try deterministic recovery (no LLM)
                    res = run_query_no_llm(q)
                    fixed_count += 1
                    # If we changed the query deterministically, persist it back
                    # Best-effort: write the normalized version we ended up using
                    try:
                        final_q = quick_sanitize(q)
                        if isinstance(entry.get("sqlite_refined_queries"), list):
                            entry["sqlite_refined_queries"][qi] = final_q
                    except Exception:
                        pass

                rows = res["rows"]
                log.info(f"[SR {sr_id}] Query {qi+1} results | lit={res['count_lit']}")
                retrieved_ids_union.update(str(ref_id) for ref_id, _t, _a in rows)

            # Evaluate
            truth_ids = truth_map.get(sr_id, [])
            log.info(f"[SR {sr_id}] truth_ids={truth_ids}")
            log.info(f"[SR {sr_id}] retrieved_ids_union={sorted(retrieved_ids_union)}")

            metrics = evaluate(retrieved_ids_union, truth_ids)
            log.info(
                f"[SR {sr_id}] Metrics | retrieved={metrics['retrieved_count']} "
                f"truth={metrics['truth_count']} tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} F3={metrics['f3']:.4f}"
            )
            rows_for_df.append(
                {"id": sr_id, "n_queries": len(refined_queries), **metrics}
            )

        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"[SR {sr_id}] ERROR: {e}\n{tb}")
            rows_for_df.append(
                {
                    "id": sr_id,
                    "n_queries": len(refined_queries),
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "f3": 0.0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "retrieved_count": 0,
                    "truth_count": len(truth_map.get(sr_id, [])),
                    "tp_ids": [],
                    "fp_ids": [],
                    "fn_ids": [],
                }
            )

        # Checkpoint JSON and progress
        if (i_entry % CHECKPOINT_EVERY_SR == 0) or (
            fixed_count and fixed_count % CHECKPOINT_ON_FIXES == 0
        ):
            with open(JSON_WITH_REFINED_OUT, "w", encoding="utf-8") as wf:
                json.dump(entries, wf, ensure_ascii=False, indent=2)
            log.info(
                f"[CKPT] Saved corrected JSON after SR={i_entry}, fixes_so_far={fixed_count}"
            )

    # Final saves
    with open(JSON_WITH_REFINED_OUT, "w", encoding="utf-8") as wf:
        json.dump(entries, wf, ensure_ascii=False, indent=2)
    log.info(f"Final corrected JSON written → {JSON_WITH_REFINED_OUT}")

    scores_df = pd.DataFrame(rows_for_df)
    Path(OUTPUT_SCORES).parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(OUTPUT_SCORES, index=False)
    log.info(f"Wrote scores to {OUTPUT_SCORES}")
    print(f"[OK] Wrote scores to {OUTPUT_SCORES} (deterministic_fixes={fixed_count})")
    log.info("=== Run finished ===")
