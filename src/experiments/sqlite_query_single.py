"""Execute a single SQLite FTS5 query (for testing/debugging)."""

import sqlite3
from typing import List, Tuple

# =============== Config ===============
DB_PATH = "../../data/final/exp/sqlite/refs.db"
DEBUG_SQL = False


# =============== Connection (read-only) ===============
def _connect_ro(db_path: str, debug: bool = DEBUG_SQL) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        con.execute("PRAGMA query_only = 1;")
    except sqlite3.OperationalError:
        pass
    if debug:
        con.set_trace_callback(lambda s: print(f"[TRACE] {s}"))
    return con


# =============== Core search helpers ===============
def _select_sql(table: str, with_text: bool, use_bm25: bool) -> str:
    cols = "ref_id, title, abstract" if with_text else "ref_id"
    order = f" ORDER BY bm25({table})" if use_bm25 else ""
    return f"SELECT {cols} FROM {table} WHERE {table} MATCH ?{order};"


def _fts_search_table(raw_query: str, table: str, with_text: bool) -> List[Tuple]:
    """
    Try ORDER BY bm25(table); if not available, fall back to no ORDER BY.
    """
    con = _connect_ro(DB_PATH, debug=DEBUG_SQL)
    cur = con.cursor()

    # Try with bm25 ordering
    sql = _select_sql(table, with_text, use_bm25=True)
    try:
        rows = cur.execute(sql, (raw_query,)).fetchall()
        con.close()
        return rows
    except sqlite3.OperationalError:
        pass

    # Fallback without ORDER BY
    sql_fallback = _select_sql(table, with_text, use_bm25=False)
    rows = cur.execute(sql_fallback, (raw_query,)).fetchall()
    con.close()
    return rows


# =============== Public APIs ===============
def fts_search(raw_query: str) -> List[Tuple[str, str, str]]:
    """
    Search only refs_fts_lit (with text).
    Each row is (ref_id, title, abstract).
    """
    return _fts_search_table(raw_query, "refs_fts_lit", with_text=True)


def fts_search_ids(raw_query: str) -> List[str]:
    """
    Return only ref_ids from refs_fts_lit.
    """
    ids = [
        str(r[0]) for r in _fts_search_table(raw_query, "refs_fts_lit", with_text=False)
    ]
    return ids


# =============== Demo (execute a single query) ===============
if __name__ == "__main__":
    query =     "(((title:leg OR abstract:leg) OR (title:foot OR abstract:foot) OR (title:hip OR abstract:hip) OR (title:knee OR abstract:knee) OR (title:ankle OR abstract:ankle) OR ((title:lower OR abstract:lower) AND ((title:limb* OR abstract:limb*) OR (title:extremity OR abstract:extremity) OR (title:body OR abstract:body)))) AND ((title:rehab* OR abstract:rehab*) OR (title:assist* OR abstract:assist*) OR (title:treat* OR abstract:treat*) OR (title:pathological OR abstract:pathological)) AND ((title:wearable OR abstract:wearable) OR (title:ortho* OR abstract:ortho*) OR (title:robot* OR abstract:robot*) OR (title:exoskeleton OR abstract:exoskeleton) OR (title:actuat* OR abstract:actuat*) OR (title:powered OR abstract:powered))) NOT (title:control OR abstract:control OR title:classif* OR abstract:classif* OR title:recognition OR abstract:recognition OR title:review OR abstract:review OR title:analysis OR abstract:analysis OR title:examin* OR abstract:examin* OR title:comparison OR abstract:comparison OR title:investig* OR abstract:investig* OR title:estimation OR abstract:estimation OR title:effect OR abstract:effect OR title:simul* OR abstract:simul* OR title:assess* OR abstract:assess* OR title:evaluation OR abstract:evaluation)"
    # Option A: just the IDs (fast path)
    ids = fts_search_ids(query)
    print(f"[IDs] Retrieved unique IDs: {len(ids)}")
    print(ids[:50])  # preview first 50

    # Option B: full rows with title/abstract (slower, but useful to inspect)
    rows = fts_search(query)
    print(f"\n[ROWS] Retrieved rows: {len(rows)}")
    # Pretty-print first 5 hits
    for i, (ref_id, title, abstract) in enumerate(rows[:5], 1):
        print(f"\n[{i}] ref_id={ref_id}")
        print(f"     title={title[:200] if title else ''}")
        if abstract:
            print(
                f"     abstract={abstract[:300].replace('\\n',' ')}{' …' if len(abstract) > 300 else ''}"
            )
