"""
Add metadata (title, doi, pdf_link) and references (ref_id, num_refs) to SLR JSON dataset.
"""

import json
import csv
import re
import ast
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple

# ===================== CONFIG =====================
INPUT_JSON_PATH = Path("../../final/slr_dataset_final.json")
INPUT_META_CSV = Path(
    "../_tmp/full_paper_final/dblp_systematic_review_2000_2025_filtered_final_03_07_25.csv"
)
INPUT_REFS_CSV = Path("../../refs/filtered_slr2ref_map_final.csv")
OUTPUT_JSON_PATH = Path("../../final/slr_dataset_final.json")
# ==================================================


def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _to_int(s: Any) -> Optional[int]:
    if s is None:
        return None
    if isinstance(s, int):
        return s
    s = str(s).strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_ref_id_to_int_list(raw: Any) -> Optional[List[int]]:
    """Parse ref_id like '[1, 2, 3]' or '1,2,3'; return list of ints or None."""
    if raw is None:
        return None
    if isinstance(raw, list):
        out = []
        for x in raw:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out or None
    text = str(raw).strip()
    if not text:
        return None
    # Try literal list
    if text.startswith("[") and text.endswith("]"):
        try:
            val = ast.literal_eval(text)
            if isinstance(val, list):
                out = []
                for x in val:
                    try:
                        out.append(int(x))
                    except Exception:
                        continue
                return out or None
        except Exception:
            pass
    # Fallback: pull all integers in order
    nums = re.findall(r"-?\d+", text)
    return [int(n) for n in nums] if nums else None


def load_meta_csv(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Return id -> {title, doi, pdf_link} from a proper CSV."""
    id_map: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            rid = _norm(row.get("id"))
            if not rid:
                continue
            id_map[rid] = {
                "title": _norm(row.get("title")),
                "doi": _norm(row.get("doi")),
                "pdf_link": _norm(row.get("pdf_link")),
            }
    return id_map


def try_read_refs_as_csv(f) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """Try to read sr_refs.csv as a proper CSV. Return (ok, map)."""
    f.seek(0)
    try:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return False, {}
        # Normalize headers
        headers = [h.strip().lower() for h in reader.fieldnames]
        if not {"id", "ref_id", "num_refs"}.issubset(set(headers)):
            # Not the headers we need
            return False, {}
        id_idx = headers.index("id")
        ref_idx = headers.index("ref_id")
        num_idx = headers.index("num_refs")
        f.seek(0)
        reader = csv.reader(f)
        next(reader, None)  # skip header
        out: Dict[str, Dict[str, Any]] = {}
        for row in reader:
            try:
                rid = _norm(row[id_idx])
            except Exception:
                continue
            if not rid:
                continue
            ref_raw = row[ref_idx] if ref_idx < len(row) else ""
            num_raw = row[num_idx] if num_idx < len(row) else ""
            ref_list = parse_ref_id_to_int_list(ref_raw)
            n_refs = _to_int(num_raw)
            out[rid] = {"ref_id": ref_list, "num_refs": n_refs}
        return True, out
    except Exception:
        return False, {}


def read_refs_with_fallback(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Read sr_refs.csv. If it's not a proper CSV with needed headers, fall back to
    parsing each line with regex:  ^(id-digits).*?(\[...]).*?(num_refs-digits)\s*$
    """
    # First attempt: proper CSV
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        ok, out = try_read_refs_as_csv(f)
        if ok:
            return out

    # Fallback: loose-line parsing
    out: Dict[str, Dict[str, Any]] = {}
    line_re = re.compile(
        r"""^\s*
            (?P<id>\d+)\s*         # leading id
            (?:[^\[]*?)            # anything up to first '[' (non-greedy)
            (?P<bracket>\[[^\]]*\])# bracketed list
            (?:.*?)(?P<num>\d+)\s*$# trailing number
        """,
        re.VERBOSE,
    )

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.lower().startswith("id"):
                continue
            m = line_re.match(line)
            if not m:
                # Try a simpler pattern: id then numbers; take last number as num_refs
                simple_nums = re.findall(r"\d+", line)
                if not simple_nums:
                    continue
                rid = simple_nums[0]
                if len(simple_nums) >= 2:
                    n_refs = int(simple_nums[-1])
                    # everything between first and last numbers as potential list
                    # extract bracket content if any; else collect middle ints
                    br = re.search(r"\[([^\]]*)\]", line)
                    if br:
                        ref_list = parse_ref_id_to_int_list("[" + br.group(1) + "]")
                    else:
                        mid = simple_nums[1:-1]
                        ref_list = [int(x) for x in mid] if mid else None
                else:
                    n_refs, ref_list = None, None
            else:
                rid = m.group("id")
                bracket = m.group("bracket")
                n_refs = _to_int(m.group("num"))
                ref_list = parse_ref_id_to_int_list(bracket)

            out[rid] = {
                "ref_id": ref_list if (ref_list and len(ref_list) > 0) else None,
                "num_refs": n_refs,
            }
    return out


def main():
    # Load JSON list
    with INPUT_JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list of objects.")

    # Load CSV maps
    meta_map = load_meta_csv(INPUT_META_CSV)
    refs_map = read_refs_with_fallback(INPUT_REFS_CSV)

    matched_meta, matched_refs = 0, 0
    unmatched_refs = []  ### collect IDs with no refs

    for obj in data:
        rid = _norm(str(obj.get("id", "")))
        if not rid:
            continue

        # Metadata join
        meta = meta_map.get(rid)
        if meta:
            if meta.get("title"):
                obj["sr_title"] = meta["title"]
            if meta.get("doi"):
                obj["sr_doi"] = meta["doi"]
            if meta.get("pdf_link"):
                obj["sr_pdf_link"] = meta["pdf_link"]
            matched_meta += 1

        # Refs join (only set when we have something real)
        refs = refs_map.get(rid)
        if refs:
            ref_list = refs.get("ref_id")
            n_refs = refs.get("num_refs")
            if ref_list:
                obj["ref_id"] = ref_list
            if n_refs is not None:
                obj["num_refs"] = n_refs
            matched_refs += 1
        else:
            unmatched_refs.append(rid)  ### mark as unmatched

    # Save enriched JSON
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"Done. Entries: {len(data)} | Matched meta: {matched_meta} | Matched refs: {matched_refs}"
    )
    print(f"Wrote: {OUTPUT_JSON_PATH.resolve()}")
    print("Unmatched refs IDs:", unmatched_refs)  ### print list


if __name__ == "__main__":
    main()
