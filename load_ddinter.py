#!/usr/bin/env python3
"""RxSentinel — Phase 2: load drug-drug interactions into the `interactions` collection.

Two sources:
1. The full DDInter 2.0 export (recommended for the real system).
   DDInter 2.0 is free (CC BY-NC-SA 4.0): 2,310 drugs, 302,516 pairs.
   Get it from the Download page on https://ddinter2.scbdd.com/download/
   (the export is distributed as 8 per-ATC CSV files, ~13 MB in total
   as listed on the page).
   This loader auto-detects the export's columns, so exact header names
   do not matter — it looks for drug-a / drug-b / severity / mechanism /
   management columns by name.

Why this is fast: the old version sent one update_one() per pair —
302,516 separate round-trips to MongoDB (very slow on Atlas). This
version sends them in bulk_write() batches (default 2,000 per batch),
so one file is a few dozen round-trips instead of tens of thousands.
Converting CSV to JSON first does NOT speed this up by itself — the
slow part was never parsing, it was one network trip per row.
For the absolute fastest path, use --jsonl-out to write the normalised
pairs as JSON Lines, then load them with mongoimport (see below).
2. The curated demo seed (7 well-known pairs) in data/sample_interactions.csv,
   so the engine works and tests today:
       python load_ddinter.py --input data/sample_interactions.csv

Each pair is stored once, normalised and sorted:
    {drug_a, drug_b, severity, mechanism, management, source}
Severity is kept as Major / Moderate / Minor; rows with no usable severity
are stored as Unknown — never upgraded to safe. Severity "None" is the one
exception: a curated, source-labelled verdict that no known interaction
exists (data/curated_no_interaction.csv, verified against Drugs.com) — the
engine treats it as checked-and-clear, not as an interaction and not as
Unknown. Load the curated file AFTER the DDInter reload, so its None rows
are not overwritten by DDInter's ungraded Unknown rows for the same pairs.
Re-running is safe (upsert by drug pair).

Usage:
    python load_ddinter.py --input <ddinter-export.csv> [--mongo-uri ...] [--db rxsentinel]
    python load_ddinter.py --input data/sample_interactions.csv --dry-run
    # Fastest path (mongoimport, no Python loop at write time):
    python load_ddinter.py --input ddinter_downloads_code_A.csv --jsonl-out ddinter_A.jsonl
    mongoimport --uri "mongodb+srv://..." --db rxsentinel --collection interactions \
        --file ddinter_A.jsonl --mode upsert --upsertFields drug_a,drug_b
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

# pymongo is imported lazily inside write_pairs_bulk()/main() so the
# offline modes (--dry-run, --jsonl-out, and verify_ddinter.py's CSV
# preview) work on a machine without the driver installed.


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")


def _find(norm_map: dict[str, str], candidates: list[str]) -> str:
    for c in candidates:
        if c in norm_map:
            return c
    for norm in norm_map:
        if any(k in norm for k in candidates):
            return norm
    return ""


def _norm_drug(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


# DDInter uses international/US names; the Bangladesh dataset (and the
# engine) uses the names on local packs. Without this map the lookup
# misses pairs DDInter actually HAS — e.g. DDInter stores
# "Acetaminophen + Omeprazole" but the engine asks for
# "Paracetamol + Omeprazole" and gets no record.
# Keep this small and explicit: only names verified in the DDInter files.
DRUG_SYNONYMS = {
    "acetaminophen": "paracetamol",
    "acetylsalicylic acid": "aspirin",
}


# Same salt-suffix list as safety_engine.base_drug_name — DDInter names
# are already base names, so this is a no-op for them, but it keeps the
# loader and the engine from ever drifting apart on a name.
SALT_SUFFIXES = (
    "axetil", "trihydrate", "dihydrate", "hemihydrate", "monohydrate",
    "hydrochloride", "tromethamine", "fumarate", "maleate", "magnesium",
    "sodium", "calcium", "potassium", "proxetil", "mesylate", "besylate",
    "tartrate", "gluconate", "succinate", "hydrobromide", "nitrate",
    "bisulfate", "bisulphate",
)


def _base_drug_name(name: str) -> str:
    n = re.sub(r"\s*[\(\[].*?[\)\]]\s*$", "", (name or "").strip().lower())
    parts = re.sub(r"\s+", " ", n).split()
    while len(parts) > 1 and parts[-1] in SALT_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _canonical_drug(name: str) -> str:
    base = _base_drug_name(name)
    return DRUG_SYNONYMS.get(base, base)


def _norm_severity(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in ("major", "severe", "high"):
        return "Major"
    if s in ("moderate", "medium"):
        return "Moderate"
    if s in ("minor", "mild", "low"):
        return "Minor"
    if s in ("none", "no interaction", "no known interaction",
             "no interactions found"):
        # Curated "checked and clear" rows (see data/curated_no_interaction.csv).
        # "None" is a real verdict, not missing data: the engine treats it as
        # checked-with-no-finding instead of Unknown. Never collapse it to
        # Unknown, or the pair reads as ungraded again.
        return "None"
    return "Unknown"


def read_pairs(path: Path) -> tuple[list[dict], dict]:
    text = path.read_bytes().decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    if not rows:
        raise SystemExit("No data rows found in the input file.")
    fieldnames = list(rows[0].keys())
    norm_map = {_norm_col(c): c for c in fieldnames}
    col_a = _find(norm_map, ["drug_a", "drug1", "drug_1", "first_drug", "interacting_drug"])
    col_b = _find(norm_map, ["drug_b", "drug2", "drug_2", "second_drug", "interacted_drug"])
    col_sev = _find(norm_map, ["severity", "risk", "level", "risk_level"])
    col_mech = _find(norm_map, ["mechanism", "description", "detail"])
    col_mgmt = _find(norm_map, ["management", "recommendation", "advice"])
    col_src = _find(norm_map, ["source"])
    if not col_a or not col_b:
        raise SystemExit(
            "Could not find two drug columns. "
            f"Columns found: {fieldnames}")
    pairs: list[dict] = []
    stats = {"rows": len(rows), "skipped_no_drug": 0, "skipped_self_pair": 0,
             "unknown_severity": 0}
    for r in rows:
        a = _canonical_drug(str(r.get(norm_map[col_a], "") or ""))
        b = _canonical_drug(str(r.get(norm_map[col_b], "") or ""))
        if not a or not b:
            stats["skipped_no_drug"] += 1
            continue
        if a == b:
            stats["skipped_self_pair"] += 1
            continue
        sev = _norm_severity(str(r.get(norm_map[col_sev], "") or "")) if col_sev else "Unknown"
        if sev == "Unknown":
            stats["unknown_severity"] += 1
        a, b = sorted([a, b])
        pairs.append({
            "drug_a": a, "drug_b": b, "severity": sev,
            "mechanism": str(r.get(norm_map[col_mech], "") or "").strip() if col_mech else "",
            "management": str(r.get(norm_map[col_mgmt], "") or "").strip() if col_mgmt else "",
            "source": (str(r.get(norm_map[col_src], "") or "").strip()
                       if col_src else "DDInter 2.0"),
        })
    return pairs, stats


def _dedupe(pairs: list[dict]) -> list[dict]:
    """One entry per (drug_a, drug_b) — last row wins. Bulk upserts of
    the same pair in one batch can otherwise trip a duplicate-key error."""
    seen: dict[tuple[str, str], dict] = {}
    for p in pairs:
        seen[(p["drug_a"], p["drug_b"])] = p
    return list(seen.values())


def write_pairs_bulk(db, pairs: list[dict], batch_size: int) -> tuple[int, int]:
    """Write pairs with bulk_write() in batches. Returns (upserted, modified)."""
    from pymongo import UpdateOne

    upserted = modified = 0
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        ops = [UpdateOne({"drug_a": p["drug_a"], "drug_b": p["drug_b"]},
                         {"$set": p}, upsert=True) for p in batch]
        result = db.interactions.bulk_write(ops, ordered=False)
        upserted += result.upserted_count
        modified += result.modified_count
    return upserted, modified


def main() -> int:
    ap = argparse.ArgumentParser(description="RxSentinel Phase 2 — load interactions into MongoDB.")
    ap.add_argument("--input", required=True, help="DDInter export CSV or sample_interactions.csv.")
    ap.add_argument("--mongo-uri", default="mongodb://127.0.0.1:27017")
    ap.add_argument("--db", default="rxsentinel")
    ap.add_argument("--drop", action="store_true", help="Drop the interactions collection first.")
    ap.add_argument("--dry-run", action="store_true", help="Parse only, write nothing.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=2000,
                    help="Pairs per bulk_write batch (default 2000).")
    ap.add_argument("--jsonl-out", default="",
                    help="Write normalised pairs as JSON Lines to this file and exit (for mongoimport). No MongoDB write.")
    args = ap.parse_args()

    started = time.time()
    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"Input not found: {in_path}")
        return 1
    pairs, stats = read_pairs(in_path)
    if args.limit:
        pairs = pairs[: args.limit]
    pairs = _dedupe(pairs)
    print(f"Rows read: {stats['rows']}")
    print(f"Pairs: {len(pairs)}")
    print(f"Skipped (missing drug): {stats['skipped_no_drug']}")
    print(f"Skipped (same drug twice): {stats['skipped_self_pair']}")
    print(f"Unknown severity: {stats['unknown_severity']} (stored as Unknown, never as safe)")
    if args.dry_run:
        print("Dry run — nothing written.")
        return 0
    if args.jsonl_out:
        out = Path(args.jsonl_out)
        with out.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Wrote {len(pairs)} pairs to {out} in {time.time() - started:.1f}s")
        print("Load it with: mongoimport --uri <uri> --db rxsentinel "
              "--collection interactions --file "
              f"{out} --mode upsert --upsertFields drug_a,drug_b")
        return 0
    from pymongo import MongoClient

    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[args.db]
    if args.drop:
        db.interactions.drop()
        print("Dropped the interactions collection.")
    db.interactions.create_index([("drug_a", 1), ("drug_b", 1)], unique=True)
    upserted, modified = write_pairs_bulk(db, pairs, max(1, args.batch_size))
    print(f"Bulk wrote {len(pairs)} pairs in {time.time() - started:.1f}s "
          f"({upserted} new, {modified} updated)")
    print(f"Interactions in MongoDB '{args.db}': {db.interactions.count_documents({})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
