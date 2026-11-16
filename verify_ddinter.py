#!/usr/bin/env python3
"""RxSentinel — verify the DDInter load after the Paracetamol/Aspirin name fix.

Two modes:
  1. MongoDB check (default): count pairs, confirm no leftover
     "acetaminophen" / "acetylsalicylic acid" names, and show the
     severity of the pairs you tested with (Napa Extra + Seclo).
  2. CSV preview (--csv FILE): parse a DDInter CSV with the same
     loader logic, without touching MongoDB, and report how many
     rows get renamed. Run this BEFORE the reload to sanity-check.

Usage:
  python verify_ddinter.py --mongo-uri "mongodb+srv://..." 
  python verify_ddinter.py --csv ddinter_downloads_code_A.csv --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse the loader's own parsing / name-map so this check can never
# drift from what load_ddinter.py actually does on import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_ddinter import read_pairs, _dedupe  # noqa: E402

EXPECTED_FULL_COUNT = 160_235  # verified 2026-11-14 from the 8 DDInter files
OLD_NAMES = ("acetaminophen", "acetylsalicylic acid")
# Pairs from the Napa Extra + Seclo test (ingredients: Paracetamol,
# Caffeine, Omeprazole). DDInter lists these as Unknown severity —
# found, but ungraded. After the curated no-interaction file is loaded
# (AFTER the DDInter reload), they are stored as "None" instead —
# verified no known interaction — which is also a PASS.
TEST_PAIRS = [
    ("paracetamol", "caffeine"),
    ("caffeine", "omeprazole"),
    ("omeprazole", "paracetamol"),
]


def check_csv(path: Path) -> int:
    pairs, stats = read_pairs(path)
    deduped = _dedupe(pairs)
    renamed = sum(
        1 for p in pairs
        if p["drug_a"] in OLD_NAMES or p["drug_b"] in OLD_NAMES
    )
    # After _canonical_drug, no pair should still carry an old name.
    leftover = [p for p in deduped if p["drug_a"] in OLD_NAMES or p["drug_b"] in OLD_NAMES]
    print(f"CSV: {path}")
    print(f"  rows read:            {stats['rows']:,}")
    print(f"  after dedupe:         {len(deduped):,}")
    print(f"  renamed by name-map:  {renamed:,} rows used acetaminophen/aspirin-acid names")
    print(f"  leftover old names:   {len(leftover):,}  (must be 0)")
    for a, b in TEST_PAIRS:
        hit = next((p for p in deduped if {p["drug_a"], p["drug_b"]} == {a, b}), None)
        if hit:
            note = ("Unknown = found but ungraded by DDInter" if hit['severity'] == "Unknown"
                    else "None = verified no known interaction (curated layer)" if hit['severity'] == "None"
                    else "graded by DDInter")
            print(f"  FOUND {a} + {b}: severity={hit['severity']} ({note})")
        else:
            print(f"  MISSING {a} + {b}: no record in this file (may be in another ATC file)")
    return 0 if not leftover else 1


def check_mongo(uri: str, db_name: str) -> int:
    from pymongo import MongoClient

    db = MongoClient(uri)[db_name]
    coll = db["interactions"]
    total = coll.estimated_document_count()
    print(f"MongoDB: {db_name}.interactions")
    print(f"  total pairs: {total:,}")
    if total == EXPECTED_FULL_COUNT:
        print(f"  -> matches the expected full DDInter load ({EXPECTED_FULL_COUNT:,}).")
    elif total < EXPECTED_FULL_COUNT:
        print(f"  -> below {EXPECTED_FULL_COUNT:,}: partial load, demo seed only, or not reloaded yet.")

    leftover = coll.count_documents(
        {"$or": [{"drug_a": {"$in": list(OLD_NAMES)}}, {"drug_b": {"$in": list(OLD_NAMES)}}]}
    )
    print(f"  leftover old names (acetaminophen / acetylsalicylic acid): {leftover:,}  (must be 0 after reload)")
    status = 0 if leftover == 0 else 1

    for drug in ("paracetamol", "aspirin"):
        n = coll.count_documents({"$or": [{"drug_a": drug}, {"drug_b": drug}]})
        print(f"  pairs involving {drug}: {n:,}")

    for a, b in TEST_PAIRS:
        hit = coll.find_one({"drug_a": a, "drug_b": b}) or coll.find_one({"drug_a": b, "drug_b": a})
        if hit:
            sev = hit.get('severity', '?')
            note = ("Unknown = found but ungraded — correct, not a failure" if sev == "Unknown"
                    else "None = verified no known interaction (curated layer)" if sev == "None"
                    else "graded by DDInter")
            print(f"  FOUND {a} + {b}: severity={sev} ({note})")
        else:
            print(f"  MISSING {a} + {b}: still no record — engine will keep reporting Unknown for this pair")
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify RxSentinel DDInter load.")
    ap.add_argument("--mongo-uri", default="mongodb://127.0.0.1:27017")
    ap.add_argument("--db", default="rxsentinel")
    ap.add_argument("--csv", default="", help="Preview a DDInter CSV without MongoDB.")
    ap.add_argument("--dry-run", action="store_true", help="Accepted for symmetry with load_ddinter.py; --csv never writes.")
    args = ap.parse_args()
    if args.csv:
        return check_csv(Path(args.csv))
    return check_mongo(args.mongo_uri, args.db)


if __name__ == "__main__":
    raise SystemExit(main())
