#!/usr/bin/env python3
"""RxSentinel — audit interaction coverage for the most common medicines.

Answers "how many common medicines are missing from DDInter?" with a
number, instead of guessing. It takes the top N generics by brand count
from the Phase 1 MongoDB data, builds every ingredient pair between
them, and checks each pair against the `interactions` collection.

Output:
  - a printed summary: pairs checked, with data, missing (no record),
    and found-but-Ungraded (DDInter severity Unknown)
  - a CSV of the pairs that need a verified verdict, most-common first,
    so the curated layer can be filled in batches from a trusted source
    (Drugs.com interaction checker — Major / Moderate / Minor, same
    scale as DDInter) and loaded with load_ddinter.py

Nothing is written to MongoDB. The missing-pairs CSV is the work list:
each row still needs a named source before it goes into the database —
never invent a severity for a pair, a wrong "None" hides a real
interaction and a wrong "Major" creates a false alarm.

Usage:
    python audit_coverage.py --mongo-uri "mongodb+srv://..." [--top-n 50]
    python audit_coverage.py --mongo-uri "mongodb+srv://..." --top-n 100 --out missing_pairs.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


# Same name map as load_ddinter.py / the engine's alias table:
# DDInter uses the US name, the Bangladesh dataset uses the pack name.
SYNONYMS = {
    "acetaminophen": "paracetamol",
    "acetylsalicylic acid": "aspirin",
}


SALT_SUFFIXES = (
    "axetil", "trihydrate", "dihydrate", "hemihydrate", "monohydrate",
    "hydrochloride", "tromethamine", "fumarate", "maleate", "magnesium",
    "sodium", "calcium", "potassium", "proxetil", "mesylate", "besylate",
    "tartrate", "gluconate", "succinate", "hydrobromide", "nitrate",
    "bisulfate", "bisulphate",
)


def _canonical(name: str) -> str:
    # Same salt-strip as safety_engine.base_drug_name: "Cefuroxime
    # axetil" and "Cefuroxime" are the same drug for lookup purposes.
    n = re.sub(r"\s*[\(\[].*?[\)\]]\s*$", "", (name or "").strip().lower())
    parts = re.sub(r"\s+", " ", n).split()
    while len(parts) > 1 and parts[-1] in SALT_SUFFIXES:
        parts.pop()
    base = " ".join(parts)
    return SYNONYMS.get(base, base)


def audit(db, top_n: int):
    """Return (summary dict, missing rows). `db` is a pymongo database."""
    # Brand count per generic, and each generic's ingredients.
    generic_brands: Counter = Counter()
    generic_ings: dict[str, list[str]] = {}
    for b in db.brands.find({}, {"generic_title": 1, "ingredients": 1}):
        title = (b.get("generic_title") or "").strip()
        if not title:
            continue
        generic_brands[title] += 1
        if title not in generic_ings:
            generic_ings[title] = list(b.get("ingredients") or [])

    top = generic_brands.most_common(top_n)
    print(f"Top {len(top)} generics by brand count:")
    for title, count in top[:10]:
        print(f"  {count:5d} brands  {title}")

    # Ingredient universe of the top generics, with brand counts per ingredient.
    ing_brands: Counter = Counter()
    for title, _ in top:
        for ing in generic_ings.get(title, []):
            ing_brands[_canonical(ing)] += generic_brands[title]
    ingredients = sorted(ing_brands)

    pairs = list(combinations(ingredients, 2))
    with_data = missing = ungraded = 0
    rows = []
    for a, b in pairs:
        doc = db.interactions.find_one({"drug_a": a, "drug_b": b})
        if doc is None:
            missing += 1
            rows.append({
                "drug_a": a, "drug_b": b, "status": "no_record",
                "brands_a": ing_brands[a], "brands_b": ing_brands[b],
                "severity": "", "source": "",
            })
        else:
            sev = doc.get("severity", "Unknown")
            if sev == "Unknown":
                ungraded += 1
                rows.append({
                    "drug_a": a, "drug_b": b, "status": "ungraded_unknown",
                    "brands_a": ing_brands[a], "brands_b": ing_brands[b],
                    "severity": "Unknown", "source": doc.get("source", "DDInter 2.0"),
                })
            else:
                with_data += 1
    # Most common pairs first — fill those first.
    rows.sort(key=lambda r: r["brands_a"] + r["brands_b"], reverse=True)
    summary = {
        "top_generics": len(top),
        "ingredients": len(ingredients),
        "pairs_checked": len(pairs),
        "pairs_with_data": with_data,
        "pairs_missing": missing,
        "pairs_ungraded": ungraded,
    }
    return summary, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mongo-uri", default="mongodb://127.0.0.1:27017")
    ap.add_argument("--db", default="rxsentinel")
    ap.add_argument("--top-n", type=int, default=50,
                    help="Top N generics by brand count (default 50).")
    ap.add_argument("--out", default="missing_pairs.csv",
                    help="CSV work list of pairs needing a verified verdict.")
    args = ap.parse_args()

    from pymongo import MongoClient

    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[args.db]
    try:
        total = db.interactions.count_documents({})
    except Exception as exc:  # noqa: BLE001 - report plainly, don't guess
        print(f"Cannot reach MongoDB '{args.db}': {exc}")
        return 1
    print(f"Interactions in MongoDB '{args.db}': {total:,}")
    if total == 0:
        print("Interactions collection is empty — load DDInter first.")
        return 1

    summary, rows = audit(db, args.top_n)
    print()
    print(f"Ingredients in top {summary['top_generics']} generics: {summary['ingredients']}")
    print(f"Pairs checked:      {summary['pairs_checked']:,}")
    print(f"Pairs with data:    {summary['pairs_with_data']:,}")
    print(f"Pairs missing:      {summary['pairs_missing']:,}")
    print(f"Pairs ungraded:     {summary['pairs_ungraded']:,}")
    if summary["pairs_checked"]:
        cov = 100 * summary["pairs_with_data"] / summary["pairs_checked"]
        print(f"Coverage:           {cov:.1f}% of common pairs have a graded verdict")
    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "drug_a", "drug_b", "status", "brands_a", "brands_b",
            "severity", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} pairs needing a verdict to {out}")
    print("Next: verify each row against a trusted checker (Drugs.com), "
          "fill severity + source, then load with load_ddinter.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
