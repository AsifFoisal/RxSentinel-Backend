#!/usr/bin/env python3
"""RxSentinel — Phase 1: import the Global Medicine Directory CSV into MongoDB.

Reads the dataset Asif prepared (26,205 brands, Aug 2026, with the
`ingredients` column already added from Generic_Title) and loads it
into MongoDB as two collections, plus empty collections for the data
that arrives in later phases.

Collections
-----------
generics
    one document per generic (deduplicated by NORMALISED Generic_Title):
    {
      generic_id, title, ingredients,            # ingredients is a JSON
                                                  # array, e.g. ["Paracetamol", "Caffeine"]
      generic_url, indications, pharmacology,
      dosage_administration, interaction, side_effects,
      pregnancy_lactation, precautions_warnings,
      use_in_special_populations, therapeutic_class,
      storage_conditions, source, price_as_of
    }
    NOTE: Generic_Row_ID is NOT used as the key — the dataset reuses the
    same id for different generics (e.g. id 10 is both "Aceclofenac" and
    "Aloe + Black Pepper + Henbane"). Keying on the id merges different
    generics and keeps the wrong ingredients. The id is stored as a field
    only.
    Group titles are kept exactly as they are in the dataset, e.g.
    ["10 Vitamin & 6 Mineral [Pregnancy and Breast Feeding Formula]"]
    as ONE ingredient string. Nothing is split, nothing is skipped,
    nothing is invented — a group name is not a list of ingredients.

brands
    one document per brand row:
    {
      brand_name, generic_id, generic_title, ingredients,  # copied
      dosage_form, strength_raw, strength_parts,           # from the generic
      company, price_raw, unit_price_bdt, pack_price_bdt,
      source, price_as_of, medex_url
    }

interactions   empty for now; DDInter pairs land here in Phase 2.
recalls        empty for now; DGDA recall rows land here later.
aliases        seeded with verified aliases only
               (Paracetamol = Acetaminophen).

Price rule (decided Nov 14): the dataset price is stored WITH
`price_as_of` (default 2026-08-21, the dataset's date) and the MedEx
URL for a live check. A price is NEVER presented as live when it is
not. No live fetch happens at import time or at runtime.

Usage
-----
    pip install -r requirements.txt
    # dry run first — parses everything, writes nothing:
    python import_data.py --input data/global_medicine_directory_with_ingredients.csv --dry-run
    # then the real import (local MongoDB, or Atlas via --mongo-uri):
    python import_data.py --input data/global_medicine_directory_with_ingredients.csv
    python import_data.py --input data.csv --mongo-uri "mongodb+srv://..." --db rxsentinel --drop
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from pymongo import MongoClient


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")


# Normalised column names, in preference order.
GENERIC_ID_COLS = ["generic_row_id", "generic_id", "genericid"]
GENERIC_TITLE_COLS = ["generic_title", "generic", "generic_name"]
BRAND_COLS = ["brand_name", "brand"]
INGREDIENTS_COLS = ["ingredients", "ingredients_json"]


def _find(norm_map: dict[str, str], candidates: list[str]) -> str:
    for c in candidates:
        if c in norm_map:
            return c
    return ""


def _pick(row: dict, norm_map: dict[str, str], candidates: list[str]) -> str:
    col = _find(norm_map, candidates)
    return str(row.get(norm_map[col], "") or "").strip() if col else ""


def clean_ingredient(part: str) -> str:
    """'Paracetamol 500 mg' -> 'Paracetamol' (strengths are not ingredients)."""
    part = re.sub(r"\s+", " ", part.strip())
    part = re.sub(r"\s+\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|iu|%)(?:\s*/\s*\d+\s*(?:mg|ml))?\s*$",
                  "", part, flags=re.I)
    part = re.sub(r"\s+\d+(?:\.\d+)?\s*$", "", part)
    return part.strip()


def parse_ingredients(raw: str, title: str) -> list[str]:
    """Parse the dataset's `ingredients` value into a real list.

    The dataset's column is a JSON array string, e.g.
    ["Paracetamol", "Caffeine"] — including group titles kept as ONE
    string, e.g. ["10 Vitamin & 6 Mineral [Pregnancy and Breast
    Feeding Formula]"]. That is honoured as-is: one element, no split.
    Fallbacks (old files without the column): split the title on "+".
    """
    raw = (raw or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
            if isinstance(parsed, str):
                return [parsed.strip()] if parsed.strip() else []
        except json.JSONDecodeError:
            pass
        # Not JSON — a plain "A | B" or "A + B" string: split, don't guess.
        if " | " in raw:
            return [p.strip() for p in raw.split(" | ") if p.strip()]
        if " + " in raw:
            return [clean_ingredient(p) for p in raw.split("+") if clean_ingredient(p)]
        return [raw]
    if not title:
        return []
    if "+" in title:
        return [clean_ingredient(p) for p in title.split("+") if clean_ingredient(p)]
    return [title.strip()]


def parse_price(raw: str) -> tuple[float | None, float | None]:
    """'৳ 4.51 (10's pack: ৳ 45.10)' -> (4.51, 45.10). Returns Nones if absent.

    Only numbers carrying the ৳/Tk mark count as prices — the '10' in
    "10's pack" is a quantity, not a price, and must not be mistaken
    for the pack price.
    """
    if not raw:
        return None, None
    amounts = re.findall(r"[৳]\s*(\d+(?:\.\d+)?)|Tk\.?\s*(\d+(?:\.\d+)?)", raw)
    nums = [float(a or b) for a, b in amounts if a or b]
    if not nums:
        nums = [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)", raw)]
        return (nums[0], nums[1] if len(nums) > 1 else None) if nums else (None, None)
    return nums[0], nums[1] if len(nums) > 1 else None


def parse_strength(raw: str) -> list[str]:
    """'500 mg + 65 mg' -> ['500 mg', '65 mg']; 'N/A' -> []."""
    if not raw or raw.strip().upper() in ("N/A", "NA", "-"):
        return []
    return [p.strip() for p in raw.split("+") if p.strip()]


# Verified aliases only — add more only with a source.
SEED_ALIASES = [
    {"canonical": "Paracetamol", "aliases": ["Acetaminophen"]},
]

MONOGRAPH_FIELDS = [
    ("indications", ["indications"]),
    ("pharmacology", ["pharmacology"]),
    ("dosage_administration", ["dosage_administration", "dosage_and_administration"]),
    ("interaction", ["interaction"]),
    ("side_effects", ["side_effects", "side_effect"]),
    ("pregnancy_lactation", ["pregnancy_lactation", "pregnancy_and_lactation"]),
    ("precautions_warnings", ["precautions_warnings", "precautions_and_warnings"]),
    ("use_in_special_populations", ["use_in_special_populations"]),
    ("therapeutic_class", ["therapeutic_class"]),
    ("storage_conditions", ["storage_conditions"]),
]


def read_rows(path: Path) -> tuple[list[dict], list[str], dict[str, str]]:
    text = path.read_bytes().decode("utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    if not rows:
        raise SystemExit("No data rows found in the input file.")
    fieldnames = list(rows[0].keys())
    return rows, fieldnames, {_norm_col(c): c for c in fieldnames}


def _normalised_ings(ings: list[str]) -> list[str]:
    """Ingredients reduced to a comparable form: lower-cased, strengths
    stripped, sorted. Two lists that differ only by order, case, spacing
    or a stray '500 mg' in the name collapse to the same thing."""
    return sorted(clean_ingredient(i).lower() for i in ings if clean_ingredient(i))


def classify_conflict(kept: list[str], other: list[str]) -> str:
    """One-line verdict on why two ingredient lists for one generic differ."""
    if sorted(k.lower() for k in kept) == sorted(o.lower() for o in other):
        return "order_or_case_only"
    if _normalised_ings(kept) == _normalised_ings(other):
        return "strength_in_name_only"
    if set(_normalised_ings(kept)) & set(_normalised_ings(other)):
        return "overlapping_but_different"
    return "different_ingredients"


def build_documents(rows: list[dict], norm_map: dict[str, str], price_as_of: str):
    """Turn CSV rows into (generics, brands, conflicts) — pure, no DB.

    Dedupe key is the NORMALISED GENERIC TITLE, not Generic_Row_ID:
    the dataset reuses the same Generic_Row_ID for different generics
    (e.g. id 10 is both "Aceclofenac" and "Aloe + Black Pepper +
    Henbane"), so keying on the id merges different generics into one
    document and keeps the wrong ingredients. The id is stored as a
    field only. Conflicts are therefore per TITLE: same title, different
    ingredients across rows. Returns (generics, brands, stats, conflicts).
    """
    generic_id_col = _find(norm_map, GENERIC_ID_COLS)
    title_col = _find(norm_map, GENERIC_TITLE_COLS)
    brand_col = _find(norm_map, BRAND_COLS)
    ing_col = _find(norm_map, INGREDIENTS_COLS)
    url_col = _find(norm_map, ["generic_url", "source_url", "medex_url", "url"])
    source_col = _find(norm_map, ["data_source", "source"])
    if not title_col or not brand_col:
        raise SystemExit(
            "Could not find generic-title / brand columns. "
            f"Columns found: {list(norm_map.values())}")

    generics: dict[str, dict] = {}
    brands: list[dict] = []
    conflicts: dict[str, dict] = {}
    id_to_titles: dict[str, set] = {}
    title_to_ids: dict[str, set] = {}
    stats = {"rows": len(rows), "no_generic_id": 0, "no_ingredients": 0,
             "ingredient_conflicts": 0, "no_price": 0,
             "shared_ids": 0, "title_id_mismatches": 0}
    for r in rows:
        gid = str(r.get(norm_map[generic_id_col], "") or "").strip() if generic_id_col else ""
        title = str(r.get(norm_map[title_col], "") or "").strip()
        brand = str(r.get(norm_map[brand_col], "") or "").strip()
        if not brand:
            continue
        ing_raw = str(r.get(norm_map[ing_col], "") or "") if ing_col else ""
        ings = parse_ingredients(ing_raw, title)
        if not ings:
            stats["no_ingredients"] += 1
        # Key on the normalised title — Generic_Row_ID is NOT unique per
        # generic in this dataset (same id, different titles).
        key = re.sub(r"\s+", " ", title.lower()).strip() or f"noid:{gid}"
        if not gid:
            stats["no_generic_id"] += 1
        if gid:
            id_to_titles.setdefault(gid, set()).add(key)
            title_to_ids.setdefault(key, set()).add(gid)
        if key not in generics:
            doc: dict = {
                "generic_id": gid or None,
                "title": title,
                "ingredients": ings,
                "generic_url": str(r.get(norm_map[url_col], "") or "").strip() if url_col else "",
                "source": str(r.get(norm_map[source_col], "") or "").strip() if source_col else "",
                "price_as_of": price_as_of,
            }
            for field, cands in MONOGRAPH_FIELDS:
                doc[field] = _pick(r, norm_map, cands)
            generics[key] = doc
        elif generics[key]["ingredients"] != ings:
            # Same generic id, different ingredients in different rows:
            # keep the first, record the disagreement — never silently merge.
            stats["ingredient_conflicts"] += 1
            entry = conflicts.setdefault(key, {
                "generic_id": gid or "", "title": title,
                "kept": generics[key]["ingredients"], "others": [],
                "example_brands": [], "row_count": 0})
            other_json = json.dumps(ings, ensure_ascii=False)
            if other_json not in [json.dumps(o, ensure_ascii=False) for o in entry["others"]]:
                entry["others"].append(ings)
            if brand not in entry["example_brands"] and len(entry["example_brands"]) < 5:
                entry["example_brands"].append(brand)

        strength_raw = str(r.get(norm_map.get("strength", ""), "") or "").strip()
        price_raw = str(r.get(norm_map.get("price_pack_size", ""), "") or "").strip()
        unit, pack = parse_price(price_raw)
        if unit is None:
            stats["no_price"] += 1
        brands.append({
            "brand_name": brand,
            "generic_id": gid or None,
            "generic_title": title,
            "ingredients": ings,
            "dosage_form": str(r.get(norm_map.get("dosage_form", ""), "") or "").strip(),
            "strength_raw": strength_raw,
            "strength_parts": parse_strength(strength_raw),
            "company": str(r.get(norm_map.get("company", ""), "") or "").strip(),
            "medicine_type": str(r.get(norm_map.get("medicine_type", ""), "") or "").strip(),
            "price_raw": price_raw,
            "unit_price_bdt": unit,
            "pack_price_bdt": pack,
            "price_as_of": price_as_of,
            "source": str(r.get(norm_map[source_col], "") or "").strip() if source_col else "",
            "medex_url": str(r.get(norm_map[url_col], "") or "").strip() if url_col else "",
        })
    # Total rows per conflicted generic, counted in a second pass so the
    # number is exact (not "rows seen after the first disagreement").
    from collections import Counter
    row_counts = Counter(
        re.sub(r"\s+", " ", str(r.get(norm_map[title_col], "") or "").strip().lower())
        for r in rows if str(r.get(norm_map[brand_col], "") or "").strip())
    for key, entry in conflicts.items():
        entry["row_count"] = row_counts.get(key, 0)
    stats["shared_ids"] = sum(1 for titles in id_to_titles.values() if len(titles) > 1)
    stats["title_id_mismatches"] = sum(1 for ids in title_to_ids.values() if len(ids) > 1)
    return list(generics.values()), brands, stats, list(conflicts.values())


def write_conflicts_report(conflicts: list[dict], path: Path) -> dict[str, int]:
    """Write one row per conflicted generic. Returns the verdict counts."""
    counts: dict[str, int] = {}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "generic_id", "generic_title", "ingredients_kept",
            "other_ingredients", "verdict", "example_brands", "row_count"])
        w.writeheader()
        for c in sorted(conflicts, key=lambda x: (x["generic_id"], x["title"])):
            verdicts = [classify_conflict(c["kept"], o) for o in c["others"]]
            verdict = "different_ingredients" if "different_ingredients" in verdicts \
                else ("overlapping_but_different" if "overlapping_but_different" in verdicts
                      else verdicts[0])
            counts[verdict] = counts.get(verdict, 0) + 1
            w.writerow({
                "generic_id": c["generic_id"],
                "generic_title": c["title"],
                "ingredients_kept": json.dumps(c["kept"], ensure_ascii=False),
                "other_ingredients": " || ".join(
                    json.dumps(o, ensure_ascii=False) for o in c["others"]),
                "verdict": verdict,
                "example_brands": ", ".join(c["example_brands"]),
                "row_count": c["row_count"],
            })
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="RxSentinel Phase 1 — import dataset CSV into MongoDB.")
    ap.add_argument("--input", required=True, help="Dataset CSV (with the ingredients column).")
    ap.add_argument("--mongo-uri", default="mongodb://127.0.0.1:27017",
                    help="MongoDB connection string.")
    ap.add_argument("--db", default="rxsentinel", help="Database name.")
    ap.add_argument("--price-as-of", default="2026-08-21",
                    help="Date the dataset prices are from (YYYY-MM-DD).")
    ap.add_argument("--drop", action="store_true", help="Drop the collections before importing.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse everything and print the summary, write nothing.")
    ap.add_argument("--limit", type=int, default=0, help="Import only the first N rows (testing).")
    ap.add_argument("--conflicts-report", default="",
                    help="CSV listing generics whose rows disagreed on ingredients. "
                         "Default: <input-stem>_ingredient_conflicts.csv next to the input.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"Input not found: {in_path}")
        return 1
    rows, fieldnames, norm_map = read_rows(in_path)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Columns found: {fieldnames}")
    generics, brands, stats, conflicts = build_documents(rows, norm_map, args.price_as_of)

    n_combo = sum(1 for g in generics if len(g["ingredients"]) > 1)
    n_group = sum(1 for g in generics
                  if len(g["ingredients"]) == 1 and ("[" in g["ingredients"][0] or "&" in g["ingredients"][0]))
    print(f"\nRows read:                 {stats['rows']}")
    print(f"Distinct generics:         {len(generics)}")
    print(f"  of which combinations:   {n_combo}")
    print(f"  of which group titles:   {n_group}  (kept as one ingredient string, as in the dataset)")
    print(f"Brand documents:           {len(brands)}")
    print(f"Rows with no ingredients:  {stats['no_ingredients']}")
    print(f"Rows with no generic id:   {stats['no_generic_id']}")
    print(f"Ingredient conflicts:      {stats['ingredient_conflicts']} (same generic title, different ingredients — first kept)")
    print(f"Brands with no price:      {stats['no_price']}")
    print(f"Generic ids shared by 2+ titles: {stats['shared_ids']} (id is not a reliable key — deduped by title instead)")
    print(f"Titles with 2+ different ids:    {stats['title_id_mismatches']} (first id kept)")

    if conflicts:
        report_path = Path(args.conflicts_report) if args.conflicts_report else in_path.with_name(
            f"{in_path.stem}_ingredient_conflicts{in_path.suffix}")
        verdicts = write_conflicts_report(conflicts, report_path)
        print(f"\nConflicts report: {len(conflicts)} generics -> {report_path}")
        for v in ("order_or_case_only", "strength_in_name_only",
                  "overlapping_but_different", "different_ingredients"):
            if verdicts.get(v):
                print(f"  {v}: {verdicts[v]}")
        print("  Harmless = order/case or strength-in-name. "
              "'different_ingredients' needs a look before Phase 2.")
    else:
        print("\nNo ingredient conflicts — nothing to report.")

    if args.dry_run:
        print("\nDry run — nothing written.")
        return 0

    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[args.db]
    if args.drop:
        for name in ("generics", "brands", "interactions", "recalls", "aliases"):
            db[name].drop()
        print("Dropped existing collections.")
    db.generics.create_index("generic_id", unique=False)
    db.generics.create_index("title")
    db.brands.create_index("generic_id")
    db.brands.create_index("brand_name")
    db.brands.create_index("ingredients")

    # One upsert at a time (not bulk_write): 26k rows is still fast,
    # and it works the same against real MongoDB and test doubles.
    for g in generics:
        db.generics.update_one({"title": g["title"]},
                               {"$set": g}, upsert=True)
    for b in brands:
        db.brands.update_one({"brand_name": b["brand_name"], "generic_id": b["generic_id"],
                              "strength_raw": b["strength_raw"]},
                             {"$set": b}, upsert=True)
    for a in SEED_ALIASES:
        db.aliases.update_one({"canonical": a["canonical"]}, {"$set": a}, upsert=True)
    # Empty collections for later phases (created by the first write above
    # for generics/brands; interactions/recalls are created here).
    if "interactions" not in db.list_collection_names():
        db.create_collection("interactions")
    if "recalls" not in db.list_collection_names():
        db.create_collection("recalls")

    print(f"\nImported into MongoDB '{args.db}': "
          f"{db.generics.count_documents({})} generics, "
          f"{db.brands.count_documents({})} brands, "
          f"{db.aliases.count_documents({})} aliases.")
    print("interactions and recalls collections are empty — Phase 2 fills interactions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
