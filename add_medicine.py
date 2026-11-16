#!/usr/bin/env python3
"""RxSentinel — add medicines the starter dataset does not carry.

The Kaggle Global Medicine Directory (26,205 brands) is not complete
for Bangladesh: Plavix/Clopidogrel, Enzoflam, Pan D and Hexigel are all
genuinely absent (verified with safety_engine.py --check-brand), so a
prescription naming them lands in needs_confirmation no matter how
good the OCR is. This script adds them one at a time, with the
composition verified from a named source first — never from memory
alone, and never guessed.

Each entry in data/new_medicines.json carries its source URL; the
brand doc keeps that URL so the provenance is checkable later.

Safety rules:
- A brand that already resolves (exact, case-insensitive, or
  space/hyphen-tolerant — the same matching the intake uses) is
  skipped, never overwritten: the bulk import stays the authority
  for rows it already has.
- A generic title that already exists is left alone; a missing one
  is created so the generic-fallback path works too.
- --dry-run (default) prints what would change and writes nothing.
  Run with --apply to write. Re-running --apply is safe (upserts).

Run:  python add_medicine.py --dry-run
      python add_medicine.py --apply
      python add_medicine.py --json data/new_medicines.json --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from safety_engine import DrugDB  # noqa: E402
from web_verify import _connect_db, _load_dotenv  # noqa: E402

DEFAULT_JSON = Path(__file__).resolve().parent / "data" / "new_medicines.json"


def validate_entry(entry: dict) -> dict:
    """Check one JSON entry and normalise it. Raises ValueError with
    the problem — a bad row stops the run instead of writing a
    half-formed brand into the database."""
    if not isinstance(entry, dict):
        raise ValueError(f"entry is not an object: {entry!r}")
    brand = str(entry.get("brand") or "").strip()
    title = str(entry.get("generic_title") or "").strip()
    ings = [str(i).strip() for i in (entry.get("ingredients") or [])
            if str(i).strip()]
    parts = [str(p).strip() for p in (entry.get("strength_parts") or [])
             if str(p).strip()]
    if not brand:
        raise ValueError(f"entry has no brand: {entry!r}")
    if not title:
        raise ValueError(f"{brand}: no generic_title")
    if not ings:
        raise ValueError(f"{brand}: no ingredients")
    if parts and len(parts) != len(ings):
        raise ValueError(
            f"{brand}: {len(parts)} strength_parts for {len(ings)} "
            f"ingredients — they must line up one-to-one (or be empty).")
    return {
        "brand": brand,
        "generic_title": title,
        "ingredients": ings,
        "strength_parts": parts,
        "dosage_form": str(entry.get("dosage_form") or "").strip(),
        "company": str(entry.get("company") or "").strip(),
        "source": str(entry.get("source") or "").strip(),
    }


def read_entries(json_path: Path) -> list[dict]:
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"{json_path}: expected a JSON list of entries.")
    seen: set[str] = set()
    entries = []
    for raw in data:
        entry = validate_entry(raw)
        key = entry["brand"].lower()
        if key in seen:
            raise SystemExit(f"{json_path}: duplicate brand {entry['brand']!r}.")
        seen.add(key)
        entries.append(entry)
    return entries


def build_brand_doc(entry: dict) -> dict:
    """Brand doc in the import's shape (see import_data.py), plus
    provenance: where the composition was verified."""
    return {
        "brand_name": entry["brand"],
        "generic_id": None,
        "generic_title": entry["generic_title"],
        "ingredients": entry["ingredients"],
        "dosage_form": entry["dosage_form"],
        "strength_raw": " + ".join(entry["strength_parts"]),
        "strength_parts": entry["strength_parts"],
        "company": entry["company"],
        "medicine_type": "",
        "source": entry["source"],
        "added_via": "add_medicine.py",
    }


def build_generic_doc(entry: dict) -> dict:
    return {
        "title": entry["generic_title"],
        "ingredients": entry["ingredients"],
        "strength_parts": entry["strength_parts"],
        "source": entry["source"],
        "added_via": "add_medicine.py",
    }


def add_entry(drugdb: DrugDB, entry: dict, apply: bool) -> str:
    """Add one medicine (brand + its generic if missing). Returns a
    one-line status for the run summary."""
    brand = entry["brand"]
    existing = drugdb.resolve_brand(brand)
    if existing:
        return (f"  {brand}: already in database as "
                f"{existing.get('brand_name')!r} — skipped.")
    generic_missing = drugdb.resolve_generic(entry["generic_title"]) is None
    if apply:
        drugdb.db.brands.update_one(
            {"brand_name": brand}, {"$set": build_brand_doc(entry)},
            upsert=True)
        if generic_missing:
            drugdb.db.generics.update_one(
                {"title": entry["generic_title"]},
                {"$setOnInsert": build_generic_doc(entry)}, upsert=True)
        generic_note = (" + generic added" if generic_missing
                        else " (generic already present)")
        return f"  {brand}: added{generic_note}."
    generic_note = (" + generic would be added" if generic_missing
                    else " (generic already present)")
    return f"  {brand}: would add{generic_note} (dry run — nothing written)."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(DEFAULT_JSON),
                    help="JSON file with the medicines to add.")
    ap.add_argument("--apply", action="store_true",
                    help="Write the changes (default is --dry-run).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would change; write nothing.")
    ap.add_argument("--mongo-uri", default="")
    ap.add_argument("--db", default="")
    args = ap.parse_args()

    _load_dotenv()
    try:
        entries = read_entries(Path(args.json))
    except (OSError, ValueError, SystemExit) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not entries:
        print("No entries in the JSON file — nothing to do.")
        return 0
    db = _connect_db(args.mongo_uri or None, args.db or None)
    drugdb = DrugDB(db)
    apply = args.apply and not args.dry_run
    for entry in entries:
        print(add_entry(drugdb, entry, apply))
    if not apply:
        print("\nDry run — nothing was written. Re-run with --apply to add them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
