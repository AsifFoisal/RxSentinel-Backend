#!/usr/bin/env python3
"""RxSentinel — fill in missing mechanism/management text for interaction
rows that DDInter graded but left blank.

Some DDInter rows carry a severity and nothing else, so the safety
report shows a finding with no explanation ("detail": ""). This script
fills ONLY the explanation fields — mechanism, management and
mechanism_source — from the CSV (data/enrich_mechanisms.csv), one row
per pair, each sourced to a named trusted page.

Safety rules, same spirit as the curated layer:
- Severity is NEVER touched: the CSV's severity is checked against the
  stored row, and a mismatch means the row is skipped, not overwritten.
- Only empty mechanism fields are filled: a row that already has
  mechanism text is left alone.
- Nothing is invented: every row names its source page in
  mechanism_source, and the engine reports it in the finding.
- --dry-run (default) prints what would change and writes nothing.
  Run with --apply to write.

Run:  python enrich_mechanisms.py --dry-run
      python enrich_mechanisms.py --apply
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from load_ddinter import _canonical_drug
except ImportError:  # pragma: no cover - load_ddinter not on the path
    def _canonical_drug(name: str) -> str:  # type: ignore[no-redef]
        return (name or "").strip().lower()

from web_verify import _connect_db, _load_dotenv  # noqa: E402

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "enrich_mechanisms.csv"


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            a, b = sorted([_canonical_drug(r["drug_a"]),
                           _canonical_drug(r["drug_b"])])
            rows.append({
                "drug_a": a, "drug_b": b,
                "severity": (r.get("severity") or "").strip(),
                "mechanism": (r.get("mechanism") or "").strip(),
                "management": (r.get("management") or "").strip(),
                "mechanism_source": (r.get("mechanism_source") or "").strip(),
            })
        return rows


def enrich_row(db, row: dict, apply: bool) -> str:
    """One pair: fill its empty explanation fields, or say why not.
    Returns a one-line status for the run summary."""
    pair = f"{row['drug_a']} + {row['drug_b']}"
    existing = db.interactions.find_one(
        {"drug_a": row["drug_a"], "drug_b": row["drug_b"]})
    if existing is None:
        return f"  {pair}: not in the database — skipped."
    if row["severity"] and existing.get("severity") != row["severity"]:
        return (f"  {pair}: stored severity is {existing.get('severity')!r}, "
                f"CSV expects {row['severity']!r} — skipped, check the row.")
    if (existing.get("mechanism") or "").strip():
        return f"  {pair}: mechanism already filled — skipped."
    update = {"mechanism": row["mechanism"],
              "management": row["management"],
              "mechanism_source": row["mechanism_source"]}
    if apply:
        db.interactions.update_one(
            {"drug_a": row["drug_a"], "drug_b": row["drug_b"]},
            {"$set": update})
        return f"  {pair}: filled in."
    return f"  {pair}: would fill in (dry run — nothing written)."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(DEFAULT_CSV),
                    help="CSV with the mechanism rows.")
    ap.add_argument("--apply", action="store_true",
                    help="Write the changes (default is --dry-run).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would change and write nothing (default).")
    ap.add_argument("--mongo-uri", default="")
    ap.add_argument("--db", default="")
    args = ap.parse_args()

    _load_dotenv()
    rows = read_rows(Path(args.csv))
    if not rows:
        print("No rows in the CSV — nothing to do.")
        return 0
    db = _connect_db(args.mongo_uri or None, args.db or None)
    apply = args.apply and not args.dry_run
    print(f"{'Applying' if apply else 'Dry run'}: {len(rows)} row(s) "
          f"from {args.csv}")
    for row in rows:
        print(enrich_row(db, row, apply))
    if not apply:
        print("Run again with --apply to write these changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
