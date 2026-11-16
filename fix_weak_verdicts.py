#!/usr/bin/env python3
"""One-off correction (Nov 2026): remove web-verified verdicts whose
only source is the checker's index page, and fix one row's source.

What happened: web_verify's parser checked only the source DOMAIN.
Verdicts from the 2026-11-16 handwritten re-run cited
https://www.drugs.com/drug-interactions (the checker index, which
says nothing about any pair) and were stored as if read from a pair
page. One of them — Amoxicillin + Serratiopeptidase = Moderate —
does not hold up: no Drugs.com pair page exists for it, a systematic
review of 24 serrapeptase studies found no clinical drug interaction,
and "separate by 2 hours" is incoherent for ingredients sold as one
tablet. The parser now requires the exact pair page
(web_verify._source_is_pair_specific), so these rows will not come
back — but rows already stored have to be removed by hand, because a
graded row is never overwritten by the web check.

Rows removed (weak source, verdict unproven):
  amoxicillin + serratiopeptidase (Moderate)
  amoxicillin + domperidone (None)
  chlorhexidine + domperidone (None)
  chlorhexidine + serratiopeptidase (Minor, from the earlier run)
Row kept, source fixed: amoxicillin + pantoprazole = None — the
verdict is correct (the Drugs.com pair page says "No drug
interactions found"), only its stored source was the index; the
source is corrected to the real pair page.

Run:  python fix_weak_verdicts.py
Safe to re-run (deletes and the update are idempotent). After it,
re-run the intake — the removed pairs return to Unknown and are
web-checked again under the fixed rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from web_verify import _connect_db, _source_is_pair_specific  # noqa: E402

REMOVE = [
    ("amoxicillin", "serratiopeptidase"),
    ("amoxicillin", "domperidone"),
    ("chlorhexidine", "domperidone"),
    ("chlorhexidine", "serratiopeptidase"),
]
FIX_SOURCE = (
    ("amoxicillin", "pantoprazole"),
    "https://www.drugs.com/drug-interactions/amoxicillin-with-pantoprazole-187-0-1790-0.html",
)


def main() -> int:
    db = _connect_db(None, None)
    for a, b in REMOVE:
        result = db.interactions.delete_one({"drug_a": a, "drug_b": b})
        print(f"  {'Removed' if result.deleted_count else 'No row for'} "
              f"{a} + {b}.")
    (a, b), source = FIX_SOURCE
    db.interactions.update_one({"drug_a": a, "drug_b": b},
                               {"$set": {"source": source}})
    print(f"  Fixed source for {a} + {b} -> {source}")
    print("\nAudit — remaining web-verified rows with a weak source:")
    weak = 0
    for row in db.interactions.find({"verified_via": "web"}):
        if not _source_is_pair_specific(row.get("source") or ""):
            weak += 1
            print(f"  WEAK: {row['drug_a']} + {row['drug_b']} "
                  f"({row.get('severity')}) — {row.get('source')}")
    if not weak:
        print("  none — all web-verified rows cite a pair page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
