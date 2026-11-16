#!/usr/bin/env python3
"""RxSentinel — look up a missing brand's composition on the web and
draft an add_medicine.py entry for it.

The interaction side already does search-and-store automatically
(web_verify: an Unknown pair is checked against a trusted source and
the graded verdict is stored). Medicines never got the same flow: a
brand the database does not carry just lands in needs_confirmation.

This is the medicine half of that flow, with one deliberate
difference: the search and the draft are automatic, the write to the
database is not. A wrong interaction verdict affects one pair; wrong
ingredients for a brand would poison every future check on that
brand (interactions, duplicates, dose math). So this script only
writes data/suggested_medicines.json — a draft in exactly the shape
add_medicine.py reads. Review it (open the source link), then run:

    python add_medicine.py --json data/suggested_medicines.json --apply

Safety rules, same spirit as web_verify:
- Trusted sources only: MedEx (Bangladesh first), Drugs.com, Practo,
  1mg, Apollo Pharmacy, PharmEasy, TrueMeds. Any other source domain
  is rejected, and so is a reply that names a DIFFERENT brand
  ("Pan 40" is not "Pan D" — look-alikes are the classic failure).
- The composition must come with its source URL, which is stored on
  the draft and then on the brand row, so it stays checkable.
- Strengths must line up one-to-one with ingredients (or be empty).
- A failed lookup (no key, quota, no trusted source found) prints
  why to stderr and drafts nothing — never a guessed entry.

Run:  python lookup_medicine.py --brand Enzoflam --brand "Pan D"
      python lookup_medicine.py --brand Hexigel --out data/suggested_medicines.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from web_verify import _load_dotenv, _make_client  # noqa: E402

TRUSTED_DOMAINS = (
    "medex.com.bd", "drugs.com", "practo.com", "1mg.com",
    "apollopharmacy.in", "pharmeasy.in", "truemeds.in",
)
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "suggested_medicines.json"


def build_prompt(brand: str) -> str:
    return (
        f"What is the composition of the medicine brand {brand!r}? Reply "
        "with JSON ONLY, no other text:\n"
        '{"brand": "Enzoflam", "generic_title": "Diclofenac + Paracetamol '
        '+ Serratiopeptidase", "ingredients": ["Diclofenac", "Paracetamol", '
        '"Serratiopeptidase"], "strength_parts": ["50 mg", "325 mg", '
        '"15 mg"], "dosage_form": "Tablet", "company": "Alkem Laboratories '
        'Ltd", "source": "https://www.practo.com/medicine-info/..."}}\n'
        "Rules:\n"
        "- Answer ONLY from one of these sources: MedEx (medex.com.bd), "
        "Drugs.com, Practo, 1mg, Apollo Pharmacy, PharmEasy, TrueMeds. "
        "If you cannot find this exact brand on one of them, reply "
        '{"found": false}.\n'
        "- The brand must match EXACTLY — do not substitute a look-alike "
        '("Pan 40" is a different drug from "Pan D").\n'
        '- "ingredients" are the active ingredients only, in the order '
        'they appear in the composition; "strength_parts" must line up '
        "one-to-one with them (or be an empty list when unknown).\n"
        '- "source" is the exact page URL where you found the composition.'
    )


def _trusted_source(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS)


def parse_draft(text: str, brand: str) -> dict | None:
    """Strict-parse the model's reply into an add_medicine entry.
    Returns None for anything unverifiable — a bad draft is worse
    than no draft, because it would be one --apply away from the DB."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or data.get("found") is False:
        return None
    if str(data.get("brand") or "").strip().lower() != brand.strip().lower():
        return None  # the model answered for a different product
    title = str(data.get("generic_title") or "").strip()
    ings = [str(i).strip() for i in (data.get("ingredients") or [])
            if str(i).strip()]
    parts = [str(p).strip() for p in (data.get("strength_parts") or [])
             if str(p).strip()]
    source = str(data.get("source") or "").strip()
    if not title or not ings:
        return None
    if parts and len(parts) != len(ings):
        return None
    if not _trusted_source(source):
        return None
    return {
        "brand": brand.strip(),
        "generic_title": title,
        "ingredients": ings,
        "strength_parts": parts,
        "dosage_form": str(data.get("dosage_form") or "").strip(),
        "company": str(data.get("company") or "").strip(),
        "source": source,
    }


def lookup_brand(brand: str, client=None, api_key: str | None = None,
                 base_url: str | None = None,
                 model: str | None = None) -> dict | None:
    """One brand -> draft entry, or None. Never raises: a failed
    lookup (no key, quota, no trusted source) must not break a run —
    it prints why and drafts nothing."""
    try:
        if client is None:
            client = _make_client(api_key, base_url, model)
        text = client.generate(build_prompt(brand))
    except (Exception, SystemExit) as exc:
        print(f"  Lookup failed for {brand}: {exc}", file=sys.stderr)
        return None
    draft = parse_draft(text, brand)
    if draft is None:
        print(f"  No trusted composition found for {brand} — nothing drafted.",
              file=sys.stderr)
    return draft


def verify_source(url: str, ingredients: list[str]) -> bool:
    """Fetch the cited source page and check every ingredient name
    actually appears on it. This is what makes an auto-add trustworthy
    enough to run without a human in the loop: the model's citation is
    checked against the live page, not taken on faith — a model can
    invent a plausible URL on a real domain, and a domain check alone
    cannot catch that. It also catches stale knowledge (the live page
    shows the current formulation). Any fetch problem — blocked,
    timeout, unreachable — returns False, and the entry stays a draft
    for manual review instead of being auto-added."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (RxSentinel lookup)"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            html = response.read().decode("utf-8", "replace").lower()
    except Exception:
        return False
    return all(ing.strip().lower() in html
               for ing in ingredients if ing.strip())


def write_drafts(drafts: list[dict], path: Path) -> tuple[int, int]:
    """Merge drafts into the suggestions file (by brand, so a
    re-lookup updates the old draft instead of duplicating it).
    Returns (new, updated)."""
    existing: list[dict] = []
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = []
    by_brand = {str(e.get("brand", "")).strip().lower(): e for e in existing}
    new, updated = 0, 0
    for draft in drafts:
        key = draft["brand"].strip().lower()
        if key in by_brand:
            by_brand[key] = draft
            updated += 1
        else:
            by_brand[key] = draft
            new += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(by_brand.values()), indent=2,
                               ensure_ascii=False), encoding="utf-8")
    return new, updated


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brand", action="append", default=[], required=True,
                    help="Brand to look up (repeatable).")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Where to write the draft entries.")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    _load_dotenv()
    drafts = []
    for brand in args.brand:
        draft = lookup_brand(brand, api_key=args.api_key or None,
                             base_url=args.base_url or None,
                             model=args.model or None)
        if draft is not None:
            drafts.append(draft)
            print(f"  {brand}: drafted "
                  f"({draft['generic_title']}, source: {draft['source']})")
    if not drafts:
        print("No drafts written — nothing to review.")
        return 0
    new, updated = write_drafts(drafts, Path(args.out))
    print(f"\nWrote {new} new, {updated} updated draft(s) to {args.out}.\n"
          "Review them (open each source link), then run:\n"
          f"  python add_medicine.py --json {args.out} --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
