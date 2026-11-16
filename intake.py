#!/usr/bin/env python3
"""RxSentinel — Phase 3: image intake (prescription / pack / receipt).

Takes a photo, reads the medicines off it, and returns exactly the
JSON shape `safety_engine.py` already consumes:

    {"medicines": [{"brand": "Napa Extra", "doses_per_day": 2, ...}],
     "needs_confirmation": [...], "source": "paddle|llm", ...}

Hybrid OCR, as locked in PHASE3_INTAKE_PREP.md:
  * Printed text -> PaddleOCR (local, free). Runs first.
  * Handwriting, or low-confidence PaddleOCR output -> vision LLM
    (Gemini with the same key as web_verify / Phase 3, or any
    OpenAI-compatible vision provider via LLM_BASE_URL — the same
    free-tier fallback web_verify.py already uses).

Rules this module enforces (the engine enforces the rest):
  * Every extracted medicine carries `source` (paddle/llm) and
    `confidence` (0-1). Below CONFIDENCE_THRESHOLD (0.6) it goes to
    `needs_confirmation` — never silently passed to the engine as fact.
  * A name is only "resolved" when it matches a brand or generic in
    MongoDB (exact, then case-insensitive). Anything else stays
    unresolved with its raw text, for a human to confirm.
  * `doses_per_day` is only set when the frequency was actually read
    ("1+0+1", "twice daily", ...). If it can't be read it is omitted
    and the dose-load check reports it as unchecked — never invented.
  * Expiry/batch are kept as raw strings; the engine's parse_expiry
    handles the format variance.

Needs: Pillow (fixture + image bytes), pymongo (only for a real DB),
paddleocr (printed route; heavy, install on the demo machine),
google-genai (Gemini vision route). All OCR/LLM backends are lazy —
importing this module, and running the tests, needs none of them.

Usage:
    python intake.py --image data/sample_prescription.png --demo-db
    python intake.py --image prescription.jpg --mongo-uri "mongodb+srv://..."
    python intake.py --image handwritten.jpg --handwritten   # force the vision route
    python intake.py --image prescription.jpg --assess       # also run the safety engine
    python intake.py --image pack.jpg --list dispensed --assess
    python intake.py --image prescription.jpg --out prescription.json
    # Handwriting uses its OWN vision wiring (VISION_* in .env, or
    # --vision-base-url / --vision-model / --vision-api-key) — separate
    # from the web-check LLM, because the vision model must accept
    # images and the web-check model may be text-only:
    python intake.py --image handwritten.jpg --handwritten \
        --vision-base-url https://openrouter.ai/api/v1 \
        --vision-model google/gemini-2.5-flash --vision-api-key sk-or-v1-...
    # A brand the database does not carry is looked up on the web and
    # added on the spot — but only when its cited source page lists
    # the ingredients; anything unverifiable stays unconfirmed:
    python intake.py --image handwritten.jpg --handwritten --assess --auto-add-missing

Merged 2026-11-16 from the agent's rewrite (reviewed, dose bug NOT
carried over — this file still sums 1+0+1 digits, it does not count
non-zero slots): the PaddleOCR Windows MKLDNN crash fix, --assess,
--list prescription|dispensed (kept separate for Phase 5
reconciliation), --handwritten, --threshold/--lang, the manufacturer
name filter, and bare-expiry parsing that cannot mistake a phone
number for a date.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path

from safety_engine import (DrugDB, _demo_db, assess, load_max_doses,
                           parse_expiry, parse_strength_mg)

try:
    # Same .env handling as web_verify (GEMINI_API_KEY, LLM_*, MONGO_*).
    from web_verify import GEMINI_MODELS, _load_dotenv
except ImportError:  # intake.py copied away from backend/
    GEMINI_MODELS = ("gemini-2.5-flash", "gemini-2.0-flash")

    def _load_dotenv(path: Path | None = None) -> None:
        env_path = path or Path(__file__).resolve().parent / ".env"
        if not env_path.is_file():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


CONFIDENCE_THRESHOLD = 0.6
# Mean PaddleOCR confidence below this -> the page is probably
# handwritten or blurry, so the vision LLM takes over.
PADDLE_MEAN_CONFIDENCE_FLOOR = 0.7

VISION_PROMPT = """Read this medicine prescription / pack / receipt photo and
extract every medicine on it. Reply with JSON ONLY, no other text:
{"medicines": [{"name": "Napa Extra", "strength": "500mg",
  "frequency": "1+0+1", "expiry": "03/2027", "batch": "B12345",
  "generic": "Paracetamol + Caffeine",
  "confidence": "high|medium|low"}]}
Rules:
- Copy each name EXACTLY as written on the photo. Do not guess a brand
  name, do not correct spelling, do not add medicines that are not there.
- "strength" is the dose strength as written (e.g. "500mg", "20mg").
  "frequency" is how often it is taken as written (e.g. "1+0+1",
  "twice daily", "1 tablet at night"). Omit a field you cannot read.
- "generic" is the active ingredient(s) when you know them (e.g.
  Augmentin -> "Amoxicillin + Clavulanic acid"). It is a fallback for
  brands the database may not carry — only state it when you are sure,
  never guess it.
- "expiry" and "batch" only when they are printed on the photo.
- "confidence" is how sure you are about THIS medicine's name."""


# ------------------------------------------------------------------ parsing

_FREQUENCY_WORDS = (
    (r"\b(once|one time)\b.{0,12}\b(daily|day|a day)\b", 1),
    (r"\b(twice|two times?)\b.{0,12}\b(daily|day|a day)\b", 2),
    (r"\b(thrice|three times?)\b.{0,12}\b(daily|day|a day)\b", 3),
    (r"\b(four times?)\b.{0,12}\b(daily|day|a day)\b", 4),
    (r"\b(1|one)\s*(tablet|tab|cap|capsule)?\s*(daily|a day|per day)\b", 1),
    (r"\b(2|two)\s*(tablets?|tabs?|caps?|capsules?)\s*(daily|a day|per day)\b", 2),
    (r"\b(3|three)\s*(tablets?|tabs?|caps?|capsules?)\s*(daily|a day|per day)\b", 3),
    (r"\bevery\s*8\s*(h|hr|hrs|hours?)\b", 3),
    (r"\bevery\s*12\s*(h|hr|hrs|hours?)\b", 2),
    (r"\bevery\s*6\s*(h|hr|hrs|hours?)\b", 4),
    (r"\bevery\s*24\s*(h|hr|hrs|hours?)\b", 1),
    (r"\b(od|o\.d\.|once daily)\b", 1),
    (r"\b(bd|b\.d\.|twice daily)\b", 2),
    (r"\b(tds|t\.d\.s\.|three times daily)\b", 3),
    (r"\b(qds|q\.d\.s\.|four times daily)\b", 4),
    (r"\b(at night|at bedtime|at bed time|before sleep|nightly)\b", 1),
    (r"\b(morning|in the morning)\s*(only|once)?\b", 1),
)


def parse_frequency(text: str) -> int | None:
    """Doses per day from a frequency as written. None = not readable
    (the caller omits doses_per_day; it is never invented)."""
    t = (text or "").strip().lower()
    if not t:
        return None
    # "1+0+1" / "1-0-1" / "1+1+1": doses = sum of the digits.
    m = re.search(r"\b([0-3])\s*[+\-]\s*([0-3])\s*[+\-]\s*([0-3])\b", t)
    if m:
        return sum(int(g) for g in m.groups())
    m = re.search(r"\b([0-3])\s*[+\-]\s*([0-3])\b", t)
    if m:
        return sum(int(g) for g in m.groups())
    for pattern, doses in _FREQUENCY_WORDS:
        if re.search(pattern, t):
            return doses
    m = re.search(r"\b(\d+)\s*(times?|x)\s*(a|per)?\s*(daily|day)\b", t)
    if m:
        return int(m.group(1))
    return None


_STRENGTH_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|g|mcg|µg|ml|iu)\b", re.I)
_EXPIRY_RE = re.compile(
    r"\b(?:exp(?:iry|iry date)?|exp\.?|use by|valid (?:till|until))\s*[:\-]?\s*"
    r"(\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?|\d{1,2}[-/]\d{4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|"
    r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})",
    re.I)
_BATCH_RE = re.compile(
    r"\b(?:batch(?: no\.?| number)?|b\.?no\.?|lot(?: no\.?)?)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-]{2,})",
    re.I)


# Bare date shapes with no keyword — only year-first or month/year
# with a plausible month, so a phone number like 01712-345678 can
# never be read as an expiry.
_BARE_EXPIRY_RES = (
    re.compile(r"\b(20\d{2}[-/](?:0?[1-9]|1[0-2]))\b"),
    re.compile(r"\b((?:0?[1-9]|1[0-2])/20\d{2})\b"),
)


def parse_expiry_batch(text: str) -> dict:
    """Raw expiry/batch strings from a line, validated by the engine's
    parse_expiry so only real dates are kept."""
    out: dict = {}
    m = _EXPIRY_RE.search(text or "")
    if m and parse_expiry(m.group(1)) is not None:
        out["expiry"] = m.group(1)
    if "expiry" not in out:
        for pattern in _BARE_EXPIRY_RES:
            bare = pattern.search(text or "")
            if bare and parse_expiry(bare.group(1)) is not None:
                out["expiry"] = bare.group(1)
                break
    m = _BATCH_RE.search(text or "")
    if m:
        out["batch"] = m.group(1)
    return out


def _strip_dose_tokens(line: str) -> str:
    """Remove strength, frequency and expiry/batch tokens from a line,
    leaving the medicine name."""
    s = _STRENGTH_RE.sub(" ", line)
    s = re.sub(r"\b[0-3]\s*[+\-]\s*[0-3](\s*[+\-]\s*[0-3])?\b", " ", s)
    s = _EXPIRY_RE.sub(" ", s)
    s = _BATCH_RE.sub(" ", s)
    s = re.sub(r"\b\d+\s*(tablets?|tabs?|caps?|capsules?)\b", " ", s, flags=re.I)
    # Note text after a dash separator ("1+0+1 — After meal",
    # "1+0+1 - After meal"): the name is the text before the first
    # separator. This MUST run before the character filter below —
    # that filter turns the em/en dash into a space, so a later split
    # never sees it (the bug that left "After meal" and "Before
    # breakfast" inside the name while the ASCII-hyphen case worked).
    s = re.split(r"\s+[-–—]\s+", s)[0]
    # A trailing bare number is a unit-less strength ("Paracetamol 500",
    # "Dolo 650") — drop it so the name can resolve.
    s = re.sub(r"\s\d{2,4}$", " ", s)
    # Prescription list numbers ("1. Napa Extra", "2) Seclo") and
    # dosage-form prefixes ("Tab. Napa Extra") are not part of the
    # name. The dotted form needs a letter after it ("0.5% ..." must
    # not lose its "0."); the bare form ("1 Napa Extra") is only
    # stripped when the line carries a strength or dose pattern, so a
    # leading number on an address line is never mistaken for one.
    s = re.sub(r"^\s*\d{1,2}[.)]\s+(?=[A-Za-z])", "", s)
    if (_STRENGTH_RE.search(line)
            or re.search(r"\b[0-3]\s*[+\-]\s*[0-3]\b", line)):
        s = re.sub(r"^\s*\d{1,2}\s+(?=[A-Za-z])", "", s)
    s = re.sub(r"^(tab|cap|syp|inj|tablet|capsule|syrup|injection|cream|"
               r"ointment|drop|drops|supp|soln)\.?\s+", "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9 &+\-\[\]]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Dose tokens leave separators behind ("Napa Extra - 1+0+1" -> "Napa Extra -").
    return re.sub(r"^[\s\-+–—:;,.]+|[\s\-+–—:;,.]+$", "", s).strip()


# ------------------------------------------------------------------ grounding

# Dosage-form words a vision model or OCR line may glue onto a brand
# ("Hexigel gum paint" — the database brand is just "Hexigel").
# Stripped ONLY as a fallback try, after the full name fails, so a
# brand that genuinely ends in one of these words still resolves
# exactly first.
FORM_WORDS = {"tablet", "tablets", "tab", "tabs", "capsule", "capsules",
              "cap", "caps", "syrup", "syp", "injection", "inj", "cream",
              "gel", "gum", "paint", "drops", "drop", "ointment",
              "solution", "soln", "suspension", "supp", "suppository",
              "spray", "lotion", "powder"}


def _strip_form_words(name: str) -> str:
    """Drop trailing dosage-form words: 'Hexigel gum paint' -> 'Hexigel'."""
    words = name.split()
    while words and words[-1].lower().strip(".,") in FORM_WORDS:
        words.pop()
    return " ".join(words)


def _name_variants(name: str) -> list[str]:
    """Name forms to try against the database, most faithful first:
    the name as written, then with dosage-form words stripped."""
    clean = name.strip()
    variants = [clean]
    stripped = _strip_form_words(clean)
    if stripped and stripped != clean:
        variants.append(stripped)
    return variants


def ground_name(name: str, drugdb: DrugDB | None
                ) -> tuple[str | None, str | None, float, dict | None]:
    """Match a name against the database. Returns (kind, matched_name,
    confidence, doc): kind is 'brand' / 'generic' / None. Exact brand
    first, then case-insensitive, then space/hyphen-tolerant
    (DrugDB.resolve_*), then generic; each tried with dosage-form
    words stripped as a fallback ('Hexigel gum paint' -> 'Hexigel').
    The doc is returned so callers can read its ingredients (used to
    spot a pack's composition line restating a brand's ingredients)."""
    if drugdb is None or not (name or "").strip():
        return None, None, 0.0, None
    for variant in _name_variants(name):
        doc = drugdb.resolve_brand(variant)
        if doc:
            return "brand", doc.get("brand_name") or variant, 1.0, doc
    for variant in _name_variants(name):
        doc = drugdb.resolve_generic(variant)
        if doc:
            return "generic", doc.get("title") or variant, 0.85, doc
    return None, None, 0.0, None


def _ingredients_of(doc: dict | None) -> frozenset:
    if not doc:
        return frozenset()
    return frozenset(str(i).strip().lower()
                     for i in doc.get("ingredients") or [])


# Document headers, not medicines — a line that is only one of these
# is skipped instead of becoming a "please confirm" item.
HEADER_WORDS = {"prescription", "rx", "doctor", "patient", "date", "age",
                "sex", "signature", "medicines", "medicine", "drugs",
                "pharmacy", "hospital", "clinic", "advice"}

# Address / contact words — a line carrying one of these is clinic
# furniture ("12 Green Road, Dhaka • Phone: 01XXXXXXXXX"), never a
# medicine, however it is punctuated.
ADDRESS_WORDS = {"road", "street", "avenue", "lane", "phone", "tel",
                 "telephone", "mobile", "fax", "address", "house"}


def _is_manufacturer(name: str, drugdb: DrugDB | None) -> bool:
    """True when the name is a manufacturer, not a medicine — a match
    against the brands collection's own company field. Two tries: the
    full name first ("Beximco Pharmaceuticals Ltd."), then the base
    name with the legal suffix stripped ("Beximco Pharmaceuticals"),
    because the dataset may store 'Ltd.' where the pack prints 'Ltd'
    or 'Limited'. Resolved brands never reach here, so nothing is
    guessed away. Needs the DrugDB adapter (collections under .db);
    without a DB it returns False."""
    if drugdb is None or drugdb.db is None or not (name or "").strip():
        return False
    clean = name.strip().rstrip(". ")
    patterns = [
        "^" + re.escape(clean) + r"\.?\s*$",
        "^" + re.escape(re.sub(
            r"\s+(ltd\.?|limited|inc\.?|co\.?|company)$", "", clean,
            flags=re.I)) + r"\b",
    ]
    try:
        for pattern in patterns:
            if drugdb.db.brands.find_one(
                    {"company": {"$regex": pattern, "$options": "i"}}) is not None:
                return True
    except Exception:
        return False
    return False


def candidate_from_line(line: str, line_conf: float, drugdb: DrugDB | None,
                        source: str) -> dict | None:
    """One OCR text line -> one candidate medicine (or None when the
    line carries no medicine). Strength/frequency/expiry/batch are
    parsed from the same line; the name is grounded against the DB."""
    name = _strip_dose_tokens(line)
    if not name or len(name) < 2:
        return None
    # A header anywhere in the line ("City Care Clinic", "Patient: Test
    # Patient", "Advice: ...") means it is document furniture, not a
    # medicine — skip it instead of a "please confirm" item.
    if any(w in HEADER_WORDS for w in name.lower().split()):
        return None
    # A manufacturer name on a pack ("Beximco Pharmaceuticals Ltd.")
    # is not a medicine either.
    if _is_manufacturer(name, drugdb):
        return None
    doses = parse_frequency(line)
    strength = _STRENGTH_RE.search(line)
    kind, matched, ground_conf, doc = ground_name(name, drugdb)
    # Not every line is a medicine: a clinic address ("12 Green Road,
    # Dhaka • Phone: 01XXXXXXXXX") reaches this point with no strength,
    # no frequency and no database match. A real unknown brand
    # ("Zentel") has no digits in it; an address or phone line does.
    # Skip both cases instead of asking the user to confirm an address
    # as a medicine.
    if set(name.lower().split()) & ADDRESS_WORDS:
        return None
    if (kind is None and strength is None and doses is None
            and any(ch.isdigit() for ch in name)):
        return None
    conf = line_conf
    if kind == "brand":
        conf = max(conf, 0.9)
    elif kind == "generic":
        conf = max(conf, 0.75)
    else:
        conf = min(conf, 0.4)  # unresolved: confirm list, never trusted
    med: dict = {
        "name": matched or name,
        "raw_name": name,
        "source": source,
        "confidence": round(conf, 2),
        "resolved": kind is not None,
        "resolve_kind": kind,
        "ingredients": sorted(_ingredients_of(doc)),
        # Display fields from the matched doc (used by
        # to_prescription_json for brand-resolved medicines).
        "generic_title": (doc or {}).get("generic_title") or (doc or {}).get("title"),
        "strength_parts": [str(p) for p in (doc or {}).get("strength_parts") or []],
    }
    if doses is not None:
        med["doses_per_day"] = doses
    if strength:
        med["strength"] = strength.group(0)
        mg = parse_strength_mg(strength.group(0))
        if mg is not None:
            med["strength_mg"] = mg
    med.update(parse_expiry_batch(line))
    return med


def candidate_from_llm(item: dict, drugdb: DrugDB | None) -> dict:
    """One LLM-extracted medicine -> candidate. The LLM's own
    confidence is capped by grounding: a name the DB cannot resolve
    is never trusted, however confident the model claims to be.
    Fallback: when the brand is not in the database, try the model's
    stated generic (Augmentin -> Amoxicillin + Clavulanic acid) — if
    that generic IS in the database, the medicine resolves through it
    and the engine can still check its interactions and dose, with the
    brand kept as the display label. The generic link is the model's
    claim, so the confidence is capped at 0.85 for this path."""
    name = str(item.get("name") or "").strip()
    conf_map = {"high": 0.9, "medium": 0.7, "low": 0.4}
    conf = conf_map.get(str(item.get("confidence") or "low").lower(), 0.4)
    kind, matched, _, doc = ground_name(name, drugdb)
    brand_label = None
    if kind is None:
        generic = str(item.get("generic") or "").strip()
        if generic:
            gkind, gmatched, _, gdoc = ground_name(generic, drugdb)
            # Only a GENERIC match counts here: a generic string that
            # happens to match a brand name is ambiguous, so it stays
            # unresolved instead of being silently re-labelled.
            if gkind == "generic":
                kind, matched, doc = gkind, gmatched, gdoc
                brand_label = name  # keep the prescribed brand as label
                conf = min(conf, 0.85)
    if kind is None:
        conf = min(conf, 0.4)
    med: dict = {
        "name": matched or name,
        "raw_name": name,
        "source": "llm",
        "confidence": round(conf, 2),
        "resolved": kind is not None,
        "resolve_kind": kind,
        "ingredients": sorted(_ingredients_of(doc)),
        # Display fields from the matched doc (used by
        # to_prescription_json for brand-resolved medicines).
        "generic_title": (doc or {}).get("generic_title") or (doc or {}).get("title"),
        "strength_parts": [str(p) for p in (doc or {}).get("strength_parts") or []],
    }
    if brand_label:
        med["brand_label"] = brand_label
    elif kind is None and item.get("generic"):
        # Unresolved, but the model stated a composition: keep it as a
        # hint on the confirm item, so the human verifying a
        # lookup_medicine draft has a second claim to compare — never
        # passed to the engine as fact.
        med["stated_generic"] = str(item["generic"]).strip()
    doses = parse_frequency(str(item.get("frequency") or ""))
    if doses is not None:
        med["doses_per_day"] = doses
    strength = str(item.get("strength") or "").strip()
    if strength:
        med["strength"] = strength
        mg = parse_strength_mg(strength)
        if mg is not None:
            med["strength_mg"] = mg
    for key in ("expiry", "batch"):
        if item.get(key):
            med[key] = str(item[key]).strip()
    return med


def _drop_composition_lines(candidates: list[dict]) -> list[dict]:
    """A pack prints its composition under the brand ("Paracetamol
    500mg + Caffeine 65mg" under "Napa Extra"). Against the real DB
    that line resolves as the generic "Paracetamol + Caffeine" — the
    same medicine listed twice, which would make the engine flag a
    duplicate paracetamol + caffeine that isn't there. Drop a
    combination-generic candidate only when a brand in the same image
    has exactly the same ingredient set. A generic prescribed on its
    own (no brand with the same set) is kept, and single-ingredient
    generics are never dropped — a real second medicine stays."""
    brand_sets = {frozenset(c.get("ingredients") or [])
                  for c in candidates if c.get("resolve_kind") == "brand"}
    brand_sets.discard(frozenset())
    kept: list[dict] = []
    for c in candidates:
        if (c.get("resolve_kind") == "generic"
                and " + " in (c.get("name") or "")
                and frozenset(c.get("ingredients") or []) in brand_sets):
            continue
        kept.append(c)
    return kept


def to_prescription_json(candidates: list[dict], source: str,
                         doc_meta: dict | None = None,
                         threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """Split candidates into engine-ready medicines and the
    please-confirm list. Expiry/batch printed once on the document
    (not on a medicine line) is attached to every medicine — pack
    photos carry one pack's date, and the engine checks it per
    medicine."""
    candidates = _drop_composition_lines(candidates)
    medicines, confirm = [], []
    doc_meta = dict(doc_meta or {})
    for c in candidates:
        if c.get("expiry") or c.get("batch"):
            doc_meta.setdefault("expiry", c.get("expiry"))
            doc_meta.setdefault("batch", c.get("batch"))
    for c in candidates:
        med: dict = {}
        if c.get("resolve_kind") == "brand":
            med["brand"] = c["name"]
        elif c.get("resolve_kind") == "generic":
            med["generic"] = c["name"]
            if c.get("brand_label"):
                # Resolved through the stated generic (brand not in the
                # DB): the engine tries the brand first, falls back to
                # the generic, and labels the report with the brand.
                med["brand"] = c["brand_label"]
        else:
            med["brand"] = c["name"]  # engine will flag it unresolved
        if c.get("doses_per_day") is not None:
            med["doses_per_day"] = c["doses_per_day"]
        if c.get("strength_mg") is not None and c.get("resolve_kind") == "generic":
            med["strength_mg"] = c["strength_mg"]
        if c.get("resolve_kind") == "brand":
            # Display fields from the matched brand doc: the generic
            # title (the composition) and per-ingredient strengths, so
            # the UI cards and the PDF show them instead of "—".
            # Deliberately NOT the `ingredients` list itself: putting
            # it on the medicine would make
            # safety_engine.expand_medicine take its explicit-
            # ingredients path (meant for web drafts) instead of the
            # database lookup, which would drop the stated strength_mg
            # override for single-ingredient medicines.
            if c.get("generic_title"):
                med["generic"] = c["generic_title"]
            if c.get("strength_parts"):
                med["strength_parts"] = list(c["strength_parts"])
        med["expiry"] = c.get("expiry") or doc_meta.get("expiry")
        med["batch"] = c.get("batch") or doc_meta.get("batch")
        med = {k: v for k, v in med.items() if v is not None}
        med["source"] = c["source"]
        med["confidence"] = c["confidence"]
        if c["confidence"] >= threshold and c["resolved"]:
            medicines.append(med)
        else:
            item = {**med,
                    "reason": ("not in the medicine database"
                               if not c["resolved"]
                               else "low OCR confidence")}
            if c.get("stated_generic"):
                item["stated_generic"] = c["stated_generic"]
            confirm.append(item)
    return {
        "medicines": medicines,
        "needs_confirmation": confirm,
        "source": source,
        "confidence_threshold": threshold,
        "disclaimer": ("For information only — not medical advice. "
                       "Ask your pharmacist or doctor before changing any medicine."),
    }


def merge_continuation_lines(lines: list[dict], drugdb: DrugDB | None) -> list[dict]:
    """Prescriptions often wrap one medicine over two lines — the name
    on one line, the frequency on the next ("Napa Extra 500mg" /
    "1+0+1 — After meal"). A frequency line with no strength and no
    resolvable name belongs to the medicine above it, so merge it
    there instead of letting it become a phantom "After meal" medicine.
    Full medicine lines (name + strength + frequency) are untouched."""
    merged: list[dict] = []
    for line in lines:
        text, conf = line["text"], line["confidence"]
        if (merged and parse_frequency(text) is not None
                and not _STRENGTH_RE.search(text)):
            name = _strip_dose_tokens(text)
            kind, _, _, _ = ground_name(name, drugdb)
            if not name or kind is None:
                merged[-1] = {"text": merged[-1]["text"] + " " + text,
                              "confidence": min(merged[-1]["confidence"], conf)}
                continue
        merged.append({"text": text, "confidence": conf})
    return merged


# ------------------------------------------------------------------ OCR backends

class PaddleOCRReader:
    """Printed-text OCR. PaddleOCR's API changed across versions, so
    both the new predict() dict format and the old ocr() list format
    are normalised to [{"text", "confidence"}]."""

    def __init__(self, lang: str = "en"):
        # Paddle 3.x on Windows CPU crashes inside its oneDNN (MKLDNN)
        # kernels with the OCR models before OCR runs — force the plain
        # CPU kernels. The flag must be set before the first paddle
        # import in the process.
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise SystemExit(
                "PaddleOCR is not installed. Install it on the demo machine:\n"
                "  pip install paddlepaddle paddleocr\n"
                "It is heavy (needs torch) — the vision-LLM route (--mode llm) "
                "works without it.") from exc
        try:
            self._ocr = PaddleOCR(use_textline_orientation=True, lang=lang,
                                  enable_mkldnn=False)
        except TypeError:  # older paddleocr without the kwarg — env flag covers it
            self._ocr = PaddleOCR(use_textline_orientation=True, lang=lang)

    def read(self, image_path: str) -> list[dict]:
        result = None
        if hasattr(self._ocr, "predict"):
            result = self._ocr.predict(image_path)
        else:
            result = self._ocr.ocr(image_path)
        # 3.x predict() returns a LIST of dicts (one per image) with
        # rec_texts/rec_scores; 2.x ocr() returns a list of pages, each
        # a list of [box, (text, score)]. A dict page must be read as a
        # dict — iterating it as a list silently drops every line (the
        # bug that made the live run fall back to the LLM route).
        pages = result if isinstance(result, list) else [result]
        lines: list[dict] = []
        for page in pages:
            if isinstance(page, dict):
                for text, score in zip(page.get("rec_texts") or [],
                                       page.get("rec_scores") or []):
                    if text and str(text).strip():
                        lines.append({"text": str(text).strip(),
                                      "confidence": float(score or 0)})
            elif isinstance(page, list):
                for item in page or []:
                    try:
                        text, score = item[1][0], item[1][1]
                    except (TypeError, IndexError):
                        continue
                    if text and str(text).strip():
                        lines.append({"text": str(text).strip(),
                                      "confidence": float(score or 0)})
        return lines


class GeminiVisionReader:
    """Handwriting / low-confidence route, native Gemini (grounded by
    the prompt + DB grounding afterwards, not by search)."""

    def __init__(self, api_key: str | None = None):
        _load_dotenv()
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("No Gemini API key. Set GEMINI_API_KEY in backend/.env "
                             "or point the vision route at another provider "
                             "with VISION_BASE_URL + VISION_MODEL + VISION_API_KEY "
                             "(or --vision-base-url / --vision-model / --vision-api-key).")
        try:
            from google import genai
            from google.genai import errors as genai_errors
            from google.genai import types
        except ImportError as exc:
            raise SystemExit(
                "google-genai is not installed. Run: pip install google-genai") from exc
        self._client = genai.Client(api_key=key)
        self._types = types
        self._errors = genai_errors

    def extract(self, image_path: str) -> str:
        data = Path(image_path).read_bytes()
        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        config = self._types.GenerateContentConfig(response_mime_type="application/json")
        last_error: Exception | None = None
        for model in GEMINI_MODELS:
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=[self._types.Part.from_bytes(data=data, mime_type=mime),
                              VISION_PROMPT],
                    config=config)
                return response.text
            except self._errors.ClientError as exc:
                if exc.code != 404:
                    raise
                last_error = exc
        raise SystemExit(f"None of the Gemini models {GEMINI_MODELS} could be used: {last_error}")


class GenericVisionReader:
    """Handwriting route for any OpenAI-compatible vision provider
    (OpenRouter, Groq, local Ollama/LM Studio): one stdlib POST with
    the image as a base64 data URI. No SDK needed."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None):
        _load_dotenv()
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.api_key = (api_key or os.environ.get("LLM_API_KEY")
                        or os.environ.get("GEMINI_API_KEY") or "")
        missing = [n for n, v in (("LLM_BASE_URL/--base-url", self.base_url),
                                  ("LLM_MODEL/--model", self.model),
                                  ("LLM_API_KEY/--api-key", self.api_key)) if not v]
        if missing:
            raise SystemExit("Generic vision provider half-configured, missing: "
                             + ", ".join(missing))
        if not self.base_url.endswith("/chat/completions"):
            self.base_url += "/chat/completions"

    def extract(self, image_path: str) -> str:
        import urllib.error
        import urllib.request

        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{b64}"}}]}],
            "temperature": 0,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.base_url, data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()[:300]
            raise RuntimeError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
        except OSError as exc:
            raise RuntimeError(f"Could not reach {self.base_url}: {exc}") from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unusable reply from {self.base_url}: "
                             f"{json.dumps(body)[:300]}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Empty reply from {self.base_url}")
        return content


def pick_vision_reader(api_key=None, base_url=None, model=None,
                       vision_api_key=None, vision_base_url=None,
                       vision_model=None):
    """Which backend reads handwriting. The vision route has its OWN
    wiring, separate from the web-check LLM (web_verify): a vision
    model must see images, a text-only web-check model cannot — so the
    two must be allowed to point at different providers.

    Precedence, per field: --vision-* flags > VISION_* env >
    --api-key/--base-url/--model flags > LLM_* env > GEMINI_API_KEY
    (native Gemini). A dedicated vision provider wins when its base
    URL or model is set; otherwise native Gemini reads the image."""
    _load_dotenv()
    base_url = (vision_base_url or os.environ.get("VISION_BASE_URL", "")
                or base_url or os.environ.get("LLM_BASE_URL", ""))
    model = (vision_model or os.environ.get("VISION_MODEL", "")
             or model or os.environ.get("LLM_MODEL", ""))
    api_key = (vision_api_key or os.environ.get("VISION_API_KEY", "")
               or api_key or os.environ.get("LLM_API_KEY", "")
               or os.environ.get("GEMINI_API_KEY", ""))
    if base_url or model:
        return GenericVisionReader(base_url, api_key, model)
    return GeminiVisionReader(api_key)


def parse_llm_json(text: str) -> list[dict]:
    """Parse the vision LLM's JSON reply, tolerating ``` fences."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object in vision reply: {text[:200]!r}")
    data = json.loads(match.group(0))
    items = data.get("medicines") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("Vision reply has no 'medicines' list.")
    return [i for i in items if isinstance(i, dict)]


# ------------------------------------------------------------------ intake flow

def _auto_add_missing(candidates: list[dict], drugdb: DrugDB | None,
                      api_key=None, base_url=None, model=None,
                      medicine_lookup=None, source_verifier=None,
                      suggest_path=None) -> None:
    """With --auto-add-missing: for each unresolved candidate, look the
    brand up on the web and — only when the cited source page itself
    lists the ingredients (verify_source) — add it to the database and
    re-ground the candidate in place, so the same run assesses it.
    Anything unverifiable stays in needs_confirmation, and a draft is
    saved for manual review when a path is given. Never raises: a
    failed lookup or add leaves the candidate exactly as it was.
    Skipped entirely without a real database (no DB, or the in-memory
    demo DB, which has nothing durable to write to)."""
    if drugdb is None or drugdb.db is None:
        return
    if not hasattr(drugdb.db.brands, "update_one"):
        return  # demo DB: find-only, nothing to write to
    from add_medicine import add_entry
    from lookup_medicine import lookup_brand, verify_source, write_drafts

    lookup = medicine_lookup or lookup_brand
    verify = source_verifier or verify_source
    drafts: list[dict] = []
    for c in candidates:
        if c.get("resolved"):
            continue
        name = (c.get("raw_name") or c.get("name") or "").strip()
        if not name:
            continue
        try:
            draft = lookup(name, api_key=api_key, base_url=base_url,
                           model=model)
        except (Exception, SystemExit):
            draft = None
        if draft is None:
            continue
        try:
            ok = bool(verify(draft["source"], draft["ingredients"]))
        except Exception:
            ok = False
        if not ok:
            drafts.append(draft)
            print(f"  {name}: source page could not confirm the "
                  "composition — left in needs_confirmation, draft saved "
                  "for review.", file=sys.stderr)
            continue
        try:
            status = add_entry(
                drugdb, draft, apply=True,
                added_via="lookup_medicine.py (auto, source page verified)")
        except Exception as exc:  # noqa: BLE001 - never break intake
            print(f"  Auto-add failed for {name}: {exc}", file=sys.stderr)
            continue
        print(f"  {status.strip()}")
        kind, matched, _, doc = ground_name(name, drugdb)
        if kind is not None:
            c["name"] = matched
            c["resolved"] = True
            c["resolve_kind"] = kind
            c["confidence"] = 0.9 if kind == "brand" else 0.85
            c["ingredients"] = sorted(_ingredients_of(doc))
            c.pop("stated_generic", None)
    if drafts and suggest_path is not None:
        write_drafts(drafts, Path(suggest_path))


def run_intake(image_path: str, mode: str = "auto", drugdb: DrugDB | None = None,
               paddle_reader=None, vision_reader=None,
               threshold: float = CONFIDENCE_THRESHOLD, lang: str = "en",
               api_key=None, base_url=None, model=None,
               vision_api_key=None, vision_base_url=None,
               vision_model=None, auto_add_missing: bool = False,
               medicine_lookup=None, source_verifier=None,
               suggest_path=None) -> dict:
    """Image -> prescription JSON. `mode`: auto (PaddleOCR first, vision
    LLM only when it is missing or low-confidence), paddle, or llm.
    Readers are injectable for the tests. api_key/base_url/model are
    the shared LLM config (also used by the web check); the vision_*
    values are the dedicated handwriting wiring and win per field —
    see pick_vision_reader. With auto_add_missing, a brand the database
    does not carry is looked up on the web and added on the spot — but
    only when its cited source page lists the ingredients
    (see _auto_add_missing); anything unverifiable stays unconfirmed."""
    path = Path(image_path)
    if not path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    source = "paddle"
    candidates: list[dict] = []
    if mode in ("auto", "paddle"):
        try:
            reader = paddle_reader or PaddleOCRReader(lang=lang)
            lines = reader.read(str(path))
        except SystemExit:
            if mode == "paddle":
                raise
            print("PaddleOCR unavailable — using the vision LLM route.")
            lines = []
        if lines:
            mean_conf = sum(l["confidence"] for l in lines) / len(lines)
            if mode == "paddle" or mean_conf >= PADDLE_MEAN_CONFIDENCE_FLOOR:
                lines = merge_continuation_lines(lines, drugdb)
                candidates = [c for line in lines
                              if (c := candidate_from_line(
                                  line["text"], line["confidence"], drugdb, "paddle"))]
                # Expiry/batch printed on its own line (no medicine on
                # it) still belongs to the document — collect it here.
                doc_meta: dict = {}
                for line in lines:
                    for k, v in parse_expiry_batch(line["text"]).items():
                        doc_meta.setdefault(k, v)
                if auto_add_missing:
                    _auto_add_missing(candidates, drugdb, api_key, base_url,
                                      model, medicine_lookup, source_verifier,
                                      suggest_path)
                return to_prescription_json(candidates, "paddle", doc_meta, threshold)
            print(f"PaddleOCR mean confidence {mean_conf:.2f} is low — "
                  "using the vision LLM route.")
        elif mode == "paddle":
            # PaddleOCR read the image and found no text: an empty
            # result, not a silent switch to the LLM.
            return to_prescription_json([], "paddle", None, threshold)
    # Vision LLM route (handwriting, or PaddleOCR missing / low-confidence).
    source = "llm"
    reader = vision_reader or pick_vision_reader(
        api_key=api_key, base_url=base_url, model=model,
        vision_api_key=vision_api_key, vision_base_url=vision_base_url,
        vision_model=vision_model)
    text = reader.extract(str(path))
    items = parse_llm_json(text)
    candidates = [candidate_from_llm(item, drugdb) for item in items]
    if auto_add_missing:
        _auto_add_missing(candidates, drugdb, api_key, base_url, model,
                          medicine_lookup, source_verifier, suggest_path)
    return to_prescription_json(candidates, source, None, threshold)


def _web_check_configured(api_key: str = "", base_url: str = "",
                          model: str = "") -> bool:
    """Is a web-check backend configured — a key for Gemini, or a
    base URL / model for a generic provider (local Ollama/LM Studio
    needs no key)? Checked once, so an unconfigured run says so a
    single time instead of failing once per Unknown pair."""
    _load_dotenv()
    return bool(api_key or base_url or model
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("LLM_API_KEY")
                or os.environ.get("LLM_BASE_URL")
                or os.environ.get("LLM_MODEL"))


def assess_intake(result: dict, drugdb: DrugDB, web_verify=None) -> dict | None:
    """Run the deterministic safety engine on the CONFIRMED medicines
    only — never the needs_confirmation list. Returns None when there
    is nothing confirmed to check (the caller reports that instead of
    running the engine on guesses). `web_verify` is passed through to
    the engine: None keeps it fully offline, a callable lets Unknown
    pairs be answered by a web check (see web_verify.web_verify_pair)."""
    if not result.get("medicines"):
        return None
    return assess(drugdb, result["medicines"], load_max_doses(),
                  web_verify=web_verify)


def _connect_db(mongo_uri: str | None, db_name: str | None):
    _load_dotenv()
    uri = mongo_uri or os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017")
    name = db_name or os.environ.get("MONGO_DB", "rxsentinel")
    try:
        from pymongo import MongoClient
    except ImportError:
        raise SystemExit("pymongo is not installed. Run: pip install pymongo")
    return MongoClient(uri, serverSelectionTimeoutMS=5000)[name]


def _open_mongo_drugdb(mongo_uri: str | None, db_name: str | None) -> DrugDB:
    """Connect to the real MongoDB and prove it holds data BEFORE any
    OCR runs. Prints the brand/generic counts, so you can see at a
    glance that you are on the real database and not the 7-brand demo
    one. Stops with a plain message when the server cannot be reached
    or the collections are empty — an empty result almost always means
    the wrong database name (the data lives in a different db)."""
    db = _connect_db(mongo_uri, db_name)
    try:
        n_brands = db.brands.count_documents({})
        n_generics = db.generics.count_documents({})
    except Exception as exc:
        raise SystemExit(
            f"Could not reach MongoDB ({exc}). Check MONGO_URI in "
            "backend/.env or pass --mongo-uri.")
    print(f"Connected to MongoDB '{db.name}': {n_brands} brands, "
          f"{n_generics} generics.")
    if n_brands == 0:
        raise SystemExit(
            "This database has 0 brands — it looks empty. Your data is "
            "probably in a different database; check MONGO_DB in "
            "backend/.env or pass --db <name>.")
    return DrugDB(db)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="Prescription / pack / receipt photo.")
    ap.add_argument("--mode", default="auto", choices=["auto", "paddle", "llm"],
                    help="auto: PaddleOCR first, vision LLM fallback (default).")
    ap.add_argument("--list", default="prescription",
                    choices=["prescription", "dispensed"],
                    help="Which list this image is. Kept separate — Phase 5 "
                         "reconciliation depends on the separation.")
    ap.add_argument("--handwritten", action="store_true",
                    help="Skip PaddleOCR; use the vision model (same as --mode llm).")
    ap.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                    help=f"Confidence threshold (default {CONFIDENCE_THRESHOLD}). "
                         "Below it, items go to needs_confirmation, never to the engine.")
    ap.add_argument("--lang", default="en",
                    help="PaddleOCR language (default en; 'bang' for printed Bangla).")
    ap.add_argument("--mongo-uri", default="")
    ap.add_argument("--db", default="")
    ap.add_argument("--demo-db", action="store_true",
                    help="Use the built-in demo medicine list instead of MongoDB.")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--vision-api-key", default="",
                    help="Dedicated handwriting (vision) provider key — "
                         "separate from the web-check LLM. The vision model "
                         "must accept images; a text-only model cannot read "
                         "handwriting.")
    ap.add_argument("--vision-base-url", default="",
                    help="Dedicated handwriting (vision) provider base URL, "
                         "e.g. https://openrouter.ai/api/v1 or a local "
                         "LM Studio http://localhost:1234/v1.")
    ap.add_argument("--vision-model", default="",
                    help="Dedicated handwriting (vision) model, e.g. "
                         "google/gemini-2.5-flash or qwen/qwen2.5-vl-72b-instruct.")
    ap.add_argument("--auto-add-missing", action="store_true",
                    help="When a brand is not in the database, look it up "
                         "on the web and add it on the spot — but ONLY when "
                         "its cited source page lists the ingredients "
                         "(anything unverifiable stays in "
                         "needs_confirmation, with a draft saved to "
                         "data/suggested_medicines.json for review).")
    ap.add_argument("--assess", action="store_true",
                    help="Also run safety_engine.assess() on the CONFIRMED medicines.")
    ap.add_argument("--no-web-verify", action="store_true",
                    help="With --assess, do NOT web-check pairs the database "
                         "cannot answer. By default --assess checks them "
                         "automatically (one LLM call per Unknown pair, "
                         "verdict stored in MongoDB) when an API key or "
                         "provider is configured; without one it says so "
                         "once and assesses offline.")
    ap.add_argument("--out", default="", help="Write the intake JSON to this file.")
    args = ap.parse_args()

    if args.demo_db:
        drugdb = DrugDB(_demo_db())
        print("Using the demo database (7 brands) — drop --demo-db to "
              "use your real MongoDB data.")
    else:
        # MongoDB is the default: --mongo-uri / --db, or MONGO_URI /
        # MONGO_DB from backend/.env (see .env.example).
        drugdb = _open_mongo_drugdb(args.mongo_uri or None, args.db or None)
    mode = "llm" if args.handwritten else args.mode
    suggest_path = None
    if args.auto_add_missing:
        from lookup_medicine import DEFAULT_OUT as _SUGGEST_PATH
        suggest_path = _SUGGEST_PATH
    result = run_intake(args.image, mode=mode, drugdb=drugdb,
                        threshold=args.threshold, lang=args.lang,
                        api_key=args.api_key or None,
                        base_url=args.base_url or None,
                        model=args.model or None,
                        vision_api_key=args.vision_api_key or None,
                        vision_base_url=args.vision_base_url or None,
                        vision_model=args.vision_model or None,
                        auto_add_missing=args.auto_add_missing,
                        suggest_path=suggest_path)
    labelled = {args.list: result}
    if args.out:
        Path(args.out).write_text(json.dumps(labelled, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"Wrote {args.out}")
    print(json.dumps(labelled, indent=2, ensure_ascii=False))
    if result["needs_confirmation"]:
        print(f"\n{len(result['needs_confirmation'])} item(s) need confirmation "
              "before the safety check can trust them.")
    if args.assess:
        web_verify = None
        if not args.no_web_verify:
            if _web_check_configured(args.api_key, args.base_url, args.model):
                from web_verify import web_verify_pair
                web_verify = lambda a, b: web_verify_pair(
                    drugdb, a, b, api_key=args.api_key or None,
                    base_url=args.base_url or None, model=args.model or None)
            else:
                print("No API key or provider configured — web checks "
                      "skipped, Unknown pairs stay Unknown. Set "
                      "GEMINI_API_KEY in backend/.env or pass --api-key "
                      "(or LLM_BASE_URL + LLM_MODEL for a local provider).")
        report = assess_intake(result, drugdb, web_verify=web_verify)
        if report is None:
            print("\nNOT ASSESSED: no confirmed medicines — every candidate is in "
                  "needs_confirmation, and uncertain items are never passed to "
                  "the engine as fact. Confirm them first.")
            return 1
        print("\nMedicine Safety Report (deterministic engine — rules decide):")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
