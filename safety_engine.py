#!/usr/bin/env python3
"""RxSentinel — Phase 2: the safety engine (no UI).

Pure rule-based checks over the Phase 1 MongoDB data. Nothing here is
an LLM judgement: the same medicines always give the same findings.
The LLM/RAG layer (Phase 4) only EXPLAINS these findings afterwards.

Checks, in order
---------------
1. Duplicate ingredient — the same active ingredient in two or more
   medicines, including hidden in combination products
   (Napa + Napa Extra both contain paracetamol). Found from the
   `ingredients[]` array, never from the brand name.
2. Drug-drug interaction — every pair of ingredients looked up in the
   `interactions` collection (DDInter 2.0), severity Major / Moderate /
   Minor. A pair with NO record is reported as Unknown — never as safe.
   The one exception is severity "None": a curated, source-labelled
   verdict that no known interaction exists (data/curated_no_interaction.csv)
   — checked-and-clear, no finding.
3. Total daily dose load — per ingredient, mg per dose x doses per day,
   summed across all medicines, compared with the max-daily-dose table.
   Over the max is High; 75-100% of the max is a Warning.
4. Expiry & batch — when a medicine carries an expiry date (from the
   pack or receipt, Phase 3), an expired medicine is High and one
   expiring within 30 days is a Warning. Medicines with no expiry
   supplied are counted in coverage as not checked — never assumed
   fresh. A batch number, when supplied, is carried into the finding.

Risk levels (overall)
---------------------
High    — any Major interaction, or any ingredient over its daily max.
Warning — any Moderate interaction, any duplicate, or any ingredient
          at 75-100% of its daily max.
Unknown — no High/Warning finding, but some pair or ingredient has no
          data to check against. Shown as Unknown, never as Low/safe.
Low     — no findings and everything was checkable.

Input — one medicine per line of a prescription:
    {"brand": "Napa Extra", "doses_per_day": 2}
    {"brand": "Napa", "doses_per_day": 2, "days": 5}
    {"generic": "Paracetamol", "strength_mg": 500, "doses_per_day": 3}
Brand names resolve through the brands collection; unknown names are
reported as unresolved (never guessed) and the check continues with
the rest. Strengths come from the brand's `strength_parts`, aligned
by position with its `ingredients[]`.

Usage
-----
    # Demo on built-in sample data (no MongoDB needed):
    python safety_engine.py --demo

    # Check a prescription JSON file against your MongoDB:
    python safety_engine.py --mongo-uri "mongodb+srv://..." --input prescription.json

    # prescription.json:
    #   {"patient": "demo", "medicines": [{"brand": "Napa Extra",
    #    "doses_per_day": 2}, {"brand": "Napa", "doses_per_day": 2}]}
    #
    # Expiry (from a pack/receipt) is optional per medicine:
    #   {"brand": "Napa", "doses_per_day": 2, "expiry": "2027-03",
    #    "batch": "B12345"}
    # Accepted expiry formats: "2027-03", "03/2027", "2027-03-15",
    # "Mar 2027", "15 Mar 2027". A bare month means the end of that
    # month (packs print expiry as month/year).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from itertools import combinations
from pathlib import Path

SEVERITY_RANK = {"Minor": 1, "Moderate": 2, "Major": 3, "Unknown": 0}


def norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


# Salt / ester suffixes on Bangladesh pack names. DDInter (and the
# max-dose table) use the base drug name — "Cefuroxime", not
# "Cefuroxime axetil"; "Metformin", not "Metformin hydrochloride".
# Stripping these is what makes the lookup hit at all: of the 1,256
# missing pairs in the top-50 audit (Nov 15), 282 are in DDInter
# once the salt form is stripped. Only suffixes that are never the
# drug itself are listed — "sulfate", "acetate", "citrate" and
# "phosphate" are deliberately NOT here ("zinc acetate",
# "potassium citrate" are drugs in their own right).
SALT_SUFFIXES = (
    "axetil", "trihydrate", "dihydrate", "hemihydrate", "monohydrate",
    "hydrochloride", "tromethamine", "fumarate", "maleate", "magnesium",
    "sodium", "calcium", "potassium", "proxetil", "mesylate", "besylate",
    "tartrate", "gluconate", "succinate", "hydrobromide", "nitrate",
    "bisulfate", "bisulphate",
)


def base_drug_name(name: str) -> str:
    """'Cefuroxime axetil' -> 'cefuroxime'; 'Ketotifen fumarate (oral)'
    -> 'ketotifen'. Names with no salt suffix come back unchanged."""
    n = re.sub(r"\s*[\(\[].*?[\)\]]\s*$", "", (name or "").strip().lower())
    parts = re.sub(r"\s+", " ", n).split()
    while len(parts) > 1 and parts[-1] in SALT_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def display_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).title()


def parse_strength_mg(part: str) -> float | None:
    """'500 mg' -> 500.0; '0.5 g' -> 500.0; '65mg' -> 65.0; 'N/A' -> None."""
    if not part:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mg|g|mcg|µg|iu)\b", part, re.I)
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2).lower()
    if unit == "g":
        return value * 1000
    if unit in ("mcg", "µg"):
        return value / 1000
    if unit == "iu":
        return None  # IU is not a mass — cannot be summed as mg.
    return value


_MONTHS = {name: i for i, name in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def parse_expiry(value) -> date | None:
    """Parse a pack expiry into a date, or None if unparseable.

    A bare month/year means the END of that month (packs print expiry
    as month/year, and the medicine is good through that month).
    Accepts: date/datetime objects, "2027-03", "03/2027", "2027-03-15",
    "Mar 2027", "15 Mar 2027", "2027/03". Anything else returns None
    so the caller can flag it as unparseable instead of guessing.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})(?:-(\d{2}))?", text)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if m.group(3):
            try:
                return date(year, month, int(m.group(3)))
            except ValueError:
                return None
        return _month_end(year, month)
    m = re.fullmatch(r"(\d{1,2})/(\d{4})", text)
    if m:
        return _month_end(int(m.group(2)), int(m.group(1)))
    m = re.fullmatch(r"(\d{4})/(\d{1,2})", text)
    if m:
        return _month_end(int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"([A-Za-z]{3,9})\.?\s+(\d{4})", text)
    if m and m.group(1)[:3].lower() in _MONTHS:
        return _month_end(int(m.group(2)), _MONTHS[m.group(1)[:3].lower()])
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]{3,9})\.?\s+(\d{4})", text)
    if m and m.group(2)[:3].lower() in _MONTHS:
        try:
            return date(int(m.group(3)), _MONTHS[m.group(2)[:3].lower()],
                        int(m.group(1)))
        except ValueError:
            return None
    return None


def _month_end(year: int, month: int) -> date | None:
    if not (1 <= month <= 12):
        return None
    if month == 12:
        return date(year, 12, 31)
    return date.fromordinal(date(year, month + 1, 1).toordinal() - 1)


# ---------------------------------------------------------------- DB layer
def _tolerant_name_pattern(name: str) -> str | None:
    """Regex for a drug name that ignores spaces, hyphens and
    underscores, so 'PanD' matches a database entry stored as 'Pan D'
    (and vice versa). Anchored, so 'Napa' never matches 'Napa Extra';
    every letter must still match, in order — this is punctuation
    tolerance, not fuzzy guessing."""
    chars = [c for c in (name or "").strip() if c.isalnum()]
    if not chars:
        return None
    sep = r"[\s\-_]*"
    return "^" + sep.join(re.escape(c) for c in chars) + sep + "$"


class DrugDB:
    """Thin adapter over the Phase 1 collections. Works with pymongo,
    mongomock, or any object exposing the same find_one/find calls."""

    def __init__(self, db, aliases: dict[str, str] | None = None):
        self.db = db
        self._alias_map = aliases  # norm alias -> canonical (for tests)
        self._alias_cache: dict[str, str] | None = None

    def canonical(self, ingredient: str) -> str:
        """Map an ingredient to its canonical name (Paracetamol for
        Acetaminophen). The aliases collection is tiny, so it is loaded
        once and matched in Python — exact and case-insensitive."""
        n = base_drug_name(ingredient)
        if self._alias_map is not None:
            return self._alias_map.get(n, n)
        if self._alias_cache is None:
            self._alias_cache = {}
            for doc in self.db.aliases.find({}):
                canon = norm_name(doc.get("canonical", ""))
                if canon:
                    self._alias_cache[canon] = canon
                for a in doc.get("aliases", []) or []:
                    self._alias_cache[norm_name(a)] = canon
        return self._alias_cache.get(n, n)

    def resolve_brand(self, name: str) -> dict | None:
        doc = self.db.brands.find_one({"brand_name": name})
        if doc:
            return doc
        doc = self.db.brands.find_one(
            {"brand_name": {"$regex": f"^{re.escape(name)}$", "$options": "i"}})
        if doc:
            return doc
        pattern = _tolerant_name_pattern(name)
        if pattern:
            return self.db.brands.find_one(
                {"brand_name": {"$regex": pattern, "$options": "i"}})
        return None

    def resolve_generic(self, name: str) -> dict | None:
        doc = self.db.generics.find_one(
            {"title": {"$regex": f"^{re.escape(name)}$", "$options": "i"}})
        if doc:
            return doc
        pattern = _tolerant_name_pattern(name)
        if pattern:
            return self.db.generics.find_one(
                {"title": {"$regex": pattern, "$options": "i"}})
        return None

    def find_interaction(self, a: str, b: str) -> dict | None:
        pair = sorted([base_drug_name(a), base_drug_name(b)])
        return self.db.interactions.find_one(
            {"drug_a": pair[0], "drug_b": pair[1]})


def describe_medicine(drugdb: DrugDB, name: str) -> dict:
    """What the database knows about a name — the --check-brand
    diagnostic. Tries brand first (tolerant matching), then generic.
    A missing brand is a dataset gap, not a guess: found=False means
    the engine will flag it unresolved, never invent ingredients."""
    doc = drugdb.resolve_brand(name)
    if doc:
        return {"found": True, "kind": "brand",
                "name": doc.get("brand_name") or name,
                "generic_title": doc.get("generic_title"),
                "ingredients": [str(i) for i in doc.get("ingredients") or []],
                "strength_parts": [str(p) for p in doc.get("strength_parts") or []]}
    doc = drugdb.resolve_generic(name)
    if doc:
        return {"found": True, "kind": "generic",
                "name": doc.get("title") or name,
                "generic_title": doc.get("title"),
                "ingredients": [str(i) for i in doc.get("ingredients") or []],
                "strength_parts": [str(p) for p in doc.get("strength_parts") or []]}
    return {"found": False, "kind": None, "name": name,
            "generic_title": None, "ingredients": [], "strength_parts": []}


# ------------------------------------------------------------ medicine input
def expand_medicine(drugdb: DrugDB, med: dict) -> dict:
    """Resolve one prescribed medicine to its ingredients + mg per dose.

    Returns {label, ingredients: [{ingredient, mg_per_dose}], unresolved, note}.
    """
    label = med.get("brand") or med.get("generic") or med.get("name") or "?"
    doses = med.get("doses_per_day")
    out: dict = {"label": label, "doses_per_day": doses,
                 "expiry": med.get("expiry") or med.get("expiry_date")
                 or med.get("exp"),
                 "batch": med.get("batch") or med.get("batch_no"),
                 "ingredients": [], "unresolved": False, "note": ""}
    # Phase 4 (§5): an explicit `ingredients` list — supplied by the API
    # from a source-verified web draft — is used directly instead of the
    # DB lookup, so a brand the database does not carry can still be
    # checked for this one assessment. Per-ingredient mg comes from
    # `strength_parts` via parse_strength_mg; a missing or unparseable
    # part gives mg None and the dose check skips it, exactly like the
    # DB path below. Nothing is written to the database here.
    explicit = med.get("ingredients")
    if isinstance(explicit, list) and explicit:
        parts = [str(p) for p in (med.get("strength_parts") or [])]
        for i, ing in enumerate(explicit):
            mg = parse_strength_mg(parts[i]) if i < len(parts) else None
            out["ingredients"].append({"ingredient": drugdb.canonical(str(ing)),
                                       "mg_per_dose": mg})
        return out
    doc = None
    if med.get("brand"):
        doc = drugdb.resolve_brand(med["brand"])
    if doc is None and med.get("generic"):
        doc = drugdb.resolve_generic(med["generic"])
    if doc is None and med.get("name"):
        doc = drugdb.resolve_brand(med["name"]) or drugdb.resolve_generic(med["name"])
    if doc is None:
        out["unresolved"] = True
        out["note"] = f"'{label}' is not in the medicine database — not guessed."
        if med.get("strength_mg") and med.get("generic"):
            out["ingredients"] = [{"ingredient": drugdb.canonical(med["generic"]),
                                   "mg_per_dose": float(med["strength_mg"])}]
            out["note"] += " Using the stated generic and strength as given."
        return out
    ings = [str(i) for i in doc.get("ingredients") or []]
    parts = [str(p) for p in doc.get("strength_parts") or []]
    if med.get("strength_mg") is not None and len(ings) == 1:
        out["ingredients"] = [{"ingredient": drugdb.canonical(ings[0]),
                               "mg_per_dose": float(med["strength_mg"])}]
    else:
        for i, ing in enumerate(ings):
            mg = parse_strength_mg(parts[i]) if i < len(parts) else None
            out["ingredients"].append({"ingredient": drugdb.canonical(ing),
                                       "mg_per_dose": mg})
    return out


# ------------------------------------------------------------------ checks
def check_duplicates(expanded: list[dict]) -> list[dict]:
    """Same ingredient (canonical) in two or more medicines."""
    by_ing: dict[str, list[str]] = {}
    for med in expanded:
        for ing in med["ingredients"]:
            by_ing.setdefault(ing["ingredient"], [])
            if med["label"] not in by_ing[ing["ingredient"]]:
                by_ing[ing["ingredient"]].append(med["label"])
    findings = []
    for ing, labels in sorted(by_ing.items()):
        if len(labels) > 1:
            findings.append({
                "code": "duplicate_ingredient", "severity": "Warning",
                "ingredient": display_name(ing), "medicines": labels,
                "title": f"Duplicate ingredient: {display_name(ing)}",
                "detail": (f"{display_name(ing)} appears in {len(labels)} medicines "
                           f"({', '.join(labels)}). Taken together they can push the "
                           f"total daily dose over the safe limit without it being obvious."),
                "action": "Ask your pharmacist or doctor whether both are needed.",
            })
    return findings


# Severities a stored or web-verified row can carry. Anything else
# ("Unknown", or a value no source graded) means the pair still needs
# an answer — it is never treated as safe or as a finding on its own.
GRADED_SEVERITIES = ("Major", "Moderate", "Minor", "None")


def check_interactions(drugdb: DrugDB, expanded: list[dict],
                       web_verify=None) -> tuple[list[dict], dict]:
    """Every unordered ingredient pair spanning two DIFFERENT medicines,
    looked up in the interactions collection. `web_verify` is an optional
    callable (a, b) -> row-or-None (see web_verify.web_verify_pair): it
    is called ONLY for a pair the database cannot answer — missing, or
    stored ungraded Unknown — and a graded row it returns is used exactly
    like a database row, so a web-verified "None" clears the Unknown and
    a graded interaction is reported with its source. A failed check (no
    key, quota, outage) never breaks the assessment: the pair simply
    stays Unknown.

    Pairs inside ONE medicine are NOT checked: the two ingredients of a
    combination tablet (Amoxicillin + Clavulanic acid in Augmentin,
    Paracetamol + Caffeine in Napa Extra) are co-formulated by design,
    not a drug-drug interaction — checking them as one produced a false
    Unknown for every combination product. The dose-load and duplicate
    checks still count both ingredients; only the interaction lookup
    skips them. A pair that occurs both inside one medicine AND across
    two medicines is still checked (the cross-medicine occurrence)."""
    pair_set: set[tuple[str, str]] = set()
    for first, second in combinations(expanded, 2):
        for a in {i["ingredient"] for i in first["ingredients"]}:
            for b in {i["ingredient"] for i in second["ingredients"]}:
                if a != b:
                    pair_set.add(tuple(sorted((a, b))))
    findings, checked, with_data = [], 0, 0
    for a, b in sorted(pair_set):
        checked += 1
        doc = drugdb.find_interaction(a, b)
        if doc is None or doc.get("severity") not in GRADED_SEVERITIES:
            if web_verify is not None:
                try:
                    verified = web_verify(a, b)
                except (Exception, SystemExit):
                    verified = None
                if verified and verified.get("severity") in GRADED_SEVERITIES:
                    doc = verified
        if doc is None:
            findings.append({
                "code": "interaction_unknown", "severity": "Unknown",
                "ingredient_a": display_name(a), "ingredient_b": display_name(b),
                "title": f"No interaction data: {display_name(a)} + {display_name(b)}",
                "detail": ("No interaction record was found for this pair. That means "
                           "NOT CHECKED — not safe. The pair is outside the interaction "
                           "database's coverage."),
                "action": "Ask your pharmacist or doctor about this combination.",
            })
            continue
        with_data += 1
        sev = doc.get("severity", "Unknown")
        if sev == "None":
            # Curated verdict: no known interaction for this pair
            # (data/curated_no_interaction.csv, source-labelled). The pair
            # is checked and clear — no finding, but it counts as having
            # data, so it never reads as "not checked". Do not emit an
            # "interaction" finding for it: that would flag a safe pair.
            continue
        finding = {
            "code": "interaction", "severity": sev,
            "ingredient_a": display_name(a), "ingredient_b": display_name(b),
            "title": f"{sev} interaction: {display_name(a)} + {display_name(b)}",
            "detail": doc.get("mechanism", ""),
            "management": doc.get("management", ""),
            "source": doc.get("source", "DDInter 2.0"),
            "action": ("Ask your pharmacist or doctor before taking these together."
                       if sev in ("Major", "Moderate") else
                       "Usually manageable — still mention it to your pharmacist."),
        }
        if doc.get("verified_via") == "web":
            # The report says plainly that this verdict came from a
            # web check at assessment time, not from DDInter.
            finding["verified_via"] = "web"
        if doc.get("mechanism_source"):
            # Severity is DDInter's (or the web check's); the mechanism
            # text may come from elsewhere — say where, in the report.
            finding["mechanism_source"] = doc["mechanism_source"]
        findings.append(finding)
    return findings, {"pairs_checked": checked, "pairs_with_data": with_data}


def check_dose_load(expanded: list[dict], max_doses: dict) -> tuple[list[dict], dict]:
    """Per-ingredient total daily mg vs the max-daily-dose table."""
    totals: dict[str, dict] = {}
    for med in expanded:
        doses = med.get("doses_per_day")
        if not doses:
            continue
        for ing in med["ingredients"]:
            mg = ing.get("mg_per_dose")
            if mg is None:
                continue
            slot = totals.setdefault(ing["ingredient"], {"mg": 0.0, "medicines": []})
            slot["mg"] += mg * float(doses)
            if med["label"] not in slot["medicines"]:
                slot["medicines"].append(med["label"])
    findings, with_max = [], 0
    no_max: list[str] = []
    for ing, slot in sorted(totals.items()):
        entry = max_doses.get(ing)
        if entry is None:
            # One aggregated finding, not one per drug — otherwise every
            # prescription would drown in "no maximum on file" notes.
            no_max.append(display_name(ing))
            continue
        with_max += 1
        limit = float(entry["max_mg_per_day"])
        pct = slot["mg"] / limit * 100 if limit else 0
        if slot["mg"] > limit:
            findings.append({
                "code": "dose_over_max", "severity": "High",
                "ingredient": display_name(ing), "medicines": slot["medicines"],
                "title": (f"Over daily maximum: {display_name(ing)} "
                          f"({slot['mg']:.0f} mg/day vs max {limit:.0f} mg)"),
                "detail": (f"Across {', '.join(slot['medicines'])} the total is "
                           f"{slot['mg']:.0f} mg per day — {pct:.0f}% of the reference "
                           f"maximum ({limit:.0f} mg). {entry.get('note', '')}"),
                "action": "Do not take this combination as-is — ask your pharmacist or doctor today.",
            })
        elif pct >= 75:
            findings.append({
                "code": "dose_near_max", "severity": "Warning",
                "ingredient": display_name(ing), "medicines": slot["medicines"],
                "title": (f"Near daily maximum: {display_name(ing)} "
                          f"({slot['mg']:.0f} mg/day, {pct:.0f}% of max {limit:.0f} mg)"),
                "detail": ("One more dose, or one more paracetamol-containing medicine, "
                           "would push this over the reference maximum."),
                "action": "Check with your pharmacist before adding any similar medicine.",
            })
    if no_max:
        findings.append({
            "code": "dose_max_unknown", "severity": "Unknown",
            "ingredient": ", ".join(no_max),
            "title": f"No maximum-dose data: {', '.join(no_max)}",
            "detail": ("No reference daily maximum is on file for these ingredients, "
                       "so their total daily dose cannot be judged safe or unsafe. "
                       "That means NOT CHECKED — not safe."),
            "action": "Ask your pharmacist or doctor what the daily limit is.",
        })
    return findings, {"ingredients_with_max": with_max,
                      "ingredients_totalled": len(totals)}


EXPIRY_WARNING_DAYS = 30


def check_expiry(expanded: list[dict],
                 today: date | None = None) -> tuple[list[dict], dict]:
    """Expired -> High; expiring within 30 days -> Warning.

    Only medicines WITH an expiry supplied are checked; the rest are
    reported in coverage as not checked (never assumed fresh). An
    expiry value that cannot be parsed is an Unknown finding — it is
    flagged, never silently dropped.
    """
    today = today or date.today()
    findings, checked, unparseable = [], 0, 0
    for med in expanded:
        raw = med.get("expiry")
        if raw in (None, ""):
            continue
        exp = parse_expiry(raw)
        batch_note = (f" Batch {med['batch']}." if med.get("batch") else "")
        if exp is None:
            unparseable += 1
            findings.append({
                "code": "expiry_unparseable", "severity": "Unknown",
                "medicine": med["label"],
                "title": f"Could not read expiry date: {med['label']}",
                "detail": (f"The expiry value '{raw}' could not be understood, "
                           f"so this medicine's expiry was NOT checked. "
                           f"Check the pack and enter it as YYYY-MM.{batch_note}"),
                "action": "Read the expiry date off the pack and re-enter it.",
            })
            continue
        checked += 1
        days_left = (exp - today).days
        if exp < today:
            findings.append({
                "code": "expired", "severity": "High",
                "medicine": med["label"], "expiry": exp.isoformat(),
                "title": f"Expired: {med['label']} (expired {exp.isoformat()})",
                "detail": (f"This medicine expired on {exp.isoformat()} "
                           f"({-days_left} days ago). Do not take it.{batch_note}"),
                "action": "Do not take this medicine. Return it to the pharmacy.",
            })
        elif days_left <= EXPIRY_WARNING_DAYS:
            findings.append({
                "code": "expiring_soon", "severity": "Warning",
                "medicine": med["label"], "expiry": exp.isoformat(),
                "days_left": days_left,
                "title": (f"Expiring soon: {med['label']} "
                          f"(expires {exp.isoformat()}, in {days_left} days)"),
                "detail": ("Finish this pack before it expires, and do not "
                           f"buy a replacement until you have checked the "
                           f"new pack's expiry.{batch_note}"),
                "action": "Use it before the expiry date; check new packs.",
            })
    return findings, {"expiry_checked": checked,
                      "expiry_unparseable": unparseable,
                      "expiry_total": len(expanded)}


# ------------------------------------------------------------------ assess
def assess(drugdb: DrugDB, medicines: list[dict], max_doses: dict,
           today: date | None = None, web_verify=None) -> dict:
    """The full assessment. `web_verify` is passed straight through to
    check_interactions — None keeps the engine fully deterministic
    (no network, no LLM), a callable lets Unknown pairs be answered
    by a web check at assessment time (see web_verify.web_verify_pair)."""
    expanded = [expand_medicine(drugdb, m) for m in medicines]
    findings: list[dict] = []
    for med in expanded:
        if med["unresolved"]:
            findings.append({
                "code": "unresolved_medicine", "severity": "Unknown",
                "title": f"Could not identify: {med['label']}",
                "detail": med["note"],
                "action": "Check the spelling, or add this medicine to the database.",
            })
    findings += check_duplicates(expanded)
    inter_findings, inter_cov = check_interactions(drugdb, expanded, web_verify)
    findings += inter_findings
    dose_findings, dose_cov = check_dose_load(expanded, max_doses)
    findings += dose_findings
    exp_findings, exp_cov = check_expiry(expanded, today=today)
    findings += exp_findings

    # Two scales meet here: findings carry their own severity
    # ("Major"/"Moderate"/"Minor" for interactions, "High"/"Warning" for
    # dose and duplicate findings, "Unknown" for no-data). Map both onto
    # the overall risk level — a Major interaction IS a High risk.
    severities = [f["severity"] for f in findings]
    if any(s in ("High", "Major") for s in severities):
        level = "High"
    elif any(s in ("Warning", "Moderate") for s in severities):
        level = "Warning"
    elif "Unknown" in severities:
        level = "Unknown"
    else:
        level = "Low"
    return {
        "risk_level": level,
        "medicines_checked": len(medicines),
        "medicines_resolved": sum(1 for m in expanded if not m["unresolved"]),
        "findings": findings,
        "counts": {s: severities.count(s) for s in sorted(set(severities))},
        "coverage": {**inter_cov, **dose_cov, **exp_cov},
        "disclaimer": ("For information only — not medical advice. "
                       "Ask your pharmacist or doctor before changing any medicine."),
    }


def load_max_doses(path: Path | None = None) -> dict:
    if path is None:
        # Look in the same folder as safety_engine.py first (root),
        # then in data/ — both layouts work, root wins if both exist.
        base = Path(__file__).resolve().parent
        for candidate in (base / "max_daily_dose.json",
                          base / "data" / "max_daily_dose.json"):
            if candidate.is_file():
                path = candidate
                break
        else:
            path = base / "max_daily_dose.json"
    if not path.is_file():
        # Never fail silently here: a missing file used to make EVERY
        # ingredient report "no maximum-dose data", which looked like
        # the table was empty. Say so loudly instead.
        print(f"WARNING: max-dose file not found: {path}\n"
              "Dose check will report Unknown for all ingredients. "
              "Put max_daily_dose.json next to safety_engine.py "
              "(or in data/), or pass --max-doses <path>.", file=sys.stderr)
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {norm_name(k): v for k, v in raw.get("max_mg_per_day", {}).items()}


# --------------------------------------------------------------------- demo
def _demo_db():
    """In-memory stand-in for the MongoDB collections (tests + --demo)."""
    class Coll:
        def __init__(self, docs):
            self.docs = docs

        def find(self, query=None):
            return list(self.docs)

        def find_one(self, query):
            for d in self.docs:
                ok = True
                for k, v in query.items():
                    if isinstance(v, dict) and "$regex" in v:
                        if not re.search(v["$regex"], str(d.get(k, "")), re.I):
                            ok = False
                    elif isinstance(v, dict) and "$in" in v:
                        if not any(x in d.get(k, []) for x in v["$in"]):
                            ok = False
                    elif d.get(k) != v:
                        ok = False
                if ok:
                    return d
            return None

    class DB:
        pass

    db = DB()
    db.brands = Coll([
        {"brand_name": "Napa", "generic_title": "Paracetamol",
         "ingredients": ["Paracetamol"], "strength_parts": ["500 mg"]},
        {"brand_name": "Napa Extra", "generic_title": "Paracetamol + Caffeine",
         "ingredients": ["Paracetamol", "Caffeine"], "strength_parts": ["500 mg", "65 mg"]},
        {"brand_name": "Seclo", "generic_title": "Omeprazole",
         "ingredients": ["Omeprazole"], "strength_parts": ["20 mg"]},
        {"brand_name": "Plavix", "generic_title": "Clopidogrel",
         "ingredients": ["Clopidogrel"], "strength_parts": ["75 mg"]},
        {"brand_name": "Warfarin", "generic_title": "Warfarin",
         "ingredients": ["Warfarin"], "strength_parts": ["5 mg"]},
        {"brand_name": "Aspirin", "generic_title": "Aspirin",
         "ingredients": ["Aspirin"], "strength_parts": ["75 mg"]},
        {"brand_name": "Ace Plus", "generic_title": "Paracetamol + Caffeine",
         "ingredients": ["Paracetamol", "Caffeine"], "strength_parts": ["500 mg", "65 mg"]},
    ])
    db.generics = Coll([
        {"title": "Acetaminophen", "ingredients": ["Acetaminophen"],
         "strength_parts": ["500 mg"]},
    ])
    db.interactions = Coll([
        {"drug_a": "clopidogrel", "drug_b": "omeprazole", "severity": "Major",
         "mechanism": "Omeprazole inhibits CYP2C19, reducing clopidogrel activation and its antiplatelet effect.",
         "management": "Consider an alternative acid reducer; discuss with the prescriber.",
         "source": "DDInter 2.0 (demo seed)"},
        {"drug_a": "aspirin", "drug_b": "warfarin", "severity": "Major",
         "mechanism": "Additive bleeding risk: aspirin impairs platelets and irritates the stomach lining while warfarin thins the blood.",
         "management": "Avoid the combination unless specifically directed; watch for bruising or bleeding.",
         "source": "DDInter 2.0 (demo seed)"},
        # Verified no known interaction (Drugs.com checker, 2026-11-14) —
        # severity "None" means checked-and-clear, not ungraded. These are
        # the three pairs from the Napa Extra + Seclo test, which DDInter
        # only lists as Unknown.
        {"drug_a": "caffeine", "drug_b": "omeprazole", "severity": "None",
         "mechanism": "Drugs.com interaction checker: no drug interactions found between caffeine and omeprazole.",
         "management": "No action — still mention it to your pharmacist if unsure.",
         "source": "Drugs.com interaction checker (curated, 2026-11-14)"},
        {"drug_a": "caffeine", "drug_b": "paracetamol", "severity": "None",
         "mechanism": "Drugs.com interaction checker: no drug interactions found between caffeine and Paracetamol. Also co-formulated in one tablet (Napa Extra).",
         "management": "No action — still mention it to your pharmacist if unsure.",
         "source": "Drugs.com interaction checker (curated, 2026-11-14)"},
        {"drug_a": "omeprazole", "drug_b": "paracetamol", "severity": "None",
         "mechanism": "Drugs.com interaction checker: no drug interactions found between omeprazole and Paracetamol.",
         "management": "No action — still mention it to your pharmacist if unsure.",
         "source": "Drugs.com interaction checker (curated, 2026-11-14)"},
    ])
    db.aliases = Coll([{"canonical": "Paracetamol", "aliases": ["Acetaminophen"]}])
    return db


def demo_cases():
    # Expiring-soon uses the current month-end (always <= 31 days away,
    # never already past), so the case stays "expiring soon" whenever
    # the demo is run.
    today = date.today()
    soon = f"{today.month:02d}/{today.year}"
    return [
        ("Hidden duplicate — Napa + Napa Extra (same paracetamol, twice)",
         [{"brand": "Napa", "doses_per_day": 2},
          {"brand": "Napa Extra", "doses_per_day": 2}]),
        ("Major interaction — Seclo (omeprazole) + Plavix (clopidogrel)",
         [{"brand": "Seclo", "doses_per_day": 1},
          {"brand": "Plavix", "doses_per_day": 1}]),
        ("Over the daily max — three paracetamol products, 3x a day each",
         [{"brand": "Napa", "doses_per_day": 3},
          {"brand": "Napa Extra", "doses_per_day": 3},
          {"brand": "Ace Plus", "doses_per_day": 3}]),
        ("Clean case — Napa alone, normal dose",
         [{"brand": "Napa", "doses_per_day": 2}]),
        ("Unknown medicine — name not in the database",
         [{"brand": "Napa", "doses_per_day": 2},
          {"brand": "MadeUpBrand", "doses_per_day": 1}]),
        ("Expired pack — Napa Extra with a past expiry (batch shown)",
         [{"brand": "Napa Extra", "doses_per_day": 2,
           "expiry": "2020-01", "batch": "B12345"}]),
        ("Expiring soon — Napa expiring this month",
         [{"brand": "Napa", "doses_per_day": 2, "expiry": soon}]),
    ]


DEMO_CASES = demo_cases()


def main() -> int:
    ap = argparse.ArgumentParser(description="RxSentinel Phase 2 — safety engine.")
    ap.add_argument("--demo", action="store_true", help="Run the built-in demo cases (no MongoDB).")
    ap.add_argument("--input", help="Prescription JSON file to check.")
    ap.add_argument("--mongo-uri", default="mongodb://127.0.0.1:27017")
    ap.add_argument("--db", default="rxsentinel")
    ap.add_argument("--max-doses", default="", help="Path to max_daily_dose.json.")
    ap.add_argument("--web-verify", action="store_true",
                    help="Web-check pairs the database cannot answer, then "
                         "store the verdict (needs an API key — see "
                         "web_verify.py). Off by default: the engine is "
                         "deterministic with no network calls.")
    ap.add_argument("--check-brand", action="append", default=[],
                    metavar="NAME",
                    help="Check whether a brand/generic is in the database "
                         "and what ingredients it carries (repeatable). "
                         "Use --demo to check against the demo list, or "
                         "--mongo-uri/--db for the real database.")
    args = ap.parse_args()

    if args.max_doses and not Path(args.max_doses).is_file():
        print(f"ERROR: max-dose file not found: {args.max_doses}\n"
              "Check the path — run from the backend folder, or give the "
              "full path, e.g. --max-doses data/max_daily_dose.json",
              file=sys.stderr)
        return 2
    max_doses = load_max_doses(Path(args.max_doses) if args.max_doses else None)
    if args.demo:
        drugdb = DrugDB(_demo_db())
        if args.check_brand:
            for name in args.check_brand:
                print(json.dumps(describe_medicine(drugdb, name),
                                 indent=2, ensure_ascii=False))
            return 0
        for title, meds in DEMO_CASES:
            print(f"\n=== {title} ===")
            print(json.dumps(assess(drugdb, meds, max_doses), indent=2, ensure_ascii=False))
        return 0
    if args.check_brand and not args.input:
        from pymongo import MongoClient
        drugdb = DrugDB(MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)[args.db])
        for name in args.check_brand:
            print(json.dumps(describe_medicine(drugdb, name),
                             indent=2, ensure_ascii=False))
        return 0
    if not args.input:
        print("Give --demo or --input prescription.json (see the module docstring).")
        return 1
    from pymongo import MongoClient
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    drugdb = DrugDB(MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)[args.db])
    web_verify = None
    if args.web_verify:
        from web_verify import web_verify_pair
        web_verify = lambda a, b: web_verify_pair(drugdb, a, b)
    print(json.dumps(assess(drugdb, data.get("medicines", []), max_doses,
                            web_verify=web_verify),
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
