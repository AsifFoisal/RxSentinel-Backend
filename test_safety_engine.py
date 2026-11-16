#!/usr/bin/env python3
"""RxSentinel — Phase 2 tests. Offline, no MongoDB needed.

Run:  python test_safety_engine.py
"""
import json
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from safety_engine import (  # noqa: E402
    DrugDB, _demo_db, assess, check_dose_load, check_duplicates,
    check_expiry, check_interactions, describe_medicine, expand_medicine,
    load_max_doses, parse_expiry, parse_strength_mg,
)


class TestStrengthParsing(unittest.TestCase):
    def test_units(self):
        self.assertEqual(parse_strength_mg("500 mg"), 500.0)
        self.assertEqual(parse_strength_mg("0.5 g"), 500.0)
        self.assertEqual(parse_strength_mg("65mg"), 65.0)
        self.assertEqual(parse_strength_mg("250 mcg"), 0.25)
        self.assertIsNone(parse_strength_mg("N/A"))
        self.assertIsNone(parse_strength_mg("100 IU"))  # IU is not a mass


class TestExpiryParsing(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(parse_expiry("2027-03"), date(2027, 3, 31))
        self.assertEqual(parse_expiry("03/2027"), date(2027, 3, 31))
        self.assertEqual(parse_expiry("2027-03-15"), date(2027, 3, 15))
        self.assertEqual(parse_expiry("Mar 2027"), date(2027, 3, 31))
        self.assertEqual(parse_expiry("15 Mar 2027"), date(2027, 3, 15))
        self.assertEqual(parse_expiry("2027/3"), date(2027, 3, 31))
        self.assertEqual(parse_expiry(date(2027, 3, 15)), date(2027, 3, 15))

    def test_unparseable_returns_none(self):
        self.assertIsNone(parse_expiry("whenever"))
        self.assertIsNone(parse_expiry("2027-13"))
        self.assertIsNone(parse_expiry(""))
        self.assertIsNone(parse_expiry(None))


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.db = DrugDB(_demo_db())
        self.max_doses = load_max_doses()

    def codes(self, result):
        return [f["code"] for f in result["findings"]]

    def test_hidden_duplicate_napa_plus_napa_extra(self):
        """The core demo: paracetamol hidden in two brands must be caught."""
        result = assess(self.db, [
            {"brand": "Napa", "doses_per_day": 2},
            {"brand": "Napa Extra", "doses_per_day": 2},
        ], self.max_doses)
        self.assertEqual(result["risk_level"], "Warning")
        self.assertIn("duplicate_ingredient", self.codes(result))
        dup = [f for f in result["findings"] if f["code"] == "duplicate_ingredient"][0]
        self.assertEqual(dup["ingredient"], "Paracetamol")
        self.assertEqual(sorted(dup["medicines"]), ["Napa", "Napa Extra"])

    def test_alias_acetaminophen_is_paracetamol(self):
        result = assess(self.db, [
            {"brand": "Napa", "doses_per_day": 2},
            {"generic": "Acetaminophen", "strength_mg": 500, "doses_per_day": 2},
        ], self.max_doses)
        self.assertIn("duplicate_ingredient", self.codes(result))

    def test_salt_form_is_the_same_drug(self):
        # Bangladesh pack names carry the salt; DDInter and the
        # max-dose table use the base name. Found in the top-50 audit
        # (Nov 15): 282 of 1,256 "missing" pairs are in DDInter once
        # the salt is stripped.
        self.assertEqual(self.db.canonical("Cefuroxime axetil"), "cefuroxime")
        self.assertEqual(self.db.canonical("Metformin hydrochloride"), "metformin")
        self.assertEqual(self.db.canonical("Amoxicillin trihydrate"), "amoxicillin")
        self.assertEqual(self.db.canonical("Ketotifen fumarate (oral)"), "ketotifen")
        self.assertEqual(self.db.canonical("Zinc sulfate monohydrate"), "zinc sulfate")
        self.assertEqual(self.db.canonical("Paracetamol"), "paracetamol")

    def test_salt_form_duplicate_is_caught(self):
        result = assess(self.db, [
            {"generic": "Amoxicillin trihydrate", "strength_mg": 500, "doses_per_day": 2},
            {"generic": "Amoxicillin", "strength_mg": 500, "doses_per_day": 2},
        ], self.max_doses)
        self.assertIn("duplicate_ingredient", self.codes(result))

    def test_salt_form_interaction_is_found(self):
        # Clopidogrel bisulfate + omeprazole (Seclo) is the Major
        # interaction from the demo seed, reached through the salt form.
        result = assess(self.db, [
            {"brand": "Seclo", "doses_per_day": 1},
            {"generic": "Clopidogrel bisulfate", "strength_mg": 75, "doses_per_day": 1},
        ], self.max_doses)
        self.assertEqual(result["risk_level"], "High")
        self.assertIn("interaction", self.codes(result))

    def test_major_interaction_is_high(self):
        result = assess(self.db, [
            {"brand": "Seclo", "doses_per_day": 1},
            {"brand": "Plavix", "doses_per_day": 1},
        ], self.max_doses)
        self.assertEqual(result["risk_level"], "High")
        inter = [f for f in result["findings"] if f["code"] == "interaction"][0]
        self.assertEqual(inter["severity"], "Major")

    def test_unknown_pair_is_never_safe(self):
        """Aspirin + Seclo have no interaction record in the demo DB:
        must be Unknown, not Low."""
        result = assess(self.db, [
            {"brand": "Aspirin", "doses_per_day": 1},
            {"brand": "Seclo", "doses_per_day": 1},
        ], self.max_doses)
        self.assertIn("interaction_unknown", self.codes(result))
        self.assertEqual(result["risk_level"], "Unknown")

    def test_verified_no_interaction_pairs_are_clear(self):
        """Napa Extra + Seclo: the two cross-medicine pairs are curated
        "None" rows (Drugs.com: no interactions found) — checked and
        clear, so no interaction finding at all and risk Low. The third
        pair, Paracetamol + Caffeine inside one Napa Extra tablet, is
        co-formulated by design and is not checked at all."""
        result = assess(self.db, [
            {"brand": "Napa Extra", "doses_per_day": 2},
            {"brand": "Seclo", "doses_per_day": 1},
        ], self.max_doses)
        self.assertNotIn("interaction", self.codes(result))
        self.assertNotIn("interaction_unknown", self.codes(result))
        self.assertEqual(result["coverage"]["pairs_checked"], 2)
        self.assertEqual(result["coverage"]["pairs_with_data"], 2)
        self.assertEqual(result["risk_level"], "Low")

    def test_single_combination_tablet_has_no_interaction_pairs(self):
        """One Augmentin-style tablet (two ingredients, one medicine):
        its own two ingredients are never checked against each other,
        so there is no false Unknown and no interaction finding."""
        result = assess(self.db, [
            {"brand": "Napa Extra", "doses_per_day": 2},
        ], self.max_doses)
        self.assertEqual(result["coverage"]["pairs_checked"], 0)
        self.assertNotIn("interaction", self.codes(result))
        self.assertNotIn("interaction_unknown", self.codes(result))
        self.assertEqual(result["risk_level"], "Low")

    def test_pair_inside_and_across_medicines_is_still_checked(self):
        """Paracetamol + Caffeine occur inside Napa Extra AND across
        medicines (Napa Extra vs a separate paracetamol tablet): the
        cross-medicine occurrence is still looked up (curated None),
        so the pair counts as checked."""
        result = assess(self.db, [
            {"brand": "Napa Extra", "doses_per_day": 2},
            {"generic": "Acetaminophen", "doses_per_day": 1},
        ], self.max_doses)
        self.assertEqual(result["coverage"]["pairs_checked"], 1)
        self.assertEqual(result["coverage"]["pairs_with_data"], 1)
        self.assertNotIn("interaction_unknown", self.codes(result))

    def test_dose_over_max(self):
        result = assess(self.db, [
            {"brand": "Napa", "doses_per_day": 3},
            {"brand": "Napa Extra", "doses_per_day": 3},
            {"brand": "Ace Plus", "doses_per_day": 3},
        ], self.max_doses)
        self.assertEqual(result["risk_level"], "High")
        over = [f for f in result["findings"] if f["code"] == "dose_over_max"]
        self.assertEqual(len(over), 1)
        self.assertEqual(over[0]["ingredient"], "Paracetamol")

    def test_clean_case_is_low(self):
        result = assess(self.db, [{"brand": "Napa", "doses_per_day": 2}], self.max_doses)
        self.assertEqual(result["risk_level"], "Low")
        self.assertEqual(result["findings"], [])

    def test_unresolved_brand_is_flagged_not_guessed(self):
        result = assess(self.db, [
            {"brand": "Napa", "doses_per_day": 2},
            {"brand": "MadeUpBrand", "doses_per_day": 1},
        ], self.max_doses)
        self.assertIn("unresolved_medicine", self.codes(result))
        self.assertEqual(result["medicines_resolved"], 1)

    def test_duplicate_check_is_ingredient_based(self):
        meds = [expand_medicine(self.db, {"brand": "Napa", "doses_per_day": 1}),
                expand_medicine(self.db, {"brand": "Napa Extra", "doses_per_day": 1})]
        dups = check_duplicates(meds)
        self.assertEqual(len(dups), 1)
        self.assertEqual(dups[0]["ingredient"], "Paracetamol")

    def test_interaction_lookup_is_bidirectional(self):
        meds = [expand_medicine(self.db, {"brand": "Plavix", "doses_per_day": 1}),
                expand_medicine(self.db, {"brand": "Seclo", "doses_per_day": 1})]
        findings, cov = check_interactions(self.db, meds)
        self.assertEqual(cov["pairs_with_data"], 1)
        self.assertEqual(findings[0]["severity"], "Major")

    def test_expired_is_high(self):
        result = assess(self.db, [
            {"brand": "Napa Extra", "doses_per_day": 2,
             "expiry": "2020-01", "batch": "B12345"},
        ], self.max_doses, today=date(2026, 11, 14))
        self.assertEqual(result["risk_level"], "High")
        exp = [f for f in result["findings"] if f["code"] == "expired"]
        self.assertEqual(len(exp), 1)
        self.assertIn("B12345", exp[0]["detail"])

    def test_expiring_soon_is_warning(self):
        result = assess(self.db, [
            {"brand": "Napa", "doses_per_day": 2, "expiry": "11/2026"},
        ], self.max_doses, today=date(2026, 11, 14))
        self.assertIn("expiring_soon", self.codes(result))
        self.assertEqual(result["risk_level"], "Warning")

    def test_far_future_expiry_no_finding(self):
        result = assess(self.db, [
            {"brand": "Napa", "doses_per_day": 2, "expiry": "2030-06"},
        ], self.max_doses, today=date(2026, 11, 14))
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["risk_level"], "Low")
        self.assertEqual(result["coverage"]["expiry_checked"], 1)

    def test_no_expiry_supplied_is_not_checked_not_assumed(self):
        result = assess(self.db, [{"brand": "Napa", "doses_per_day": 2}],
                        self.max_doses, today=date(2026, 11, 14))
        self.assertEqual(result["coverage"]["expiry_checked"], 0)
        self.assertEqual(result["coverage"]["expiry_total"], 1)

    def test_unparseable_expiry_flagged_unknown(self):
        result = assess(self.db, [
            {"brand": "Napa", "doses_per_day": 2, "expiry": "whenever"},
        ], self.max_doses, today=date(2026, 11, 14))
        self.assertIn("expiry_unparseable", self.codes(result))

    def test_check_expiry_direct(self):
        meds = [expand_medicine(self.db, {"brand": "Napa", "expiry": "2020-01"}),
                expand_medicine(self.db, {"brand": "Napa Extra", "expiry": "11/2026"}),
                expand_medicine(self.db, {"brand": "Seclo"})]
        findings, cov = check_expiry(meds, today=date(2026, 11, 14))
        self.assertEqual([f["code"] for f in findings],
                         ["expired", "expiring_soon"])
        self.assertEqual(cov["expiry_checked"], 2)
        self.assertEqual(cov["expiry_total"], 3)

    def test_dose_load_sums_across_medicines(self):
        meds = [expand_medicine(self.db, {"brand": "Napa", "doses_per_day": 2}),
                expand_medicine(self.db, {"brand": "Napa Extra", "doses_per_day": 2})]
        findings, cov = check_dose_load(meds, self.max_doses)
        # 500*2 + 500*2 = 2000 mg = 50% of max: no dose finding.
        self.assertEqual([f["code"] for f in findings], [])
        meds = [expand_medicine(self.db, {"brand": "Napa Extra", "doses_per_day": 3})]
        findings, _ = check_dose_load(meds, self.max_doses)
        # Caffeine 65*3 = 195 mg (49%): still no finding.
        self.assertEqual(findings, [])


class TestExplicitIngredients(unittest.TestCase):
    """Phase 4 (§5): a medicine carrying an explicit `ingredients`
    list (a source-verified web draft, used for one assessment only)
    is expanded directly, with no DB lookup."""

    def setUp(self):
        self.db = DrugDB(_demo_db())
        self.max_doses = load_max_doses()

    def codes(self, result):
        return [f["code"] for f in result["findings"]]

    def test_explicit_ingredients_used_directly(self):
        out = expand_medicine(self.db, {
            "brand": "Enzoflam",
            "ingredients": ["Diclofenac", "Paracetamol", "Serratiopeptidase"],
            "strength_parts": ["50 mg", "325 mg", "15 mg"],
            "doses_per_day": 2,
        })
        self.assertFalse(out["unresolved"])
        self.assertEqual([i["ingredient"] for i in out["ingredients"]],
                         ["diclofenac", "paracetamol", "serratiopeptidase"])
        self.assertEqual([i["mg_per_dose"] for i in out["ingredients"]],
                         [50.0, 325.0, 15.0])

    def test_explicit_ingredients_beat_the_database(self):
        # Napa IS in the DB (paracetamol only); an explicit list wins.
        out = expand_medicine(self.db, {
            "brand": "Napa",
            "ingredients": ["Paracetamol", "Caffeine"],
            "strength_parts": ["500 mg", "65 mg"],
        })
        self.assertEqual([i["ingredient"] for i in out["ingredients"]],
                         ["paracetamol", "caffeine"])

    def test_missing_or_unparseable_strength_is_none(self):
        out = expand_medicine(self.db, {
            "brand": "Mystery",
            "ingredients": ["Diclofenac", "Paracetamol", "Serratiopeptidase"],
            "strength_parts": ["50 mg", "N/A"],
            "doses_per_day": 2,
        })
        self.assertEqual([i["mg_per_dose"] for i in out["ingredients"]],
                         [50.0, None, None])

    def test_empty_ingredients_list_falls_back_to_database(self):
        out = expand_medicine(self.db, {"brand": "Napa", "ingredients": []})
        self.assertEqual([i["ingredient"] for i in out["ingredients"]],
                         ["paracetamol"])

    def test_explicit_ingredients_drive_assessment(self):
        # Enzoflam (web draft) + Napa: paracetamol is duplicated across
        # the two, and the dose load totals it — the engine checks a
        # web-sourced composition exactly like a database one.
        result = assess(self.db, [
            {"brand": "Enzoflam",
             "ingredients": ["Diclofenac", "Paracetamol", "Serratiopeptidase"],
             "strength_parts": ["50 mg", "325 mg", "15 mg"],
             "doses_per_day": 2},
            {"brand": "Napa", "doses_per_day": 2},
        ], self.max_doses)
        self.assertIn("duplicate_ingredient", self.codes(result))
        self.assertEqual(result["medicines_resolved"], 2)
        # 325*2 + 500*2 = 1650 mg paracetamol/day: under the max, so no
        # dose finding — the mg came from strength_parts, not a guess.
        self.assertNotIn("dose_over_max", self.codes(result))

    def test_explicit_ingredients_with_no_mg_skip_dose_math(self):
        result = assess(self.db, [
            {"brand": "Mystery",
             "ingredients": ["Paracetamol"],
             "strength_parts": [],
             "doses_per_day": 3},
        ], self.max_doses)
        self.assertEqual(result["medicines_resolved"], 1)
        self.assertNotIn("dose_over_max", self.codes(result))
        self.assertNotIn("dose_near_max", self.codes(result))


class TestCuratedNoInteractionFile(unittest.TestCase):
    def test_none_severity_survives_loader(self):
        """The curated file must load as severity "None", never collapsed
        to Unknown — otherwise the pairs read as ungraded again."""
        from load_ddinter import _norm_severity, read_pairs

        self.assertEqual(_norm_severity("None"), "None")
        self.assertEqual(_norm_severity("no interactions found"), "None")
        pairs, _ = read_pairs(
            Path(__file__).resolve().parent / "data" / "curated_no_interaction.csv")
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(p["severity"] == "None" for p in pairs))
        self.assertEqual(
            {(p["drug_a"], p["drug_b"]) for p in pairs},
            {("caffeine", "omeprazole"), ("caffeine", "paracetamol"),
             ("omeprazole", "paracetamol")})
        self.assertTrue(all(p["source"].startswith("https://www.drugs.com/")
                          for p in pairs))



class TestWebVerifyHook(unittest.TestCase):
    """The engine stays deterministic by default; an optional hook
    (web_verify.web_verify_pair) answers only the pairs the database
    cannot — missing or ungraded Unknown. A failed check never breaks
    the assessment."""

    def setUp(self):
        self.db = DrugDB(_demo_db())
        self.max_doses = load_max_doses()
        self.meds = [
            {"brand": "Napa Extra", "doses_per_day": 2},
            {"brand": "Seclo", "doses_per_day": 1},
            {"brand": "Aspirin", "doses_per_day": 1},
        ]

    def _unknown_pairs(self, result):
        return sorted((f["ingredient_a"], f["ingredient_b"])
                      for f in result["findings"]
                      if f["severity"] == "Unknown"
                      and "ingredient_a" in f)

    def test_no_hook_leaves_unknown_pairs(self):
        # The prescription Asif ran: aspirin + paracetamol is in DDInter
        # but ungraded, aspirin + caffeine/omeprazole are missing here.
        result = assess(self.db, self.meds, self.max_doses)
        self.assertEqual(result["risk_level"], "Unknown")
        self.assertEqual(len(self._unknown_pairs(result)), 3)

    def test_hook_clearing_all_unknown_pairs_gives_low(self):
        def clear(a, b):
            return {"drug_a": a, "drug_b": b, "severity": "None",
                    "mechanism": "No interactions found.",
                    "management": "No action.",
                    "source": "https://www.drugs.com/drug-interactions/a-with-b.html",
                    "verified_via": "web"}
        result = assess(self.db, self.meds, self.max_doses, web_verify=clear)
        self.assertEqual(self._unknown_pairs(result), [])
        self.assertEqual(result["risk_level"], "Low")
        # 5, not 6: Paracetamol + Caffeine inside one Napa Extra tablet
        # is co-formulated and not checked (see test_verified_no_...).
        self.assertEqual(result["coverage"]["pairs_with_data"], 5)

    def test_hook_graded_verdict_is_reported_and_labelled(self):
        def moderate(a, b):
            if {a, b} == {"aspirin", "paracetamol"}:
                return {"drug_a": a, "drug_b": b, "severity": "Moderate",
                        "mechanism": "Increases bleeding risk.",
                        "management": "Monitor.",
                        "source": "https://www.drugs.com/drug-interactions/a-with-b.html",
                        "verified_via": "web"}
            return None
        result = assess(self.db, self.meds, self.max_doses, web_verify=moderate)
        hit = [f for f in result["findings"]
               if f.get("ingredient_a") == "Aspirin"
               and f.get("ingredient_b") == "Paracetamol"][0]
        self.assertEqual(hit["severity"], "Moderate")
        self.assertEqual(hit["verified_via"], "web")
        self.assertIn("drugs.com", hit["source"])
        self.assertEqual(result["risk_level"], "Warning")

    def test_hook_not_called_for_graded_pairs(self):
        calls = []

        def hook(a, b):
            calls.append((a, b))
            return None
        assess(self.db, [
            {"brand": "Seclo", "doses_per_day": 1},
            {"brand": "Plavix", "doses_per_day": 1},
        ], self.max_doses, web_verify=hook)
        self.assertEqual(calls, [])  # clopidogrel + omeprazole is Major in the DB

    def test_hook_failure_keeps_unknown(self):
        def boom(a, b):
            raise RuntimeError("provider down")
        result = assess(self.db, self.meds, self.max_doses, web_verify=boom)
        self.assertEqual(result["risk_level"], "Unknown")
        self.assertEqual(len(self._unknown_pairs(result)), 3)

    def test_hook_system_exit_keeps_unknown(self):
        def gone(a, b):
            raise SystemExit("No API key.")
        result = assess(self.db, self.meds, self.max_doses, web_verify=gone)
        self.assertEqual(result["risk_level"], "Unknown")

    def test_hook_returning_none_keeps_unknown(self):
        result = assess(self.db, self.meds, self.max_doses,
                        web_verify=lambda a, b: None)
        self.assertEqual(result["risk_level"], "Unknown")


class TestDescribeMedicine(unittest.TestCase):
    """The --check-brand diagnostic: what the DB knows about a name."""

    def setUp(self):
        self.db = DrugDB(_demo_db())

    def test_exact_brand(self):
        found = describe_medicine(self.db, "Napa Extra")
        self.assertTrue(found["found"])
        self.assertEqual(found["kind"], "brand")
        self.assertEqual(found["ingredients"], ["Paracetamol", "Caffeine"])

    def test_space_tolerant_match(self):
        # 'napaextra' (no space) still finds the 'Napa Extra' brand.
        found = describe_medicine(self.db, "napaextra")
        self.assertTrue(found["found"])
        self.assertEqual(found["name"], "Napa Extra")

    def test_missing_brand_is_a_gap_not_a_guess(self):
        found = describe_medicine(self.db, "Zentel")
        self.assertFalse(found["found"])
        self.assertEqual(found["ingredients"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
