#!/usr/bin/env python3
"""RxSentinel — Phase 3 intake tests. Offline, no PaddleOCR, no LLM
key, no MongoDB needed (stub readers + the demo DB).

Run:  python test_intake.py
"""
import json
import os
import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from intake import (  # noqa: E402
    GenericVisionReader, _connect_db, _open_mongo_drugdb, _strip_dose_tokens,
    assess_intake, candidate_from_line, candidate_from_llm,
    ground_name, merge_continuation_lines, parse_expiry_batch,
    parse_frequency, parse_llm_json, pick_vision_reader, run_intake,
    to_prescription_json,
)
from safety_engine import (  # noqa: E402
    DrugDB, _demo_db, assess, expand_medicine, load_max_doses)


class StubPaddleReader:
    def __init__(self, lines):
        self.lines = lines

    def read(self, image_path):
        return self.lines


class StubVisionReader:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def extract(self, image_path):
        self.calls += 1
        return self.text


class TestFrequencyParsing(unittest.TestCase):
    def test_plus_notation(self):
        self.assertEqual(parse_frequency("1+0+1"), 2)
        self.assertEqual(parse_frequency("1+1+1"), 3)
        self.assertEqual(parse_frequency("1-0-0"), 1)
        self.assertEqual(parse_frequency("Napa Extra 500mg - 1+0+1"), 2)

    def test_words_and_codes(self):
        self.assertEqual(parse_frequency("twice daily"), 2)
        self.assertEqual(parse_frequency("three times a day"), 3)
        self.assertEqual(parse_frequency("2 times daily"), 2)
        self.assertEqual(parse_frequency("BD"), 2)
        self.assertEqual(parse_frequency("TDS"), 3)
        self.assertEqual(parse_frequency("once daily"), 1)
        self.assertEqual(parse_frequency("1 tablet at night"), 1)
        self.assertEqual(parse_frequency("every 8 hours"), 3)
        self.assertEqual(parse_frequency("every 12 hours"), 2)

    def test_unreadable_is_none(self):
        self.assertIsNone(parse_frequency(""))
        self.assertIsNone(parse_frequency("as directed"))
        self.assertIsNone(parse_frequency("Napa Extra"))


class TestExpiryBatchParsing(unittest.TestCase):
    def test_exp_and_batch(self):
        out = parse_expiry_batch("Exp: 03/2027   Batch: B12345")
        self.assertEqual(out, {"expiry": "03/2027", "batch": "B12345"})

    def test_no_expiry(self):
        self.assertEqual(parse_expiry_batch("Napa Extra 500mg"), {})

    def test_bare_date_without_keyword(self):
        # A pack prints just the date, no "Exp:" in front.
        self.assertEqual(parse_expiry_batch("2027-03"), {"expiry": "2027-03"})
        self.assertEqual(parse_expiry_batch("03/2027"), {"expiry": "03/2027"})

    def test_phone_number_is_never_an_expiry(self):
        self.assertEqual(parse_expiry_batch("Call 01712-345678"), {})
        self.assertEqual(parse_expiry_batch("01712-345678"), {})


class TestLineCandidates(unittest.TestCase):
    def setUp(self):
        self.db = DrugDB(_demo_db())

    def test_printed_line_resolves_brand(self):
        c = candidate_from_line("Napa Extra 500mg - 1+0+1", 0.95, self.db, "paddle")
        self.assertEqual(c["name"], "Napa Extra")
        self.assertEqual(c["resolve_kind"], "brand")
        self.assertEqual(c["doses_per_day"], 2)
        self.assertEqual(c["strength_mg"], 500.0)
        self.assertTrue(c["resolved"])
        self.assertGreaterEqual(c["confidence"], 0.9)

    def test_unknown_brand_goes_to_confirm(self):
        c = candidate_from_line("Zentel 400mg - 1+0+0", 0.95, self.db, "paddle")
        self.assertFalse(c["resolved"])
        self.assertLess(c["confidence"], 0.6)

    def test_header_line_skipped(self):
        self.assertIsNone(candidate_from_line("Prescription", 0.99, self.db, "paddle"))

    def test_expiry_only_line_has_no_candidate(self):
        self.assertIsNone(candidate_from_line("Exp: 03/2027 Batch: B12345",
                                              0.99, self.db, "paddle"))

    def test_manufacturer_name_skipped(self):
        db = DrugDB(_demo_db())
        db.db.brands.docs[0]["company"] = "Beximco Pharmaceuticals Ltd."
        self.assertIsNone(candidate_from_line(
            "Beximco Pharmaceuticals Ltd.", 0.99, db, "paddle"))

    def test_manufacturer_limited_variant_skipped(self):
        # The pack prints 'Ltd' while the dataset may store 'Limited'.
        db = DrugDB(_demo_db())
        db.db.brands.docs[0]["company"] = "Beximco Pharmaceuticals Limited"
        self.assertIsNone(candidate_from_line(
            "Beximco Pharmaceuticals Ltd.", 0.99, db, "paddle"))

    def test_em_dash_note_cut_from_name(self):
        # The em dash was destroyed by the character filter before the
        # separator split could see it, leaving "After meal" in the
        # name (the ASCII-hyphen case worked, so it looked inconsistent).
        self.assertEqual(
            _strip_dose_tokens("1. Napa Extra 500mg 1+0+1 \u2014 After meal"),
            "Napa Extra")
        self.assertEqual(
            _strip_dose_tokens("2. Seclo 20mg 1+0+0 \u2014 Before breakfast"),
            "Seclo")
        self.assertEqual(
            _strip_dose_tokens("3. Aspirin 75mg 0+0+1 - At night"),
            "Aspirin")

    def test_list_number_and_form_prefix_stripped(self):
        self.assertEqual(_strip_dose_tokens("Tab. Napa Extra 500mg"),
                         "Napa Extra")
        self.assertEqual(_strip_dose_tokens("Paracetamol 500"),
                         "Paracetamol")

    def test_address_line_is_not_a_medicine(self):
        self.assertIsNone(candidate_from_line(
            "12 Green Road, Dhaka \u2022 Phone: 01XXXXXXXXX",
            0.99, self.db, "paddle"))

    def test_wrapped_prescription_line_resolves(self):
        c = candidate_from_line("1. Napa Extra 500mg 1+0+1 \u2014 After meal",
                                0.99, self.db, "paddle")
        self.assertEqual(c["name"], "Napa Extra")
        self.assertEqual(c["resolve_kind"], "brand")
        self.assertEqual(c["doses_per_day"], 2)

    def test_multi_tablet_dose_is_summed_not_slot_counted(self):
        # Regression guard against the agent's version: 2+0+1 is 3
        # tablets a day, not 2 dosing occasions — the engine multiplies
        # mg-per-tablet by this number, so slot-counting understates it.
        c = candidate_from_line("Napa 500mg - 2+0+1", 0.95, self.db, "paddle")
        self.assertEqual(c["doses_per_day"], 3)


class TestPrescriptionJson(unittest.TestCase):
    def setUp(self):
        self.db = DrugDB(_demo_db())

    def test_confident_and_confirm_split(self):
        cands = [
            candidate_from_line("Napa Extra 500mg - 1+0+1", 0.95, self.db, "paddle"),
            candidate_from_line("Zentel 400mg - 1+0+0", 0.95, self.db, "paddle"),
        ]
        out = to_prescription_json(cands, "paddle")
        self.assertEqual(len(out["medicines"]), 1)
        self.assertEqual(out["medicines"][0]["brand"], "Napa Extra")
        self.assertEqual(out["medicines"][0]["doses_per_day"], 2)
        self.assertEqual(len(out["needs_confirmation"]), 1)
        self.assertIn("Zentel", out["needs_confirmation"][0]["brand"])

    def test_doc_level_expiry_attaches_to_medicines(self):
        cands = [candidate_from_line("Seclo 20mg - 1+0+0", 0.95, self.db, "paddle")]
        out = to_prescription_json(cands, "paddle", {"expiry": "03/2027", "batch": "B12345"})
        self.assertEqual(out["medicines"][0]["expiry"], "03/2027")
        self.assertEqual(out["medicines"][0]["batch"], "B12345")

    def test_engine_runs_on_intake_output(self):
        cands = [
            candidate_from_line("Napa Extra 500mg - 1+0+1", 0.95, self.db, "paddle"),
            candidate_from_line("Seclo 20mg - 1+0+0", 0.95, self.db, "paddle"),
        ]
        out = to_prescription_json(cands, "paddle")
        result = assess(self.db, out["medicines"], load_max_doses())
        self.assertEqual(result["medicines_checked"], 2)
        self.assertEqual(result["medicines_resolved"], 2)
        self.assertEqual(result["risk_level"], "Low")

    def test_pack_composition_line_dropped_when_brand_present(self):
        # A pack prints its composition under the brand; against a
        # real DB that line resolves as the generic "Paracetamol +
        # Caffeine" — the same medicine twice, which would fake a
        # duplicate-ingredient finding. Only Napa Extra must survive.
        self.db.db.generics.docs.append(
            {"title": "Paracetamol + Caffeine",
             "ingredients": ["Paracetamol", "Caffeine"],
             "strength_parts": ["500 mg", "65 mg"]})
        cands = [
            candidate_from_line("Napa Extra", 0.99, self.db, "paddle"),
            candidate_from_line("Paracetamol 500mg + Caffeine 65mg",
                                0.99, self.db, "paddle"),
        ]
        out = to_prescription_json(cands, "paddle")
        self.assertEqual(len(out["medicines"]), 1)
        self.assertEqual(out["medicines"][0]["brand"], "Napa Extra")
        self.assertEqual(out["needs_confirmation"], [])

    def test_printed_prescription_reads_three_medicines(self):
        # The City Care Clinic sample: wrapped frequency lines, list
        # numbers, an address line, headers — the full run must yield
        # 3 confirmed medicines and an empty confirm list.
        lines = [
            {"text": "City Care Clinic", "confidence": 0.99},
            {"text": "12 Green Road, Dhaka \u2022 Phone: 01XXXXXXXXX",
             "confidence": 0.99},
            {"text": "Patient: Test Patient", "confidence": 0.99},
            {"text": "Age: 30 \u2022 Date: 15 Nov 2026", "confidence": 0.99},
            {"text": "Rx", "confidence": 0.99},
            {"text": "1. Napa Extra 500mg", "confidence": 0.99},
            {"text": "1+0+1 \u2014 After meal", "confidence": 0.99},
            {"text": "2. Seclo 20mg", "confidence": 0.99},
            {"text": "1+0+0 \u2014 Before breakfast", "confidence": 0.99},
            {"text": "3. Aspirin 75mg", "confidence": 0.99},
            {"text": "0+0+1 - At night", "confidence": 0.99},
            {"text": "Advice: Drink plenty of water. Come back after 7 days.",
             "confidence": 0.99},
            {"text": "Signature", "confidence": 0.99},
        ]
        out = run_intake("data/sample_prescription_printed.png",
                         mode="paddle", drugdb=self.db,
                         paddle_reader=StubPaddleReader(lines))
        self.assertEqual(
            [(m.get("brand"), m.get("doses_per_day")) for m in out["medicines"]],
            [("Napa Extra", 2), ("Seclo", 1), ("Aspirin", 1)])
        self.assertEqual(out["needs_confirmation"], [])
        result = assess(self.db, out["medicines"], load_max_doses())
        self.assertEqual(result["medicines_checked"], 3)
        self.assertEqual(result["medicines_resolved"], 3)

    def test_combo_generic_kept_when_prescribed_alone(self):
        # The same generic line with no brand in the image is a real
        # prescribed medicine, not a composition restatement.
        self.db.db.generics.docs.append(
            {"title": "Paracetamol + Caffeine",
             "ingredients": ["Paracetamol", "Caffeine"],
             "strength_parts": ["500 mg", "65 mg"]})
        cands = [candidate_from_line("Paracetamol 500mg + Caffeine 65mg",
                                     0.99, self.db, "paddle")]
        out = to_prescription_json(cands, "paddle")
        self.assertEqual(len(out["medicines"]), 1)
        self.assertEqual(out["medicines"][0]["generic"], "Paracetamol + Caffeine")

    def test_assess_intake_web_hook_clears_unknown(self):
        result = {"medicines": [
            {"brand": "Napa Extra", "doses_per_day": 2},
            {"brand": "Seclo", "doses_per_day": 1},
            {"brand": "Aspirin", "doses_per_day": 1},
        ]}

        def clear(a, b):
            return {"drug_a": a, "drug_b": b, "severity": "None",
                    "mechanism": "No interactions found.",
                    "management": "No action.",
                    "source": "https://www.drugs.com/drug-interactions/a-with-b.html",
                    "verified_via": "web"}

        report = assess_intake(result, self.db, web_verify=clear)
        self.assertEqual(report["risk_level"], "Low")
        # 5, not 6: Paracetamol + Caffeine inside one Napa Extra tablet
        # is co-formulated and not checked.
        self.assertEqual(report["coverage"]["pairs_with_data"], 5)

    def test_assess_intake_offline_keeps_unknown(self):
        result = {"medicines": [
            {"brand": "Napa Extra", "doses_per_day": 2},
            {"brand": "Seclo", "doses_per_day": 1},
            {"brand": "Aspirin", "doses_per_day": 1},
        ]}
        report = assess_intake(result, self.db)
        self.assertEqual(report["risk_level"], "Unknown")

    def test_single_generic_beside_brand_is_not_dropped(self):
        # A separately prescribed single generic is a real second
        # medicine (and a real duplicate the engine should flag) —
        # only combination generics restating a brand are dropped.
        self.db.db.generics.docs.append(
            {"title": "Paracetamol", "ingredients": ["Paracetamol"],
             "strength_parts": ["500 mg"]})
        cands = [
            candidate_from_line("Napa Extra", 0.99, self.db, "paddle"),
            candidate_from_line("Paracetamol 500mg", 0.99, self.db, "paddle"),
        ]
        out = to_prescription_json(cands, "paddle")
        self.assertEqual(len(out["medicines"]), 2)


class TestVisionRoute(unittest.TestCase):
    def setUp(self):
        self.db = DrugDB(_demo_db())

    def test_llm_json_parses_and_grounds(self):
        items = parse_llm_json(json.dumps({
            "medicines": [
                {"name": "Seclo", "strength": "20mg", "frequency": "1+0+0",
                 "confidence": "high"},
                {"name": "MadeUpDrug", "strength": "10mg", "frequency": "twice daily",
                 "confidence": "high"},
            ]}))
        cands = [candidate_from_llm(i, self.db) for i in items]
        out = to_prescription_json(cands, "llm")
        self.assertEqual(out["medicines"][0]["brand"], "Seclo")
        self.assertEqual(out["medicines"][0]["doses_per_day"], 1)
        # The LLM claimed high confidence, but the DB can't resolve it.
        self.assertEqual(len(out["needs_confirmation"]), 1)

    def test_llm_reply_without_json_rejected(self):
        with self.assertRaises(ValueError):
            parse_llm_json("I can see two medicines on this photo.")


class TestPaddleOCRReader(unittest.TestCase):
    def _reader_with(self, ocr):
        from intake import PaddleOCRReader
        reader = PaddleOCRReader.__new__(PaddleOCRReader)
        reader._ocr = ocr
        return reader

    def test_paddleocr3_list_of_dicts(self):
        # PaddleOCR 3.x predict() returns a LIST of dicts. The old
        # reader iterated a dict page as a list and dropped every line.
        class FakeOCR3:
            def predict(self, path):
                return [{"rec_texts": ["Napa Extra 500mg", "1+0+1"],
                         "rec_scores": [0.95, 0.93]}]
        lines = self._reader_with(FakeOCR3()).read("x.png")
        self.assertEqual([l["text"] for l in lines],
                         ["Napa Extra 500mg", "1+0+1"])
        self.assertEqual(lines[0]["confidence"], 0.95)

    def test_paddleocr2_list_of_pages(self):
        class FakeOCR2:
            def ocr(self, path):
                return [[[None, ("Seclo 20mg", 0.94)]]]
        lines = self._reader_with(FakeOCR2()).read("x.png")
        self.assertEqual(lines, [{"text": "Seclo 20mg", "confidence": 0.94}])


class TestVisionWiring(unittest.TestCase):
    """The handwriting route has its OWN provider wiring (VISION_*),
    separate from the web-check LLM (LLM_*): a vision model must see
    images, a text-only web-check model cannot."""

    _EMPTY = {"VISION_BASE_URL": "", "VISION_MODEL": "", "VISION_API_KEY": "",
              "LLM_BASE_URL": "", "LLM_MODEL": "", "LLM_API_KEY": "",
              "GEMINI_API_KEY": ""}

    def test_vision_env_beats_shared_llm_env(self):
        env = dict(self._EMPTY, VISION_BASE_URL="https://vision.example/v1",
                   VISION_MODEL="vision-model", VISION_API_KEY="vision-key",
                   LLM_BASE_URL="https://llm.example/v1",
                   LLM_MODEL="llm-model", LLM_API_KEY="llm-key")
        with patch.dict(os.environ, env):
            reader = pick_vision_reader()
        self.assertIsInstance(reader, GenericVisionReader)
        self.assertEqual(reader.base_url,
                         "https://vision.example/v1/chat/completions")
        self.assertEqual(reader.model, "vision-model")
        self.assertEqual(reader.api_key, "vision-key")

    def test_shared_llm_env_used_when_no_vision_env(self):
        env = dict(self._EMPTY, LLM_BASE_URL="https://llm.example/v1",
                   LLM_MODEL="llm-model", LLM_API_KEY="llm-key")
        with patch.dict(os.environ, env):
            reader = pick_vision_reader()
        self.assertIsInstance(reader, GenericVisionReader)
        self.assertEqual(reader.model, "llm-model")
        self.assertEqual(reader.base_url,
                         "https://llm.example/v1/chat/completions")

    def test_vision_model_only_falls_back_per_field(self):
        env = dict(self._EMPTY, VISION_MODEL="vision-model",
                   LLM_BASE_URL="https://llm.example/v1",
                   LLM_API_KEY="llm-key")
        with patch.dict(os.environ, env):
            reader = pick_vision_reader()
        self.assertEqual(reader.model, "vision-model")
        self.assertEqual(reader.base_url,
                         "https://llm.example/v1/chat/completions")
        self.assertEqual(reader.api_key, "llm-key")

    def test_flags_beat_env(self):
        env = dict(self._EMPTY, VISION_BASE_URL="https://vision.example/v1",
                   VISION_MODEL="vision-model", VISION_API_KEY="vision-key")
        with patch.dict(os.environ, env):
            reader = pick_vision_reader(
                vision_base_url="https://flags.example/v1",
                vision_model="flag-model", vision_api_key="flag-key")
        self.assertEqual(reader.base_url,
                         "https://flags.example/v1/chat/completions")
        self.assertEqual(reader.model, "flag-model")

    def test_nothing_configured_fails_with_a_clear_message(self):
        with patch.dict(os.environ, self._EMPTY):
            with self.assertRaises(SystemExit) as ctx:
                pick_vision_reader()
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_run_intake_threads_vision_config(self):
        import intake as intake_mod

        seen = {}

        class RecReader:
            def extract(self, path):
                return json.dumps({"medicines": []})

        def fake_pick(*args, **kwargs):
            seen.update(kwargs)
            return RecReader()

        with patch.object(intake_mod, "pick_vision_reader",
                          side_effect=fake_pick):
            run_intake("data/sample_prescription_printed.png", mode="llm",
                       drugdb=None,
                       vision_base_url="https://vision.example/v1",
                       vision_model="vision-model", vision_api_key="k")
        self.assertEqual(seen["vision_base_url"], "https://vision.example/v1")
        self.assertEqual(seen["vision_model"], "vision-model")
        self.assertEqual(seen["vision_api_key"], "k")


class _FakeColl:
    """Minimal Mongo-like collection: exact match + $regex (re.search)."""

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
                elif d.get(k) != v:
                    ok = False
            if ok:
                return d
        return None

    def update_one(self, query, update, upsert=False):
        doc = self.find_one(query)
        if doc is None and not upsert:
            return
        if doc is None:
            doc = {k: v for k, v in query.items()
                   if not isinstance(v, dict)}
            self.docs.append(doc)
        for k, v in update.get("$set", {}).items():
            doc[k] = v
        for k, v in update.get("$setOnInsert", {}).items():
            doc.setdefault(k, v)


def _handwritten_db(with_augmentin_brand=True):
    """Fake DB for the four handwritten-prescription brands. The
    Augmentin brand can be left out to exercise the generic fallback
    (brand not in the DB, generic Amoxicillin + Clavulanic acid is)."""
    brands = [
        {"brand_name": "Enzoflam",
         "generic_title": "Diclofenac + Paracetamol + Serratiopeptidase",
         "ingredients": ["Diclofenac", "Paracetamol", "Serratiopeptidase"],
         "strength_parts": ["50 mg", "325 mg", "15 mg"]},
        {"brand_name": "Pan D",
         "generic_title": "Pantoprazole + Domperidone",
         "ingredients": ["Pantoprazole", "Domperidone"],
         "strength_parts": ["40 mg", "30 mg"]},
        {"brand_name": "Hexigel",
         "generic_title": "Chlorhexidine gluconate",
         "ingredients": ["Chlorhexidine gluconate"],
         "strength_parts": ["1% w/w"]},
        {"brand_name": "Napa", "generic_title": "Paracetamol",
         "ingredients": ["Paracetamol"], "strength_parts": ["500 mg"]},
    ]
    if with_augmentin_brand:
        brands.append(
            {"brand_name": "Augmentin",
             "generic_title": "Amoxicillin + Clavulanic acid",
             "ingredients": ["Amoxicillin", "Clavulanic acid"],
             "strength_parts": ["500 mg", "125 mg"]})
    db = types.SimpleNamespace()
    db.brands = _FakeColl(brands)
    db.generics = _FakeColl([
        {"title": "Amoxicillin + Clavulanic acid",
         "ingredients": ["Amoxicillin", "Clavulanic acid"],
         "strength_parts": ["500 mg", "125 mg"]},
        {"title": "Pantoprazole + Domperidone",
         "ingredients": ["Pantoprazole", "Domperidone"],
         "strength_parts": ["40 mg", "30 mg"]},
        {"title": "Chlorhexidine gluconate",
         "ingredients": ["Chlorhexidine gluconate"],
         "strength_parts": ["1% w/w"]},
    ])
    # One seeded pair so the end-to-end run can show an interaction
    # finding (test seed, not a medical claim).
    db.interactions = _FakeColl([
        {"drug_a": "amoxicillin", "drug_b": "diclofenac",
         "severity": "Minor", "mechanism": "Test seed.",
         "management": "Test seed.", "source": "test seed"},
    ])
    db.aliases = _FakeColl([])
    return DrugDB(db)


class TestHandwrittenGrounding(unittest.TestCase):
    """Why the first handwritten run put all four medicines in the
    confirm list: 'PanD' vs the database's 'Pan D', and 'Hexigel gum
    paint' vs the brand 'Hexigel'. Grounding now bridges both, and a
    brand missing from the DB resolves through its stated generic."""

    def test_pand_matches_pan_d(self):
        kind, matched, _, _ = ground_name("PanD", _handwritten_db())
        self.assertEqual((kind, matched), ("brand", "Pan D"))

    def test_hexigel_gum_paint_matches_hexigel(self):
        kind, matched, _, _ = ground_name("Hexigel gum paint",
                                          _handwritten_db())
        self.assertEqual((kind, matched), ("brand", "Hexigel"))

    def test_exact_brand_still_resolves_first(self):
        kind, matched, conf, _ = ground_name("Augmentin", _handwritten_db())
        self.assertEqual((kind, matched, conf), ("brand", "Augmentin", 1.0))

    def test_llm_generic_fallback_when_brand_missing(self):
        db = _handwritten_db(with_augmentin_brand=False)
        cand = candidate_from_llm(
            {"name": "Augmentin", "frequency": "twice daily",
             "generic": "Amoxicillin + Clavulanic acid",
             "confidence": "high"}, db)
        self.assertTrue(cand["resolved"])
        self.assertEqual(cand["resolve_kind"], "generic")
        self.assertEqual(cand["name"], "Amoxicillin + Clavulanic acid")
        self.assertEqual(cand["brand_label"], "Augmentin")
        self.assertEqual(cand["ingredients"],
                         ["amoxicillin", "clavulanic acid"])
        out = to_prescription_json([cand], "llm")
        self.assertEqual(len(out["medicines"]), 1)
        self.assertEqual(out["medicines"][0]["brand"], "Augmentin")
        self.assertEqual(out["medicines"][0]["generic"],
                         "Amoxicillin + Clavulanic acid")

    def test_unresolved_item_keeps_stated_generic_as_hint(self):
        # The model stated a composition the DB cannot verify: it rides
        # on the confirm item as a hint for the human verifying a
        # lookup_medicine draft — never passed to the engine as fact.
        cand = candidate_from_llm(
            {"name": "MysteryBrand", "frequency": "twice daily",
             "generic": "Drug X + Drug Y", "confidence": "high"},
            _handwritten_db())
        out = to_prescription_json([cand], "llm")
        self.assertEqual(out["medicines"], [])
        self.assertEqual(len(out["needs_confirmation"]), 1)
        self.assertEqual(out["needs_confirmation"][0]["stated_generic"],
                         "Drug X + Drug Y")

    def test_llm_generic_fallback_rejects_brand_match(self):
        # A "generic" that matches a BRAND name is ambiguous — it stays
        # unresolved instead of being silently re-labelled.
        cand = candidate_from_llm(
            {"name": "UnknownBrand", "frequency": "twice daily",
             "generic": "Napa", "confidence": "high"}, _handwritten_db())
        self.assertFalse(cand["resolved"])
        self.assertEqual(cand["confidence"], 0.4)

    def test_brand_medicine_carries_composition_and_strengths(self):
        # The UI cards showed "—" for Enzoflam, Pan D and Hexigel:
        # to_prescription_json only emitted brand + doses for a
        # brand-resolved medicine, never the composition or strengths
        # the candidate already carried from the matched doc.
        cand = candidate_from_llm(
            {"name": "Enzoflam", "frequency": "twice daily",
             "confidence": "high"}, _handwritten_db())
        out = to_prescription_json([cand], "llm")
        med = out["medicines"][0]
        self.assertEqual(med["generic"],
                         "Diclofenac + Paracetamol + Serratiopeptidase")
        self.assertEqual(med["strength_parts"],
                         ["50 mg", "325 mg", "15 mg"])
        self.assertEqual(med["doses_per_day"], 2)
        # The `ingredients` list itself must NOT ride on the medicine:
        # it would make expand_medicine take its explicit-ingredients
        # path (meant for web drafts) instead of the database lookup.
        self.assertNotIn("ingredients", med)

    def test_printed_brand_line_carries_composition_and_strengths(self):
        cand = candidate_from_line("Enzoflam - 1+0+1", 0.95,
                                   _handwritten_db(), "paddle")
        out = to_prescription_json([cand], "paddle")
        med = out["medicines"][0]
        self.assertEqual(med["generic"],
                         "Diclofenac + Paracetamol + Serratiopeptidase")
        self.assertEqual(med["strength_parts"],
                         ["50 mg", "325 mg", "15 mg"])

    def test_display_fields_do_not_change_engine_maths(self):
        # The new generic/strength_parts fields are display-only: the
        # engine must still resolve through the database, with
        # per-ingredient mg from the doc's strength parts.
        cand = candidate_from_llm(
            {"name": "Enzoflam", "frequency": "twice daily",
             "confidence": "high"}, _handwritten_db())
        out = to_prescription_json([cand], "llm")
        expanded = expand_medicine(_handwritten_db(), out["medicines"][0])
        self.assertEqual(
            [(i["ingredient"], i["mg_per_dose"])
             for i in expanded["ingredients"]],
            [("diclofenac", 50.0), ("paracetamol", 325.0),
             ("serratiopeptidase", 15.0)])

    def test_auto_add_missing_brand_on_verified_source(self):
        # Enzoflam is NOT in this DB and the vision item states no
        # generic, so only the auto-add path can resolve it: lookup
        # finds it, the source page confirms the ingredients, it is
        # added, and the same run assesses it as a confirmed medicine.
        db = _handwritten_db()
        db.db.brands.docs = [d for d in db.db.brands.docs
                             if d["brand_name"] != "Enzoflam"]
        draft = {
            "brand": "Enzoflam",
            "generic_title": "Diclofenac + Paracetamol + Serratiopeptidase",
            "ingredients": ["Diclofenac", "Paracetamol", "Serratiopeptidase"],
            "strength_parts": ["50 mg", "325 mg", "15 mg"],
            "company": "Alkem Laboratories Ltd",
            "source": "https://www.practo.com/medicine-info/enzoflam-tablet-53621",
        }

        def fake_lookup(brand, api_key=None, base_url=None, model=None,
                        client=None):
            return draft if brand == "Enzoflam" else None

        vision_json = json.dumps({"medicines": [
            {"name": "Enzoflam", "strength": "50mg",
             "frequency": "twice daily", "confidence": "high"},
        ]})
        out = run_intake("data/sample_prescription_printed.png", mode="llm",
                         drugdb=db, vision_reader=StubVisionReader(vision_json),
                         auto_add_missing=True, medicine_lookup=fake_lookup,
                         source_verifier=lambda url, ings: True)
        self.assertEqual(len(out["medicines"]), 1)
        self.assertEqual(out["medicines"][0]["brand"], "Enzoflam")
        self.assertEqual(out["needs_confirmation"], [])
        self.assertIsNotNone(db.resolve_brand("Enzoflam"))

    def test_auto_add_refused_when_source_page_does_not_confirm(self):
        # The lookup found a draft but the cited page does not list
        # the ingredients: nothing is written, the item stays
        # unconfirmed, and the draft is saved for manual review.
        import tempfile

        db = _handwritten_db()
        db.db.brands.docs = [d for d in db.db.brands.docs
                             if d["brand_name"] != "Enzoflam"]
        draft = {
            "brand": "Enzoflam",
            "generic_title": "Diclofenac + Paracetamol + Serratiopeptidase",
            "ingredients": ["Diclofenac", "Paracetamol", "Serratiopeptidase"],
            "strength_parts": ["50 mg", "325 mg", "15 mg"],
            "company": "Alkem Laboratories Ltd",
            "source": "https://www.practo.com/medicine-info/enzoflam-tablet-53621",
        }
        vision_json = json.dumps({"medicines": [
            {"name": "Enzoflam", "strength": "50mg",
             "frequency": "twice daily", "confidence": "high"},
        ]})
        with tempfile.TemporaryDirectory() as d:
            suggest = Path(d) / "suggested.json"
            out = run_intake(
                "data/sample_prescription_printed.png", mode="llm",
                drugdb=db, vision_reader=StubVisionReader(vision_json),
                auto_add_missing=True,
                medicine_lookup=lambda brand, **kw: draft,
                source_verifier=lambda url, ings: False,
                suggest_path=suggest)
            self.assertEqual(out["medicines"], [])
            self.assertEqual(len(out["needs_confirmation"]), 1)
            self.assertIsNone(db.resolve_brand("Enzoflam"))
            rows = json.loads(suggest.read_text(encoding="utf-8"))
            self.assertEqual([r["brand"] for r in rows], ["Enzoflam"])

    def test_handwritten_run_now_produces_a_full_report(self):
        vision_json = json.dumps({"medicines": [
            {"name": "Augmentin", "strength": "625mg",
             "frequency": "twice daily",
             "generic": "Amoxicillin + Clavulanic acid",
             "confidence": "high"},
            {"name": "Enzoflam", "strength": "50mg",
             "frequency": "twice daily", "confidence": "high"},
            {"name": "PanD", "frequency": "once daily",
             "confidence": "high"},
            {"name": "Hexigel gum paint", "frequency": "twice daily",
             "confidence": "high"},
        ]})
        db = _handwritten_db()
        out = run_intake("data/sample_prescription_printed.png", mode="llm",
                         drugdb=db, vision_reader=StubVisionReader(vision_json))
        self.assertEqual(len(out["medicines"]), 4)
        self.assertEqual(out["needs_confirmation"], [])
        report = assess_intake(out, db)
        self.assertIsNotNone(report)
        self.assertEqual(report["medicines_checked"], 4)
        self.assertEqual(report["medicines_resolved"], 4)
        minor = [f for f in report["findings"] if f["severity"] == "Minor"]
        self.assertEqual(
            {(f["ingredient_a"], f["ingredient_b"]) for f in minor},
            {("Amoxicillin", "Diclofenac")})


class TestRunIntake(unittest.TestCase):
    def setUp(self):
        self.db = DrugDB(_demo_db())
        self.image = str(Path(__file__).resolve().parent
                         / "data" / "sample_prescription.png")

    def test_wrapped_medicine_lines_merge(self):
        lines = [
            {"text": "Napa Extra 500mg", "confidence": 0.95},
            {"text": "1+0+1 - After meal", "confidence": 0.93},
            {"text": "Seclo 20mg - 1+0+0", "confidence": 0.94},
        ]
        merged = merge_continuation_lines(lines, self.db)
        self.assertEqual(len(merged), 2)  # frequency line joined the one above
        self.assertIn("1+0+1", merged[0]["text"])
        self.assertIn("Seclo", merged[1]["text"])

    def test_paddle_route_with_stub(self):
        lines = [
            {"text": "Prescription", "confidence": 0.99},
            {"text": "Napa Extra 500mg - 1+0+1", "confidence": 0.95},
            {"text": "Seclo 20mg - 1+0+0", "confidence": 0.93},
            {"text": "Exp: 03/2027   Batch: B12345", "confidence": 0.98},
        ]
        out = run_intake(self.image, mode="paddle", drugdb=self.db,
                         paddle_reader=StubPaddleReader(lines))
        self.assertEqual(out["source"], "paddle")
        self.assertEqual(len(out["medicines"]), 2)
        self.assertEqual(out["medicines"][0]["expiry"], "03/2027")
        self.assertEqual(out["needs_confirmation"], [])

    def test_low_confidence_paddle_falls_back_to_vision(self):
        vision = StubVisionReader(json.dumps({
            "medicines": [{"name": "Seclo", "strength": "20mg",
                           "frequency": "1+0+0", "confidence": "high"}]}))
        lines = [{"text": "Seclo 20mg - 1+0+0", "confidence": 0.3}]
        out = run_intake(self.image, mode="auto", drugdb=self.db,
                         paddle_reader=StubPaddleReader(lines),
                         vision_reader=vision)
        self.assertEqual(out["source"], "llm")
        self.assertEqual(vision.calls, 1)
        self.assertEqual(out["medicines"][0]["brand"], "Seclo")

    def test_wrapped_layout_like_the_sample_image(self):
        # Same layout as data/sample_prescription_printed.png: name on
        # one line, frequency on the next.
        lines = [
            {"text": "City Care Clinic", "confidence": 0.99},
            {"text": "Patient: Test Patient", "confidence": 0.99},
            {"text": "Napa Extra 500mg", "confidence": 0.95},
            {"text": "1+0+1 - After meal", "confidence": 0.93},
            {"text": "Seclo 20mg", "confidence": 0.94},
            {"text": "1+0+0 - Before breakfast", "confidence": 0.92},
            {"text": "Aspirin 75mg", "confidence": 0.94},
            {"text": "0+0+1 - At night", "confidence": 0.93},
        ]
        out = run_intake(self.image, mode="paddle", drugdb=self.db,
                         paddle_reader=StubPaddleReader(lines))
        self.assertEqual(len(out["medicines"]), 3)
        self.assertEqual(out["medicines"][0]["doses_per_day"], 2)
        self.assertEqual(out["medicines"][2]["brand"], "Aspirin")
        self.assertEqual(out["medicines"][2]["doses_per_day"], 1)
        self.assertEqual(out["needs_confirmation"], [])

    def test_threshold_param_moves_items_to_confirm(self):
        lines = [{"text": "Seclo 20mg - 1+0+0", "confidence": 0.95}]
        out = run_intake(self.image, mode="paddle", drugdb=self.db,
                         paddle_reader=StubPaddleReader(lines), threshold=0.99)
        self.assertEqual(out["medicines"], [])
        self.assertEqual(len(out["needs_confirmation"]), 1)

    def test_assess_intake_runs_engine_on_confirmed_only(self):
        lines = [{"text": "Napa Extra 500mg - 1+0+1", "confidence": 0.95}]
        out = run_intake(self.image, mode="paddle", drugdb=self.db,
                         paddle_reader=StubPaddleReader(lines))
        report = assess_intake(out, self.db)
        self.assertEqual(report["risk_level"], "Low")
        self.assertEqual(report["medicines_checked"], 1)

    def test_assess_intake_returns_none_when_nothing_confirmed(self):
        self.assertIsNone(assess_intake({"medicines": []}, self.db))

    def test_missing_image_fails_loudly(self):
        with self.assertRaises(SystemExit):
            run_intake("no-such-image.png", mode="paddle", drugdb=self.db,
                       paddle_reader=StubPaddleReader([]))


class TestMongoConnection(unittest.TestCase):
    """The MongoDB path: URI/db precedence and the startup check that
    proves the connected database actually holds the medicine data.
    pymongo is stubbed — no server needed."""

    def setUp(self):
        self._fake = types.ModuleType("pymongo")

        class Coll:
            def __init__(self, count):
                self._count = count

            def count_documents(self, query):
                if isinstance(self._count, Exception):
                    raise self._count
                return self._count

        class DB:
            def __init__(self, name, counts):
                self.name = name
                self.brands = Coll(counts[0])
                self.generics = Coll(counts[1])

        class Client:
            last = None

            def __init__(self, uri, **kw):
                self.uri = uri
                self.kw = kw
                Client.last = self

            def __getitem__(self, name):
                return DB(name, Client.counts)

        Client.counts = (25679, 1988)
        self._fake.MongoClient = Client
        self._client = Client
        sys.modules["pymongo"] = self._fake
        self.addCleanup(sys.modules.pop, "pymongo", None)

    def test_flag_uri_beats_env(self):
        with patch.dict(os.environ, {"MONGO_URI": "mongodb://env:27017",
                                     "MONGO_DB": "envdb"}):
            db = _connect_db("mongodb://flag:27017", "flagdb")
            self.assertEqual(db.name, "flagdb")
            self.assertEqual(self._client.last.uri, "mongodb://flag:27017")

    def test_env_beats_local_default(self):
        with patch.dict(os.environ, {"MONGO_URI": "mongodb://env:27017",
                                     "MONGO_DB": "envdb"}):
            db = _connect_db(None, None)
            self.assertEqual(db.name, "envdb")
            self.assertEqual(self._client.last.uri, "mongodb://env:27017")

    def test_open_reports_counts(self):
        with patch.dict(os.environ, {"MONGO_URI": "mongodb://env:27017",
                                     "MONGO_DB": "rxsentinel"}):
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                drugdb = _open_mongo_drugdb(None, None)
            self.assertIn("25679 brands", buf.getvalue())
            self.assertIn("1988 generics", buf.getvalue())
            self.assertIsNotNone(drugdb)

    def test_open_refuses_empty_database(self):
        self._client.counts = (0, 0)
        with patch.dict(os.environ, {"MONGO_URI": "mongodb://env:27017",
                                     "MONGO_DB": "emptydb"}):
            with self.assertRaises(SystemExit) as ctx:
                _open_mongo_drugdb(None, None)
            self.assertIn("0 brands", str(ctx.exception))

    def test_open_refuses_unreachable_server(self):
        self._client.counts = (RuntimeError("timed out"), 0)
        with patch.dict(os.environ, {"MONGO_URI": "mongodb://env:27017",
                                     "MONGO_DB": "rxsentinel"}):
            with self.assertRaises(SystemExit):
                _open_mongo_drugdb(None, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
