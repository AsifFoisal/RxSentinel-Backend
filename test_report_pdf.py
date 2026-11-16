#!/usr/bin/env python3
"""RxSentinel — Phase 4 report-PDF tests. Offline, needs fpdf2 only.

Run:  python test_report_pdf.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_pdf import build_report_pdf  # noqa: E402

# The sample handwritten scan shape (Nov 16 live run): 4 medicines,
# risk Warning driven by a Moderate finding, Unknown pairs remaining.
PRESCRIPTION = {
    "medicines": [
        {"brand": "Augmentin", "generic": "Amoxicillin + Clavulanic Acid",
         "ingredients": ["Amoxicillin", "Clavulanic Acid"],
         "strength_parts": ["500 mg", "125 mg"], "doses_per_day": 3,
         "provenance": "database"},
        {"brand": "Enzoflam",
         "generic": "Diclofenac + Paracetamol + Serratiopeptidase",
         "ingredients": ["Diclofenac", "Paracetamol", "Serratiopeptidase"],
         "strength_parts": ["50 mg", "325 mg", "15 mg"], "doses_per_day": 2,
         "provenance": "web-unverified",
         "source": "https://www.practo.com/medicine-info/enzoflam-tablet-53621"},
        {"brand": "Pan D",
         "generic": "Pantoprazole + Domperidone",
         "ingredients": ["Pantoprazole", "Domperidone"],
         "strength_parts": ["40 mg", "30 mg"], "doses_per_day": 1,
         "provenance": "web-unverified",
         "source": "https://www.practo.com/medicine-info/pan-d-tablet-12345"},
        {"brand": "Hexigel", "generic": "Chlorhexidine gluconate",
         "ingredients": ["Chlorhexidine gluconate"],
         "strength_parts": ["1% w/w"], "doses_per_day": 2,
         "provenance": "web-unverified",
         "source": "https://www.1mg.com/drugs/hexigel-gum-paint-12345"},
    ],
    "needs_confirmation": [],
    "source": "llm",
    "disclaimer": ("For information only — not medical advice. "
                   "Ask your pharmacist or doctor before changing any medicine."),
}

REPORT = {
    "risk_level": "Warning",
    "medicines_checked": 4,
    "medicines_resolved": 4,
    "findings": [
        {"code": "interaction", "severity": "Moderate",
         "ingredient_a": "Diclofenac", "ingredient_b": "Domperidone",
         "title": "Moderate interaction: Diclofenac + Domperidone",
         "detail": "Combined use may increase the risk of stomach bleeding.",
         "management": "Monitor for stomach pain or black stools.",
         "source": "https://www.drugs.com/drug-interactions/diclofenac-with-domperidone.html",
         "verified_via": "web"},
        {"code": "interaction_unknown", "severity": "Unknown",
         "ingredient_a": "Amoxicillin", "ingredient_b": "Domperidone",
         "title": "No interaction data: Amoxicillin + Domperidone",
         "detail": "No interaction record was found for this pair. That means NOT CHECKED — not safe.",
         "action": "Ask your pharmacist or doctor about this combination."},
    ],
    "counts": {"Moderate": 1, "Unknown": 1},
    "coverage": {"pairs_checked": 23, "pairs_with_data": 21,
                 "ingredients_with_max": 1, "ingredients_totalled": 6,
                 "expiry_checked": 0, "expiry_total": 4},
    "disclaimer": ("For information only — not medical advice. "
                   "Ask your pharmacist or doctor before changing any medicine."),
}


class TestBuildReportPdf(unittest.TestCase):
    def test_valid_pdf_bytes(self):
        pdf = build_report_pdf(PRESCRIPTION, REPORT)
        self.assertIsInstance(pdf, bytes)
        self.assertTrue(pdf.startswith(b"%PDF"))

    def test_contains_risk_level_and_finding_title(self):
        pdf = build_report_pdf(PRESCRIPTION, REPORT)
        self.assertIn(b"Warning", pdf)
        self.assertIn(b"Moderate interaction: Diclofenac + Domperidone", pdf)

    def test_web_unverified_label_and_source(self):
        pdf = build_report_pdf(PRESCRIPTION, REPORT)
        self.assertIn(b"not yet verified in our database", pdf)
        self.assertIn(b"practo.com/medicine-info/enzoflam-tablet-53621", pdf)

    def test_disclaimer_present(self):
        pdf = build_report_pdf(PRESCRIPTION, REPORT)
        self.assertIn(b"not medical advice", pdf)

    def test_no_report_still_builds(self):
        pdf = build_report_pdf(PRESCRIPTION, None)
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertIn(b"No medicines were confirmed", pdf)

    def test_never_calls_low_safe(self):
        import re
        low = {**REPORT, "risk_level": "Low", "findings": []}
        pdf = build_report_pdf(PRESCRIPTION, low)
        # "safety" contains "safe" as a substring — match the word only.
        self.assertIsNone(re.search(rb"\bsafe\b", pdf.lower()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
