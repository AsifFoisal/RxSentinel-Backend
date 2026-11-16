#!/usr/bin/env python3
"""RxSentinel — Phase 4: the safety report PDF (fpdf2).

Pure function: build_report_pdf(prescription, report) -> bytes.
No scan persistence, no database — the frontend sends back exactly
what POST /api/scan returned.

Content: header (RxSentinel, date, "Medicine Safety Report"), risk
level, medicines table (web-unverified rows labelled with their
source), findings (severity, title, detail, management, source),
dose summary, coverage, and the standard disclaimer in every page
footer. Never calls a Low result "safe"; a web-sourced composition
is labelled as not yet verified in the database, never presented
as database fact.
"""
from __future__ import annotations

from datetime import date

from fpdf import FPDF
from fpdf.enums import XPos, YPos

DISCLAIMER = ("For information only — not medical advice. "
              "Ask your pharmacist or doctor before changing any medicine.")

WEB_UNVERIFIED_LABEL = "web search — not yet verified in our database"


def _safe(text) -> str:
    """Core PDF fonts are Latin-1 only; drop anything outside it
    rather than crashing on a Bangla or accented character.
    Dashes are translated first so they render as "-" instead of "?"."""
    if text is None:
        return ""
    text = str(text).replace("—", "-").replace("–", "-")
    return text.encode("latin-1", "replace").decode("latin-1")


def _cell(pdf, text, size=11, bold=False):
    pdf.set_font("Helvetica", "B" if bold else "", size)
    pdf.cell(0, 10 if size >= 14 else 8, _safe(text),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _mc(pdf, text, size=11, bold=False, italic=False):
    """multi_cell that returns the cursor to the left margin on the
    next line — fpdf2's default leaves it at the right margin, so a
    second multi_cell(0, ...) would have no width to render in."""
    style = "B" if bold else ("I" if italic else "")
    pdf.set_font("Helvetica", style, size)
    pdf.multi_cell(0, 6, _safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _finding_lines(finding: dict) -> list[str]:
    lines = [f"[{finding.get('severity', '')}] "
             f"{finding.get('title', '')}"]
    for key in ("detail", "management", "source"):
        value = finding.get(key)
        if value:
            lines.append(f"{key.capitalize()}: {value}")
    if finding.get("verified_via"):
        lines.append(f"Verified via: {finding['verified_via']}")
    return lines


class _ReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 5, _safe(DISCLAIMER), align="C",
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)


def build_report_pdf(prescription: dict, report: dict | None) -> bytes:
    """Render {prescription, report} as a PDF. `report` may be None
    (nothing was confirmed to assess) — the PDF then says so instead
    of inventing findings."""
    prescription = prescription or {}
    report = report or {}
    pdf = _ReportPDF()
    pdf.set_compression(False)  # keep text searchable in the bytes
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    _cell(pdf, "RxSentinel", size=18, bold=True)
    _cell(pdf, "Medicine Safety Report", size=12)
    _cell(pdf, f"Date: {date.today().isoformat()}", size=12)
    pdf.ln(4)

    risk = report.get("risk_level", "Unknown") if report else "Unknown"
    _cell(pdf, f"Risk level: {risk}", size=14, bold=True)
    if not report:
        _mc(pdf, "No medicines were confirmed, so no safety assessment "
                 "could be run. Items needing confirmation are listed below.")
    pdf.ln(2)

    medicines = prescription.get("medicines") or []
    _cell(pdf, "Medicines", size=13, bold=True)
    if not medicines:
        _mc(pdf, "No confirmed medicines.")
    for med in medicines:
        label = (med.get("brand") or med.get("generic")
                 or med.get("name") or "?")
        parts = [str(label)]
        if med.get("generic") and med.get("brand"):
            parts.append(f"generic: {med['generic']}")
        if med.get("doses_per_day") is not None:
            parts.append(f"doses/day: {med['doses_per_day']}")
        if med.get("strength_mg") is not None:
            parts.append(f"strength: {med['strength_mg']} mg")
        elif med.get("strength_parts"):
            parts.append("strength: " + ", ".join(
                str(p) for p in med["strength_parts"]))
        _mc(pdf, " - ".join(parts))
        if med.get("provenance") == "web-unverified":
            line = WEB_UNVERIFIED_LABEL
            if med.get("source"):
                line += f" — source: {med['source']}"
            _mc(pdf, line, size=10, italic=True)
    confirm = prescription.get("needs_confirmation") or []
    if confirm:
        pdf.ln(2)
        _cell(pdf, "Needs confirmation", size=13, bold=True)
        for item in confirm:
            name = item.get("brand") or item.get("name") or "?"
            reason = item.get("reason", "not confirmed")
            _mc(pdf, f"{name} — {reason}")
    pdf.ln(2)

    findings = report.get("findings") or []
    _cell(pdf, "Findings", size=13, bold=True)
    if not findings:
        _mc(pdf, "No findings.")
    for finding in findings:
        for line in _finding_lines(finding):
            _mc(pdf, line)
        pdf.ln(2)

    coverage = report.get("coverage") or {}
    _cell(pdf, "Dose summary and coverage", size=13, bold=True)
    dose_lines = []
    if coverage.get("ingredients_totalled") is not None:
        dose_lines.append(
            f"Ingredients totalled for dose: {coverage['ingredients_totalled']}")
    if coverage.get("ingredients_with_max") is not None:
        dose_lines.append(
            f"Ingredients with a daily maximum on file: "
            f"{coverage['ingredients_with_max']}")
    if coverage.get("pairs_checked") is not None:
        dose_lines.append(
            f"Interaction pairs checked: {coverage['pairs_checked']} "
            f"({coverage.get('pairs_with_data', 0)} with data)")
    if coverage.get("expiry_checked") is not None:
        dose_lines.append(
            f"Expiry checked: {coverage['expiry_checked']} of "
            f"{coverage.get('expiry_total', 0)} medicines")
    for f in findings:
        if f.get("code", "").startswith("dose_"):
            dose_lines.append(str(f.get("title", "")))
    if not dose_lines:
        dose_lines.append("No dose or coverage information available.")
    for line in dose_lines:
        _mc(pdf, line)
    return bytes(pdf.output())
