#!/usr/bin/env python3
"""RxSentinel — Phase 4: FastAPI backend (port 8000).

Endpoints (see PHASE4_SPEC.md §2):
    GET  /api/health
    POST /api/scan      — multipart image; returns {prescription, report, provenance}
    GET  /api/admin/medicine-requests?status=pending|approved|rejected
    POST /api/admin/medicine-requests/{id}/approve
    POST /api/admin/medicine-requests/{id}/reject
    POST /api/report    — {prescription, report} -> PDF bytes
    POST /api/chat      — {messages, context} -> {answer, sources}

Scan flow (§3): the upload is saved under backend/uploads/, run
through intake.run_intake, and for each needs_confirmation brand a
web lookup (lookup_medicine.lookup_brand + verify_source) is tried.
A composition that passes BOTH is used for THIS assessment only
(safety_engine.expand_medicine's explicit-ingredients path), labelled
provenance "web-unverified" with its source URL, and upserted into
the `medicine_requests` review queue (one pending request per brand,
case-insensitive). Anything unverifiable stays in needs_confirmation
— never invented. Database-resolved medicines get provenance
"database".

Admin endpoints need header X-Admin-Password matching ADMIN_PASSWORD
from .env. This is demo-grade protection for a course project, not
production auth: a single shared password, no per-user accounts, no
rate limiting. It fails closed (401) when ADMIN_PASSWORD is unset.

Errors are always {"detail": "..."} with a proper HTTP status, never
a stack trace. No scan persistence: /api/report and /api/chat take
back exactly what /api/scan returned.

Run from backend/ (Windows PowerShell — see PHASE4_RUN.md):
    uvicorn api:app --port 8000
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from add_medicine import add_entry, validate_entry
from doc_chat import answer_question
from intake import run_intake
from lookup_medicine import lookup_brand, verify_source
from report_pdf import build_report_pdf
from safety_engine import DrugDB, assess, load_max_doses
from web_verify import _load_dotenv, web_verify_pair

DISCLAIMER = ("For information only — not medical advice. "
              "Ask your pharmacist or doctor before changing any medicine.")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
REQUESTS_STATUSES = ("pending", "approved", "rejected")

app = FastAPI(title="RxSentinel API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ DB deps
def get_db():
    """MongoDB database handle (MONGO_URI / MONGO_DB from .env).
    Overridden by tests via app.dependency_overrides."""
    _load_dotenv()
    uri = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017")
    name = os.environ.get("MONGO_DB", "rxsentinel")
    try:
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client[name]
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail=f"MongoDB unavailable: {exc}")


def get_drugdb(db=Depends(get_db)) -> DrugDB:
    return DrugDB(db)


# ------------------------------------------------------------------ admin auth
def _admin_password() -> str:
    _load_dotenv()
    return os.environ.get("ADMIN_PASSWORD", "").strip()


def _require_admin(x_admin_password: str | None = Header(default=None)):
    """Demo-grade admin check: fail closed with 401 when the password
    is unset in .env or the header does not match it."""
    expected = _admin_password()
    if not expected or not x_admin_password or x_admin_password != expected:
        raise HTTPException(status_code=401,
                            detail="Admin password required.")
    return True


# ------------------------------------------------------------------ helpers
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_request(doc: dict) -> dict:
    out = {
        "id": str(doc.get("_id", "")),
        "brand": doc.get("brand", ""),
        "generic_title": doc.get("generic_title", ""),
        "ingredients": list(doc.get("ingredients") or []),
        "strength_parts": list(doc.get("strength_parts") or []),
        "dosage_form": doc.get("dosage_form", ""),
        "company": doc.get("company", ""),
        "source": doc.get("source", ""),
        "source_verified": bool(doc.get("source_verified", False)),
        "scan_count": int(doc.get("scan_count", 0) or 0),
        "first_seen": doc.get("first_seen", ""),
        "last_seen": doc.get("last_seen", ""),
        "status": doc.get("status", "pending"),
    }
    if doc.get("reason"):
        out["reason"] = doc["reason"]
    return out


def _find_request(coll, request_id: str):
    """Find one medicine_requests doc by id — works for ObjectId ids
    (real MongoDB) and plain-string ids (test stubs)."""
    doc = coll.find_one({"_id": request_id})
    if doc is not None:
        return doc
    try:
        from bson import ObjectId
        oid = ObjectId(request_id)
    except Exception:
        return None
    return coll.find_one({"_id": oid})


def _upsert_medicine_request(db, draft: dict, source_verified: bool) -> None:
    """One pending request per brand (case-insensitive brand_key).
    A repeat scan of a pending brand bumps scan_count/last_seen and
    refreshes the composition/source; otherwise a fresh pending doc
    is inserted (a later scan may re-open a fresh request after a
    rejection — a rejection is about that data, not a ban)."""
    coll = getattr(db, "medicine_requests", None)
    if coll is None:
        return
    brand = str(draft.get("brand") or "").strip()
    brand_key = brand.lower()
    now = _now_iso()
    existing = coll.find_one({"brand_key": brand_key})
    if existing is not None and existing.get("status") == "pending":
        coll.update_one(
            {"brand_key": brand_key},
            {"$set": {
                "brand": brand,
                "generic_title": draft.get("generic_title", ""),
                "ingredients": list(draft.get("ingredients") or []),
                "strength_parts": list(draft.get("strength_parts") or []),
                "dosage_form": draft.get("dosage_form", ""),
                "company": draft.get("company", ""),
                "source": draft.get("source", ""),
                "source_verified": bool(source_verified),
                "scan_count": int(existing.get("scan_count", 0) or 0) + 1,
                "last_seen": now,
            }})
        return
    doc = {
        "brand": brand,
        "brand_key": brand_key,
        "generic_title": draft.get("generic_title", ""),
        "ingredients": list(draft.get("ingredients") or []),
        "strength_parts": list(draft.get("strength_parts") or []),
        "dosage_form": draft.get("dosage_form", ""),
        "company": draft.get("company", ""),
        "source": draft.get("source", ""),
        "source_verified": bool(source_verified),
        "verified_via": "verify_source",
        "scan_count": 1,
        "first_seen": now,
        "last_seen": now,
        "status": "pending",
    }
    if hasattr(coll, "insert_one"):
        coll.insert_one(doc)
    else:  # pragma: no cover - pymongo always has insert_one
        coll.update_one({"brand_key": brand_key}, {"$set": doc}, upsert=True)


def _resolve_unconfirmed(prescription: dict, db) -> None:
    """Try the web lookup for each needs_confirmation item. A brand
    whose composition passes lookup + source-page check moves into
    medicines[] with provenance "web-unverified" for this assessment
    only, and upserts its review-queue record. Anything else stays
    in needs_confirmation — never invented."""
    remaining = []
    for item in prescription.get("needs_confirmation") or []:
        name = (item.get("brand") or item.get("name") or "").strip()
        if not name:
            remaining.append(item)
            continue
        try:
            draft = lookup_brand(name)
        except Exception:
            draft = None
        if not draft:
            remaining.append(item)
            continue
        try:
            verified = bool(verify_source(draft.get("source", ""),
                                          draft.get("ingredients") or []))
        except Exception:
            verified = False
        if not verified:
            remaining.append(item)
            continue
        med: dict = {
            "brand": str(draft.get("brand") or name),
            "generic": draft.get("generic_title", ""),
            "ingredients": list(draft.get("ingredients") or []),
            "strength_parts": list(draft.get("strength_parts") or []),
            "provenance": "web-unverified",
            "source": draft.get("source", ""),
        }
        if item.get("doses_per_day") is not None:
            med["doses_per_day"] = item["doses_per_day"]
        for key in ("expiry", "batch"):
            if item.get(key):
                med[key] = item[key]
        prescription.setdefault("medicines", []).append(med)
        try:
            _upsert_medicine_request(db, draft, verified)
        except Exception as exc:  # never break a scan over the queue
            print(f"  medicine_requests upsert failed for {name}: {exc}",
                  file=sys.stderr)
    prescription["needs_confirmation"] = remaining


# ------------------------------------------------------------------ models
class RejectRequest(BaseModel):
    reason: str = ""


class ReportRequest(BaseModel):
    prescription: dict
    report: dict | None = None


class ChatRequest(BaseModel):
    messages: list[dict] = []
    context: dict = {}


# ------------------------------------------------------------------ endpoints
@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/scan")
async def scan(image: UploadFile = File(...),
               list_: str = Form("prescription", alias="list"),
               handwritten: str = Form("false"),
               drugdb: DrugDB = Depends(get_drugdb),
               db=Depends(get_db)):
    if list_ not in ("prescription", "dispensed"):
        raise HTTPException(status_code=400,
                            detail="list must be 'prescription' or 'dispensed'.")
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(image.filename or "").suffix or ".jpg"
        saved = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        saved.write_bytes(await image.read())
        mode = "auto"
        # The `handwritten` field is accepted for backward compatibility
        # but ignored: auto mode already decides (PaddleOCR first, and
        # the vision LLM when mean line confidence is below 0.7), so
        # the caller never has to guess whether a photo is printed
        # or handwritten. See PHASE4_UI_SPEC §1.
        prescription = run_intake(str(saved), mode=mode, drugdb=drugdb)
    except HTTPException:
        raise
    except (Exception, SystemExit) as exc:
        raise HTTPException(status_code=500,
                            detail=f"Scan failed: {exc}")
    if not isinstance(prescription, dict):
        raise HTTPException(status_code=500,
                            detail="Scan failed: intake returned no prescription.")
    prescription["list"] = list_
    for med in prescription.get("medicines") or []:
        med.setdefault("provenance", "database")
    _resolve_unconfirmed(prescription, db)
    medicines = list(prescription.get("medicines") or [])
    report = None
    if medicines:
        try:
            report = assess(drugdb, medicines, load_max_doses(),
                            web_verify=partial(web_verify_pair, drugdb))
        except (Exception, SystemExit) as exc:
            raise HTTPException(status_code=500,
                                detail=f"Assessment failed: {exc}")
    provenance = {}
    for med in medicines:
        label = med.get("brand") or med.get("generic") or med.get("name") or "?"
        provenance[str(label)] = med.get("provenance", "database")
    return {"prescription": prescription, "report": report,
            "provenance": provenance}


@app.get("/api/admin/medicine-requests")
def list_requests(status: str = Query("pending"),
                  db=Depends(get_db),
                  _admin: bool = Depends(_require_admin)):
    if status not in REQUESTS_STATUSES:
        raise HTTPException(status_code=400,
                            detail="status must be pending, approved or rejected.")
    coll = getattr(db, "medicine_requests", None)
    docs = list(coll.find({"status": status})) if coll is not None else []
    docs.sort(key=lambda d: int(d.get("scan_count", 0) or 0), reverse=True)
    return {"requests": [_serialize_request(d) for d in docs]}


@app.post("/api/admin/medicine-requests/{request_id}/approve")
def approve_request(request_id: str,
                    drugdb: DrugDB = Depends(get_drugdb),
                    db=Depends(get_db),
                    _admin: bool = Depends(_require_admin)):
    coll = getattr(db, "medicine_requests", None)
    doc = _find_request(coll, request_id) if coll is not None else None
    if doc is None:
        raise HTTPException(status_code=404,
                            detail=f"No medicine request with id {request_id}.")
    entry = validate_entry({
        "brand": doc.get("brand"),
        "generic_title": doc.get("generic_title"),
        "ingredients": doc.get("ingredients"),
        "strength_parts": doc.get("strength_parts"),
        "dosage_form": doc.get("dosage_form"),
        "company": doc.get("company"),
        "source": doc.get("source"),
    })
    try:
        add_entry(drugdb, entry, apply=True, added_via="admin review queue")
    except (Exception, SystemExit) as exc:
        raise HTTPException(status_code=500,
                            detail=f"Could not add {entry['brand']}: {exc}")
    coll.update_one({"_id": doc["_id"]},
                    {"$set": {"status": "approved",
                              "approved_at": _now_iso()}})
    return {"ok": True, "brand": entry["brand"]}


@app.post("/api/admin/medicine-requests/{request_id}/reject")
def reject_request(request_id: str, body: RejectRequest,
                   db=Depends(get_db),
                   _admin: bool = Depends(_require_admin)):
    coll = getattr(db, "medicine_requests", None)
    doc = _find_request(coll, request_id) if coll is not None else None
    if doc is None:
        raise HTTPException(status_code=404,
                            detail=f"No medicine request with id {request_id}.")
    coll.update_one({"_id": doc["_id"]},
                    {"$set": {"status": "rejected",
                              "reason": body.reason or "",
                              "rejected_at": _now_iso()}})
    return {"ok": True}


@app.post("/api/report")
def report_pdf(body: ReportRequest):
    try:
        pdf_bytes = build_report_pdf(body.prescription, body.report)
    except (Exception, SystemExit) as exc:
        raise HTTPException(status_code=400,
                            detail=f"Could not build the report PDF: {exc}")
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition":
                             'attachment; filename="rxsentinel-report.pdf"'})


@app.post("/api/chat")
def chat(body: ChatRequest, drugdb: DrugDB = Depends(get_drugdb)):
    try:
        return answer_question(body.messages, body.context, drugdb=drugdb)
    except (RuntimeError, SystemExit) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail=f"Chat unavailable: {exc}")
