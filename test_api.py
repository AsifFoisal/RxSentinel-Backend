#!/usr/bin/env python3
"""RxSentinel — Phase 4 API tests. FastAPI TestClient + stub DB
(no MongoDB, no OCR, no network). The intake reader, the web
lookup and the web-verify hook are stubbed at the api module level;
the DB is injected through app.dependency_overrides.

Run:  python test_api.py
"""
import os
import re
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["ADMIN_PASSWORD"] = "test-admin"

import api  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from safety_engine import DrugDB, _demo_db  # noqa: E402

ENZOFLAM_DRAFT = {
    "brand": "Enzoflam",
    "generic_title": "Diclofenac + Paracetamol + Serratiopeptidase",
    "ingredients": ["Diclofenac", "Paracetamol", "Serratiopeptidase"],
    "strength_parts": ["50 mg", "325 mg", "15 mg"],
    "dosage_form": "Tablet",
    "company": "Alkem Laboratories Ltd",
    "source": "https://www.practo.com/medicine-info/enzoflam-tablet-53621",
}

STUB_PRESCRIPTION = {
    "medicines": [
        {"brand": "Napa", "doses_per_day": 2, "source": "paddle",
         "confidence": 0.9},
    ],
    "needs_confirmation": [
        {"brand": "Enzoflam", "doses_per_day": 2, "source": "llm",
         "confidence": 0.4, "reason": "not in the medicine database"},
    ],
    "source": "paddle",
    "confidence_threshold": 0.6,
    "disclaimer": ("For information only — not medical advice. "
                   "Ask your pharmacist or doctor before changing any medicine."),
}


class FakeColl:
    """Minimal Mongo-like collection for the stub DB."""

    def __init__(self, docs=None):
        self.docs = list(docs or [])

    def _matches(self, doc, query):
        for k, v in (query or {}).items():
            if isinstance(v, dict) and "$regex" in v:
                if not re.search(v["$regex"], str(doc.get(k, "")), re.I):
                    return False
            elif isinstance(v, dict) and "$in" in v:
                if not any(x in (doc.get(k) or []) for x in v["$in"]):
                    return False
            elif doc.get(k) != v:
                return False
        return True

    def find(self, query=None):
        return [d for d in self.docs if self._matches(d, query)]

    def find_one(self, query):
        for d in self.docs:
            if self._matches(d, query):
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

    def insert_one(self, doc):
        doc = dict(doc)
        if "_id" not in doc:
            doc["_id"] = str(len(self.docs) + 1)
        self.docs.append(doc)

        class _Result:
            inserted_id = doc["_id"]

        return _Result()

    def delete_one(self, query):
        for d in list(self.docs):
            if self._matches(d, query):
                self.docs.remove(d)

                class _Result:
                    deleted_count = 1

                return _Result()

        class _Result:
            deleted_count = 0

        return _Result()

    def count_documents(self, query=None):
        return len(self.find(query))


class FakeDB:
    def __init__(self):
        demo = _demo_db()
        self.brands = FakeColl(list(demo.brands.docs))
        self.generics = FakeColl(list(demo.generics.docs))
        self.interactions = FakeColl(list(demo.interactions.docs))
        self.aliases = FakeColl(list(demo.aliases.docs))
        self.medicine_requests = FakeColl([])


def stub_intake(image_path, mode="auto", drugdb=None, **kwargs):
    import copy
    stub_intake.last_mode = mode
    return copy.deepcopy(STUB_PRESCRIPTION)


stub_intake.last_mode = None


def stub_lookup(name):
    if name == "Enzoflam":
        return dict(ENZOFLAM_DRAFT)
    return None


class TestApi(unittest.TestCase):
    def setUp(self):
        self.fake_db = FakeDB()
        self.drugdb = DrugDB(self.fake_db)
        api.app.dependency_overrides[api.get_db] = lambda: self.fake_db
        api.app.dependency_overrides[api.get_drugdb] = lambda: self.drugdb
        self._orig = (api.run_intake, api.lookup_brand, api.verify_source,
                      api.web_verify_pair)
        api.run_intake = stub_intake
        api.lookup_brand = stub_lookup
        api.verify_source = lambda url, ings: True
        api.web_verify_pair = lambda drugdb, a, b: None
        self.client = TestClient(api.app, raise_server_exceptions=False)
        self.headers = {"X-Admin-Password": "test-admin"}

    def tearDown(self):
        (api.run_intake, api.lookup_brand, api.verify_source,
         api.web_verify_pair) = self._orig
        api.app.dependency_overrides.clear()
        if api.UPLOAD_DIR.exists():
            shutil.rmtree(api.UPLOAD_DIR)

    def _scan(self):
        return self.client.post(
            "/api/scan",
            files={"image": ("rx.jpg", b"fake-image-bytes", "image/jpeg")},
            data={"list": "prescription", "handwritten": "false"})

    def test_scan_ignores_handwritten_field_and_uses_auto(self):
        # PHASE4_UI_SPEC §1: the frontend no longer sends a mode choice;
        # the API must always run intake in auto mode, even when an old
        # client still sends handwritten=true.
        for value in ("true", "false"):
            resp = self.client.post(
                "/api/scan",
                files={"image": ("rx.jpg", b"fake-image-bytes",
                                 "image/jpeg")},
                data={"list": "prescription", "handwritten": value})
            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertEqual(stub_intake.last_mode, "auto")

    def test_health(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True})

    def test_scan_resolves_unconfirmed_via_web_and_upserts_request(self):
        resp = self._scan()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        meds = {m["brand"]: m for m in body["prescription"]["medicines"]}
        self.assertEqual(meds["Napa"]["provenance"], "database")
        self.assertEqual(meds["Enzoflam"]["provenance"], "web-unverified")
        self.assertEqual(meds["Enzoflam"]["source"], ENZOFLAM_DRAFT["source"])
        self.assertEqual(body["prescription"]["needs_confirmation"], [])
        self.assertEqual(body["prescription"]["list"], "prescription")
        # Enzoflam + Napa share paracetamol: a duplicate finding, so
        # the report is Warning — the web composition was really used.
        self.assertIsNotNone(body["report"])
        self.assertEqual(body["report"]["risk_level"], "Warning")
        self.assertEqual(body["provenance"]["Enzoflam"], "web-unverified")
        # Review queue: one pending request, scan_count 1.
        listed = self.client.get("/api/admin/medicine-requests",
                                 headers=self.headers)
        self.assertEqual(listed.status_code, 200)
        requests = listed.json()["requests"]
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["brand"], "Enzoflam")
        self.assertEqual(requests[0]["scan_count"], 1)
        self.assertEqual(requests[0]["status"], "pending")
        self.assertTrue(requests[0]["source_verified"])
        # A second scan of the same brand bumps scan_count, no duplicate.
        self.assertEqual(self._scan().status_code, 200)
        listed = self.client.get("/api/admin/medicine-requests",
                                 headers=self.headers)
        requests = listed.json()["requests"]
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["scan_count"], 2)

    def test_scan_lookup_failure_keeps_needs_confirmation(self):
        api.lookup_brand = lambda name: None
        resp = self._scan()
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body["prescription"]["medicines"]), 1)
        self.assertEqual(len(body["prescription"]["needs_confirmation"]), 1)
        self.assertIsNotNone(body["report"])  # Napa alone still assessed
        listed = self.client.get("/api/admin/medicine-requests",
                                 headers=self.headers)
        self.assertEqual(listed.json()["requests"], [])

    def test_approve_then_reject_flow(self):
        self.assertEqual(self._scan().status_code, 200)
        listed = self.client.get("/api/admin/medicine-requests",
                                 headers=self.headers).json()["requests"]
        req_id = listed[0]["id"]
        approved = self.client.post(
            f"/api/admin/medicine-requests/{req_id}/approve",
            headers=self.headers)
        self.assertEqual(approved.status_code, 200, approved.text)
        self.assertEqual(approved.json(), {"ok": True, "brand": "Enzoflam"})
        # The brand is now in the database (add_entry ran).
        self.assertIsNotNone(self.drugdb.resolve_brand("Enzoflam"))
        done = self.client.get(
            "/api/admin/medicine-requests?status=approved",
            headers=self.headers).json()["requests"]
        self.assertEqual([r["brand"] for r in done], ["Enzoflam"])
        # Reject path: a fresh pending request, rejected with a reason.
        self.fake_db.medicine_requests.insert_one({
            "brand": "Pan D", "brand_key": "pan d",
            "generic_title": "Pantoprazole + Domperidone",
            "ingredients": ["Pantoprazole", "Domperidone"],
            "strength_parts": ["40 mg", "30 mg"],
            "dosage_form": "Tablet", "company": "Alkem",
            "source": "https://www.practo.com/medicine-info/pan-d",
            "source_verified": True, "scan_count": 1,
            "first_seen": "2026-11-16T00:00:00+00:00",
            "last_seen": "2026-11-16T00:00:00+00:00",
            "status": "pending"})
        pending = self.client.get("/api/admin/medicine-requests",
                                  headers=self.headers).json()["requests"]
        pan_id = [r for r in pending if r["brand"] == "Pan D"][0]["id"]
        rejected = self.client.post(
            f"/api/admin/medicine-requests/{pan_id}/reject",
            json={"reason": "wrong composition"}, headers=self.headers)
        self.assertEqual(rejected.status_code, 200)
        self.assertEqual(rejected.json(), {"ok": True})
        gone = self.client.get(
            "/api/admin/medicine-requests?status=rejected",
            headers=self.headers).json()["requests"]
        self.assertEqual(gone[0]["brand"], "Pan D")
        self.assertEqual(gone[0]["reason"], "wrong composition")

    def test_admin_wrong_password_is_401(self):
        resp = self.client.get("/api/admin/medicine-requests",
                               headers={"X-Admin-Password": "wrong"})
        self.assertEqual(resp.status_code, 401)
        self.assertIn("detail", resp.json())
        resp = self.client.get("/api/admin/medicine-requests")
        self.assertEqual(resp.status_code, 401)

    def test_admin_unknown_request_is_404(self):
        resp = self.client.post("/api/admin/medicine-requests/999/approve",
                                headers=self.headers)
        self.assertEqual(resp.status_code, 404)
        self.assertIn("detail", resp.json())

    def test_missing_image_is_4xx_with_detail(self):
        resp = self.client.post("/api/scan", data={"list": "prescription"})
        self.assertGreaterEqual(resp.status_code, 400)
        self.assertLess(resp.status_code, 500)
        self.assertIn("detail", resp.json())

    def test_report_endpoint_returns_pdf(self):
        resp = self.client.post("/api/report", json={
            "prescription": STUB_PRESCRIPTION,
            "report": {"risk_level": "Low", "findings": [], "coverage": {}}})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("application/pdf", resp.headers["content-type"])
        self.assertTrue(resp.content.startswith(b"%PDF"))

    def test_chat_endpoint_uses_answer_question(self):
        orig = api.answer_question
        api.answer_question = lambda messages, context, drugdb=None: {
            "answer": "No interaction was found in this scan.\n\n"
                      "For information only — not medical advice. "
                      "Ask your pharmacist or doctor before changing any medicine.",
            "sources": [{"id": "risk-level", "source": "safety report",
                         "text": "Overall risk level: Low."}]}
        try:
            resp = self.client.post("/api/chat", json={
                "messages": [{"role": "user", "content": "Any interaction?"}],
                "context": {"prescription": STUB_PRESCRIPTION,
                            "report": {"risk_level": "Low", "findings": []}}})
        finally:
            api.answer_question = orig
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(resp.json()["answer"].endswith(
            "Ask your pharmacist or doctor before changing any medicine."))
        self.assertTrue(resp.json()["sources"])

    def test_chat_failure_is_503_not_a_made_up_answer(self):
        orig = api.answer_question

        def _boom(messages, context, drugdb=None):
            raise RuntimeError("Embeddings unavailable: test")

        api.answer_question = _boom
        try:
            resp = self.client.post("/api/chat", json={
                "messages": [{"role": "user", "content": "hi"}],
                "context": {}})
        finally:
            api.answer_question = orig
        self.assertEqual(resp.status_code, 503)
        self.assertIn("Embeddings unavailable", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
