#!/usr/bin/env python3
"""Offline tests for the web verification client (both backends)."""
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from google.genai import errors as genai_errors
from google.genai import types

import web_verify as web_verify


class _FakeResponse:
    """Minimal urlopen() response: context manager + read()."""
    def __init__(self, body):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestGeminiClient(unittest.TestCase):
    @patch("google.genai.Client")
    def test_uses_google_search_grounding(self, client_factory):
        client = client_factory.return_value
        client.models.generate_content.return_value = SimpleNamespace(text="{}")

        verifier = web_verify._gemini_client("test-api-key")
        self.assertEqual(verifier.generate("check this pair"), "{}")

        client_factory.assert_called_once_with(api_key="test-api-key")
        request = client.models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], web_verify.GEMINI_MODELS[0])
        self.assertEqual(request["contents"], "check this pair")
        self.assertEqual(request["config"].response_mime_type, "application/json")
        self.assertIsInstance(request["config"].tools[0], types.Tool)
        self.assertIsInstance(request["config"].tools[0].google_search, types.GoogleSearch)

    @patch("google.genai.Client")
    def test_falls_back_when_a_model_is_retired(self, client_factory):
        """A 404 'model no longer available' must move down GEMINI_MODELS,
        not crash the run."""
        client = client_factory.return_value
        client.models.generate_content.side_effect = [
            genai_errors.ClientError(
                404, {"error": {"message": "This model is no longer available."}}, None),
            SimpleNamespace(text="{}"),
        ]

        verifier = web_verify._gemini_client("test-api-key")
        self.assertEqual(verifier.generate("check this pair"), "{}")

        models = [call.kwargs["model"]
                  for call in client.models.generate_content.call_args_list]
        self.assertEqual(models, list(web_verify.GEMINI_MODELS)[:2])

    @patch("google.genai.Client")
    def test_non_404_client_error_is_raised(self, client_factory):
        """Only a 404 falls back — a bad key (401 etc.) must surface."""
        client = client_factory.return_value
        client.models.generate_content.side_effect = genai_errors.ClientError(
            401, {"error": {"message": "API key not valid."}}, None)

        verifier = web_verify._gemini_client("test-api-key")
        with self.assertRaises(genai_errors.ClientError):
            verifier.generate("check this pair")

    def test_api_failure_is_reported_not_stored(self):
        """A dead API must print FAILED, never write a verdict row."""
        class _Dead:
            def generate(self, prompt):
                raise RuntimeError("503 outage")

        with self.assertRaises(SystemExit) as ctx:
            web_verify.verify_pair("A", "B", client=_Dead())
        self.assertIn("LLM API call failed", str(ctx.exception))


class TestBackendSelection(unittest.TestCase):
    @patch("google.genai.Client")
    def test_gemini_is_default_without_generic_config(self, client_factory):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True), \
                patch.object(web_verify, "_load_dotenv"):
            web_verify._make_client(None, None, None)
        client_factory.assert_called_once_with(api_key="k")

    @patch("urllib.request.urlopen")
    def test_generic_provider_wins_when_configured(self, urlopen):
        """LLM_* env must pick the generic backend over native Gemini."""
        urlopen.return_value = _FakeResponse(
            {"choices": [{"message": {"content": "{}"}}]})
        env = {"LLM_BASE_URL": "https://x.example/v1",
               "LLM_API_KEY": "k-env", "LLM_MODEL": "m-1",
               "GEMINI_API_KEY": "k-gemini"}
        with patch.dict(os.environ, env), patch.object(web_verify, "_load_dotenv"):
            client = web_verify._make_client(None, None, None)
        self.assertEqual(client.generate("p"), "{}")  # generic path ran
        urlopen.assert_called_once()

    def test_half_configured_generic_provider_errors(self):
        """base_url without model (or key) must fail with what's missing."""
        with patch.object(web_verify, "_load_dotenv"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as ctx:
                web_verify._make_client(None, base_url="https://x/v1", model=None)
        self.assertIn("missing", str(ctx.exception))

    def test_no_key_at_all_explains_both_options(self):
        with patch.object(web_verify, "_load_dotenv"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as ctx:
                web_verify._make_client(None, None, None)
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))
        self.assertIn("LLM_BASE_URL", str(ctx.exception))


class TestGenericProvider(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_posts_chat_completions_with_bearer_key(self, urlopen):
        urlopen.return_value = _FakeResponse(
            {"choices": [{"message": {"content": '{"severity": "None"}'}}]})
        client = web_verify._generic_client(
            "https://openrouter.example/api/v1/", "sk-test", "vendor/model")
        self.assertEqual(client.generate("prompt text"), '{"severity": "None"}')

        request = urlopen.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/api/v1/chat/completions"))
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.headers["Authorization"], "Bearer sk-test")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "vendor/model")
        self.assertEqual(payload["messages"][0]["content"], "prompt text")

    @patch("urllib.request.urlopen")
    def test_unusable_reply_fails_loudly(self, urlopen):
        urlopen.return_value = _FakeResponse({"choices": []})
        client = web_verify._generic_client("https://x/v1", "k", "m")
        with self.assertRaises(ValueError):
            client.generate("p")

    @patch("urllib.request.urlopen")
    def test_http_error_is_reported_with_status(self, urlopen):
        import urllib.error
        urlopen.side_effect = urllib.error.HTTPError(
            "https://x/v1/chat/completions", 429, "Too Many Requests", {},
            _FakeResponse({"error": "quota"}))
        client = web_verify._generic_client("https://x/v1", "k", "m")
        with self.assertRaises(RuntimeError) as ctx:
            client.generate("p")
        self.assertIn("HTTP 429", str(ctx.exception))


class TestMongoStorage(unittest.TestCase):
    def _verdict(self, a="Cefuroxime", b="Warfarin", severity="Moderate"):
        return {"drug_a": a, "drug_b": b, "severity": severity,
                "mechanism": "vitamin K", "management": "monitor INR",
                "source": "https://www.drugs.com/drug-interactions/x.html",
                "confidence": "high"}

    @patch("pymongo.MongoClient")
    def test_connect_db_uses_env_uri_and_db(self, mc):
        env = {"MONGO_URI": "mongodb+srv://atlas.example/db0", "MONGO_DB": "db0"}
        with patch.dict(os.environ, env, clear=True), \
                patch.object(web_verify, "_load_dotenv"):
            web_verify._connect_db(None, None)
        mc.assert_called_once_with("mongodb+srv://atlas.example/db0",
                                   serverSelectionTimeoutMS=5000)
        mc.return_value.__getitem__.assert_called_once_with("db0")

    @patch("pymongo.MongoClient")
    def test_connect_db_flag_beats_env_and_defaults_to_local(self, mc):
        with patch.dict(os.environ, {"MONGO_URI": "mongodb+srv://atlas.example/db0"},
                        clear=True), patch.object(web_verify, "_load_dotenv"):
            web_verify._connect_db("mongodb://flag.example:27017", None)
        mc.assert_called_once_with("mongodb://flag.example:27017",
                                   serverSelectionTimeoutMS=5000)

        with patch.dict(os.environ, {}, clear=True), \
                patch.object(web_verify, "_load_dotenv"):
            web_verify._connect_db(None, None)
        self.assertEqual(mc.call_args_list[-1].args[0], "mongodb://127.0.0.1:27017")

    def test_store_never_overwrites_graded_row(self):
        class _Col:
            def find_one(self, query):
                return {"drug_a": query["drug_a"], "drug_b": query["drug_b"],
                        "severity": "Major", "source": "DDInter 2.0"}

            def update_one(self, *args, **kwargs):
                raise AssertionError("must not write over a graded row")

        class _DB:
            interactions = _Col()

        self.assertFalse(web_verify.store_in_db(self._verdict(), _DB()))

    def test_store_fills_missing_row_canonicalised_and_sorted(self):
        saved = {}

        class _Col:
            def find_one(self, query):
                return None

            def update_one(self, query, update, upsert=False):
                saved["query"], saved["upsert"] = query, upsert

        class _DB:
            interactions = _Col()

        self.assertTrue(web_verify.store_in_db(self._verdict("Warfarin", "Cefuroxime"), _DB()))
        self.assertEqual(saved["query"], {"drug_a": "cefuroxime", "drug_b": "warfarin"})
        self.assertTrue(saved["upsert"])

    def test_unreachable_mongodb_rescues_verdict_to_csv(self):
        """The exact production failure: MongoDB down after a graded
        verdict — the row must land in the CSV, and main must exit 1."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "rescued.csv"
            with patch.object(web_verify, "verify_pair",
                              lambda a, b, **kw: self._verdict(a, b)), \
                    patch.object(web_verify, "_connect_db",
                                 side_effect=ConnectionError("refused")), \
                    patch.object(web_verify, "RESCUE_CSV", cache), \
                    patch.object(sys, "argv",
                                 ["web_verify.py", "--pair", "Cefuroxime",
                                  "Warfarin", "--api-key", "k"]):
                rc = web_verify.main()
            self.assertEqual(rc, 1)
            with cache.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["severity"], "Moderate")

    def test_rescue_dedupes_same_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "rescued.csv"
            first = web_verify.rescue_to_csv(self._verdict(), cache)
            second = web_verify.rescue_to_csv(self._verdict(), cache)
            self.assertTrue(first)
            self.assertFalse(second)
            with cache.open(newline="", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 1)


class TestDbFirst(unittest.TestCase):
    """The pair must be answered from MongoDB when it is graded there;
    the LLM is only the fallback for missing/ungraded pairs."""

    def _verdict(self, a="Cefuroxime", b="Warfarin", severity="Moderate"):
        return {"drug_a": a, "drug_b": b, "severity": severity,
                "mechanism": "vitamin K", "management": "monitor INR",
                "source": "https://www.drugs.com/drug-interactions/x.html",
                "confidence": "high"}

    def _fake_db(self, row):
        writes = []

        class _Col:
            def find_one(self, query):
                if row and query == {"drug_a": row["drug_a"],
                                     "drug_b": row["drug_b"]}:
                    return row
                return None

            def update_one(self, *args, **kwargs):
                writes.append(1)

        class _DB:
            interactions = _Col()

        db = _DB()
        return db, writes

    def test_graded_row_answers_from_db_without_web_call(self):
        db, writes = self._fake_db(
            {"drug_a": "cefuroxime", "drug_b": "warfarin", "severity": "Major",
             "mechanism": "m", "management": "g", "source": "DDInter 2.0"})

        def _no_web(*args, **kwargs):
            raise AssertionError("graded DB row must answer — no web call")

        with patch.object(web_verify, "verify_pair", _no_web), \
                patch.object(web_verify, "_connect_db", return_value=db), \
                patch.object(sys, "argv",
                             ["web_verify.py", "--pair", "Warfarin",
                              "Cefuroxime", "--api-key", "k"]):
            rc = web_verify.main()
        self.assertEqual(rc, 0)
        self.assertEqual(writes, [])  # answered, nothing written back

    def test_unknown_row_falls_through_to_web_and_fills_in(self):
        db, writes = self._fake_db(
            {"drug_a": "cefuroxime", "drug_b": "warfarin", "severity": "Unknown"})
        verdict = self._verdict()

        with patch.object(web_verify, "verify_pair",
                          lambda a, b, **kw: verdict), \
                patch.object(web_verify, "_connect_db", return_value=db), \
                patch.object(sys, "argv",
                             ["web_verify.py", "--pair", "Cefuroxime",
                              "Warfarin", "--api-key", "k"]):
            rc = web_verify.main()
        self.assertEqual(rc, 0)
        self.assertEqual(writes, [1])  # the ungraded row was filled in

    def test_force_web_reverifies_but_does_not_overwrite(self):
        db, writes = self._fake_db(
            {"drug_a": "cefuroxime", "drug_b": "warfarin", "severity": "Major",
             "mechanism": "m", "management": "g", "source": "DDInter 2.0"})
        calls = []

        def _web(a, b, **kwargs):
            calls.append(1)
            return self._verdict(a, b)

        with patch.object(web_verify, "verify_pair", _web), \
                patch.object(web_verify, "_connect_db", return_value=db), \
                patch.object(sys, "argv",
                             ["web_verify.py", "--pair", "Cefuroxime",
                              "Warfarin", "--api-key", "k", "--force-web"]):
            rc = web_verify.main()
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [1])  # web was consulted...
        self.assertEqual(writes, [])  # ...but the graded row stands


if __name__ == "__main__":
    unittest.main(verbosity=2)
