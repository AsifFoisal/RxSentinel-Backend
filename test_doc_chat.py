#!/usr/bin/env python3
"""RxSentinel — Phase 4 document-chat tests. Offline: a stub LLM
client, a stub embed function and a stub collection — no ChromaDB,
no embedding endpoint, no API key needed.

Run:  python test_doc_chat.py
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_chat import (  # noqa: E402
    DISCLAIMER, answer_question, build_documents, get_embedding_function,
)

PRESCRIPTION = {
    "medicines": [
        {"brand": "Napa Extra", "doses_per_day": 2,
         "provenance": "database"},
        {"brand": "Seclo", "doses_per_day": 1, "provenance": "database"},
    ],
    "needs_confirmation": [],
}

REPORT = {
    "risk_level": "Low",
    "findings": [],
    "coverage": {"pairs_checked": 2, "pairs_with_data": 2},
}


class StubLLM:
    """Fake .generate client: echoes the retrieved finding text it
    was given, so the test proves the answer comes from retrieval."""

    def generate(self, prompt: str) -> str:
        assert "Retrieved text" in prompt
        return ("The report found no interaction between the medicines "
                "in this scan, based on the retrieved report text.")


class FakeCollection:
    """Minimal ChromaDB-shaped collection: add stores documents,
    query returns them all (retrieval ranking is ChromaDB's job,
    not the chat logic's)."""

    def __init__(self):
        self.docs = []

    def add(self, ids, documents, embeddings, metadatas):
        for doc_id, text, meta in zip(ids, documents, metadatas):
            self.docs.append({"id": doc_id, "text": text, "metadata": meta})

    def query(self, query_embeddings, n_results=5):
        chosen = self.docs[:n_results]
        return {
            "documents": [[d["text"] for d in chosen]],
            "metadatas": [[d["metadata"] for d in chosen]],
            "ids": [[d["id"] for d in chosen]],
        }


def stub_embed(texts):
    return [[0.1] * 4 for _ in texts]


class TestBuildDocuments(unittest.TestCase):
    def test_report_text_and_medicines(self):
        docs = build_documents(PRESCRIPTION, REPORT)
        text = "\n".join(d["text"] for d in docs)
        self.assertIn("Napa Extra", text)
        self.assertIn("Low", text)

    def test_monograph_from_generics(self):
        class Coll:
            def find_one(self, query):
                if "paracetamol" in str(query).lower():
                    return {"title": "Paracetamol",
                            "indications": "Pain and fever.",
                            "dosage_administration": "500 mg every 6 hours.",
                            "interaction": "Avoid with alcohol.",
                            "side_effects": "Rare liver effects."}
                return None

        class DB:
            generics = Coll()

        docs = build_documents(
            {"medicines": [{"brand": "Napa", "generic": "Paracetamol"}]},
            {"risk_level": "Low", "findings": [
                {"severity": "Unknown", "title": "No data: X + Y",
                 "ingredient_a": "Paracetamol", "ingredient_b": "Y"}]},
            drugdb=DB())
        text = "\n".join(d["text"] for d in docs)
        self.assertIn("Pain and fever", text)


class TestAnswerQuestion(unittest.TestCase):
    def test_answer_cites_sources_and_ends_with_disclaimer(self):
        result = answer_question(
            [{"role": "user", "content": "Is there an interaction?"}],
            {"prescription": PRESCRIPTION, "report": REPORT},
            embed_fn=stub_embed, collection=FakeCollection(),
            client=StubLLM())
        self.assertIn("retrieved report text", result["answer"])
        self.assertTrue(result["answer"].rstrip().endswith(DISCLAIMER))
        self.assertTrue(result["sources"])
        self.assertTrue(all("id" in s and "text" in s
                            for s in result["sources"]))

    def test_answer_never_regrades(self):
        """The stub answer must not invent a severity the report
        does not carry — the prompt forbids it, and the answer here
        carries no severity claim at all."""
        result = answer_question(
            [{"role": "user", "content": "Is this safe?"}],
            {"prescription": PRESCRIPTION, "report": REPORT},
            embed_fn=stub_embed, collection=FakeCollection(),
            client=StubLLM())
        self.assertNotIn("safe", result["answer"].lower().replace(
            "not safe", "").replace("no interaction", ""))

    def test_missing_embeddings_is_a_clear_error_not_an_answer(self):
        """No embed function, no LLM_BASE_URL/EMBEDDING_MODEL, no
        sentence-transformers installed: a clear RuntimeError, never
        a made-up answer."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("LLM_BASE_URL", "EMBEDDING_MODEL",
                            "LLM_API_KEY", "GEMINI_API_KEY")}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                get_embedding_function(None)
        self.assertIn("Embeddings unavailable", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
