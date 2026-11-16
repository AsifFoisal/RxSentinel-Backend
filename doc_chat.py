#!/usr/bin/env python3
"""RxSentinel — Phase 4: document chat (ChromaDB + the web_verify LLM).

Per scan, an ephemeral collection is built from the scan's own
report text (findings + medicines + doses) plus the involved
generics' monograph text from MongoDB (indications, dosage &
administration, interaction, side effects) — retrieved per scan,
not a static dump. The chat explains the report; the rules decide:
it must never re-grade a pair, invent a severity, or contradict
the report. Every answer ends with the standard disclaimer.

Embeddings: OpenAI-compatible /embeddings via LLM_BASE_URL (model
from EMBEDDING_MODEL), using stdlib urllib like web_verify's
generic client. If that endpoint is unavailable, a local
sentence-transformers model is used ONLY if installed; otherwise
a clear error is raised — never a made-up answer. ChromaDB is
imported lazily inside the function that needs it, so importing
this module (and running the tests) needs neither chromadb nor an
embedding endpoint. Tests can inject a stub embed function and a
stub/fake collection.
"""
from __future__ import annotations

import json
import os
import re
import uuid

DISCLAIMER = ("For information only — not medical advice. "
              "Ask your pharmacist or doctor before changing any medicine.")

MONOGRAPH_FIELDS = ("indications", "dosage_administration",
                    "interaction", "side_effects")

CHAT_SYSTEM = """You explain a medicine safety report to the person it
belongs to, using ONLY the retrieved text provided. Rules:
- Explain what the report's findings mean in plain language.
- NEVER re-grade a pair: do not state a severity beyond what the
  report already says, and never call a result "safe" — a Low risk
  level means no findings were found, not that every combination is
  safe, and an Unknown finding means NOT CHECKED, not safe.
- Do not invent medicines, doses, severities, or advice. If the
  retrieved text does not answer the question, say so plainly and
  suggest asking a pharmacist or doctor.
- Cite the retrieved passages you used, by their source labels."""


def _load_dotenv():
    from web_verify import _load_dotenv as _load
    _load()


def build_documents(prescription: dict, report: dict | None,
                    drugdb=None) -> list[dict]:
    """The retrievable text for one scan: report text + monographs.

    Each document is {id, text, metadata{source}}. Monographs come
    from the generics collection for the ingredients involved in the
    scan; a missing monograph is simply skipped, never invented."""
    prescription = prescription or {}
    report = report or {}
    docs: list[dict] = []
    meds = prescription.get("medicines") or []
    med_lines = []
    for med in meds:
        label = med.get("brand") or med.get("generic") or med.get("name") or "?"
        line = str(label)
        if med.get("doses_per_day") is not None:
            line += f", {med['doses_per_day']} doses/day"
        if med.get("strength_parts"):
            line += " (" + ", ".join(str(p) for p in med["strength_parts"]) + ")"
        if med.get("provenance") == "web-unverified":
            line += " [composition from a web search, not yet verified in our database]"
        med_lines.append(line)
    if med_lines:
        docs.append({"id": "medicines",
                     "text": "Medicines in this scan:\n" + "\n".join(med_lines),
                     "metadata": {"source": "scan medicines"}})
    findings = report.get("findings") or []
    for i, finding in enumerate(findings):
        parts = [f"[{finding.get('severity', '')}] "
                 f"{finding.get('title', '')}"]
        for key in ("detail", "management", "source"):
            if finding.get(key):
                parts.append(f"{key.capitalize()}: {finding[key]}")
        docs.append({"id": f"finding-{i}",
                     "text": "\n".join(parts),
                     "metadata": {"source": "safety report finding"}})
    if report.get("risk_level"):
        docs.append({"id": "risk-level",
                     "text": f"Overall risk level for this scan: "
                             f"{report['risk_level']}.",
                     "metadata": {"source": "safety report"}})
    coverage = report.get("coverage") or {}
    if coverage:
        docs.append({"id": "coverage",
                     "text": "Coverage: "
                             + ", ".join(f"{k}: {v}" for k, v in coverage.items()),
                     "metadata": {"source": "safety report coverage"}})
    # Monographs for the generics involved: ingredients come from the
    # report findings (ingredient_a/b, ingredient) and the medicines'
    # explicit ingredient lists.
    names: set[str] = set()
    for finding in findings:
        for key in ("ingredient_a", "ingredient_b", "ingredient"):
            value = finding.get(key)
            if value:
                for part in str(value).split(","):
                    if part.strip():
                        names.add(part.strip())
    for med in meds:
        for ing in med.get("ingredients") or []:
            names.add(str(ing))
        generic = med.get("generic")
        if generic:
            names.add(str(generic))
    if drugdb is not None and names:
        db = getattr(drugdb, "db", drugdb)
        generics = getattr(db, "generics", None)
        if generics is not None:
            for name in sorted(names):
                try:
                    doc = generics.find_one(
                        {"title": {"$regex": f"^{re.escape(name)}$",
                                   "$options": "i"}})
                except Exception:
                    doc = None
                if not doc:
                    continue
                chunks = []
                for field in MONOGRAPH_FIELDS:
                    value = doc.get(field)
                    if value and str(value).strip():
                        chunks.append(f"{field.replace('_', ' ').title()}: {value}")
                if chunks:
                    docs.append({
                        "id": f"monograph-{name.lower().replace(' ', '-')}",
                        "text": f"Monograph — {doc.get('title') or name}:\n"
                                + "\n".join(chunks),
                        "metadata": {"source": f"monograph: {doc.get('title') or name}"},
                    })
    return docs


def _openai_embeddings(base_url: str, api_key: str, model: str):
    """Embedding function for an OpenAI-compatible /embeddings
    endpoint (stdlib urllib, same style as web_verify's generic
    client). Raises RuntimeError with a clear message on failure."""
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/")
    if not url.endswith("/embeddings"):
        url += "/embeddings"

    def embed(texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()[:300]
            raise RuntimeError(
                f"Embedding endpoint {url} returned HTTP {exc.code}: {detail}") from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not reach the embedding endpoint {url}: {exc}") from exc
        try:
            items = sorted(body["data"], key=lambda d: d["index"])
            return [list(item["embedding"]) for item in items]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unusable embedding reply from {url}: "
                f"{json.dumps(body)[:300]}") from exc

    return embed


def _local_embeddings():
    """Local sentence-transformers fallback — only when installed.
    Raises RuntimeError when it is not."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Embeddings unavailable: no OpenAI-compatible embedding "
            "endpoint is configured (LLM_BASE_URL + EMBEDDING_MODEL) "
            "and sentence-transformers is not installed. Set "
            "EMBEDDING_MODEL and LLM_BASE_URL in backend/.env, or "
            "install sentence-transformers for the local fallback.") from exc
    model_name = os.environ.get("LOCAL_EMBEDDING_MODEL",
                                "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> list[list[float]]:
        return [list(map(float, vec))
                for vec in model.encode(texts, convert_to_numpy=True)]

    return embed


def get_embedding_function(embed_fn=None):
    """Resolve the embedding function: an injected stub wins, then
    the OpenAI-compatible endpoint, then a local model if installed.
    Raises a clear RuntimeError when no path works — never a silent
    fallback to a made-up answer."""
    if embed_fn is not None:
        return embed_fn
    _load_dotenv()
    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    model = os.environ.get("EMBEDDING_MODEL", "").strip()
    if base_url and model:
        api_key = (os.environ.get("LLM_API_KEY")
                   or os.environ.get("GEMINI_API_KEY") or "")
        return _openai_embeddings(base_url, api_key, model)
    if base_url or model:
        raise RuntimeError(
            "Embeddings half-configured: set both LLM_BASE_URL and "
            "EMBEDDING_MODEL in backend/.env (or pass an embedding "
            "function).")
    return _local_embeddings()


def _make_collection(documents: list[dict], embed_fn, collection=None):
    """Add the scan's documents to an ephemeral ChromaDB collection.
    A caller-supplied collection (tests) is used as-is; otherwise a
    fresh in-memory collection is created (chromadb imported lazily).
    Returns the collection."""
    if collection is not None:
        collection.add(
            ids=[d["id"] for d in documents],
            documents=[d["text"] for d in documents],
            embeddings=embed_fn([d["text"] for d in documents]),
            metadatas=[d["metadata"] for d in documents])
        return collection
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "ChromaDB is not installed. Run: pip install chromadb "
            "(see backend/requirements.txt).") from exc
    client = chromadb.Client()
    coll = client.create_collection(
        name=f"rxsentinel-scan-{uuid.uuid4().hex[:12]}")
    coll.add(
        ids=[d["id"] for d in documents],
        documents=[d["text"] for d in documents],
        embeddings=embed_fn([d["text"] for d in documents]),
        metadatas=[d["metadata"] for d in documents])
    return coll


def _query_collection(collection, question: str, embed_fn, n_results: int = 5):
    """Retrieve the most relevant documents. Handles both the real
    ChromaDB result shape and stub collections used in tests."""
    query_vec = embed_fn([question])[0]
    try:
        result = collection.query(
            query_embeddings=[query_vec], n_results=n_results)
    except TypeError:
        result = collection.query(query_texts=[question], n_results=n_results)
    docs = result.get("documents") or [[]]
    metas = result.get("metadatas") or [[]]
    ids = result.get("ids") or [[]]
    out = []
    for doc_text, meta, doc_id in zip(docs[0], metas[0], ids[0]):
        out.append({"id": doc_id, "text": doc_text,
                    "metadata": meta or {}})
    return out


def answer_question(messages: list[dict], context: dict | None,
                    drugdb=None, embed_fn=None, collection=None,
                    client=None) -> dict:
    """Answer one chat turn about a scan. `messages` is the chat
    history (last message is the question); `context` is
    {prescription, report}. Returns {answer, sources}. Raises
    RuntimeError (clear message) when embeddings or ChromaDB are
    unavailable, and SystemExit/RuntimeError from the LLM client
    propagates — the API turns those into a 503, never a made-up
    answer."""
    context = context or {}
    prescription = context.get("prescription") or {}
    report = context.get("report") or {}
    documents = build_documents(prescription, report, drugdb=drugdb)
    if not documents:
        raise RuntimeError(
            "Nothing to chat about: the scan has no medicines or "
            "findings to retrieve text from.")
    embed = get_embedding_function(embed_fn)
    coll = _make_collection(documents, embed, collection=collection)
    question = ""
    for message in reversed(messages or []):
        if message.get("role") == "user" and message.get("content"):
            question = str(message["content"])
            break
    if not question:
        question = "Summarise this safety report."
    retrieved = _query_collection(coll, question, embed)
    if not retrieved:
        raise RuntimeError(
            "No relevant text could be retrieved for this question — "
            "the chat cannot answer without retrieved text.")
    retrieved_text = "\n\n".join(
        f"[{r['metadata'].get('source', r['id'])}]\n{r['text']}"
        for r in retrieved)
    history = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}"
        for m in (messages or []))
    prompt = (f"{CHAT_SYSTEM}\n\nRetrieved text for this scan:\n"
              f"{retrieved_text}\n\nConversation so far:\n{history}\n\n"
              f"Answer the last user question from the retrieved text "
              f"above. End your answer with this exact line:\n{DISCLAIMER}")
    if client is None:
        from web_verify import _make_client
        client = _make_client(None)
    answer = client.generate(prompt)
    answer = (answer or "").strip()
    if not answer:
        raise RuntimeError("The chat model returned an empty answer.")
    if not answer.endswith(DISCLAIMER):
        answer = answer.rstrip() + "\n\n" + DISCLAIMER
    sources = [{"id": r["id"],
                "source": r["metadata"].get("source", r["id"]),
                "text": (r["text"] or "")[:300]} for r in retrieved]
    return {"answer": answer, "sources": sources}
