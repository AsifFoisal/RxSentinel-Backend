#!/usr/bin/env python3
"""RxSentinel — web-verified interaction fallback.

The engine's rule: a pair with no DDInter record is Unknown, never
safe. This module is the fallback for exactly those pairs — it asks
an LLM with web search to check a trusted source and returns a
structured verdict. It never invents a severity: the severity must
come from the source page, and the source URL is stored on the row
so a human can check it.

Flow:
    1. safety_engine finds a pair Unknown (no DDInter record, or
       DDInter lists it as ungraded Unknown). Database first: if
       MongoDB already holds a graded verdict for the pair, it is
       answered from there — no web call (--force-web re-asks the
       LLM, but a graded DB row is still never overwritten).
    2. Only for a missing or ungraded-Unknown pair, web_verify asks
       the LLM, restricted to TRUSTED_SOURCES (Drugs.com checker
       first, then FDA label / DailyMed / MedlinePlus / Medscape).
    3. The answer is returned as JSON: severity (Major / Moderate /
       Minor / None), mechanism, management, source URL, confidence.
    4. The verdict is stored STRAIGHT into MongoDB's `interactions`
       collection — no CSV file, no manual load_ddinter step. Next
       time the same pair is checked, the engine finds it in the
       database and answers instantly, with no web call.
       Two guards on the store: an Unknown verdict is never stored,
       and a graded DDInter row (Major/Moderate/Minor/None) is never
       overwritten by a web result — only a missing or ungraded row
       is filled in. Stored rows carry verified_via="web" so they
       are distinguishable from DDInter rows.
       If MongoDB is unreachable, a graded verdict is never lost:
       it is rescued to data/web_verified_interactions.csv (the
       curated cache load_ddinter.py reads) to review and load once
       the database is back — never silently dropped.

Why a search-capable LLM and not scraping: Drugs.com has no public
API and blocks bulk scraping, so a scraper is the wrong tool.

Two LLM backends:
  1. Native Gemini (grounded Google Search): GEMINI_API_KEY env var
     or --api-key. This is the same key Phase 3 uses for handwriting
     OCR. NOTE: Gemini's free tier has a request limit — when it is
     hit, use backend 2 instead.
  2. Any OpenAI-compatible provider: --base-url + --model + --api-key
     (or LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in .env) — OpenRouter,
     Groq, DeepSeek, a local Ollama / LM Studio, anything that speaks
     /chat/completions. The generic path sends the same prompt but has
     no native search grounding, so review the cited URL extra
     carefully before trusting its verdict.

MongoDB: --mongo-uri / --db, or MONGO_URI / MONGO_DB from .env (same
file as the keys — copy MONGO_URI from env.example). Falls back to
local mongodb://127.0.0.1:27017 with a printed note when unset.

Without any key, run with --dry-run to see the exact prompt that
would be sent, or use the tests (test_web_verify.py) which use stubs.

Usage:
    python web_verify.py --pair Cefuroxime Paracetamol
    python web_verify.py --pair Cefuroxime Paracetamol --dry-run
    # any OpenAI-compatible provider instead of Gemini:
    python web_verify.py --pair Cefuroxime Paracetamol \
        --base-url https://openrouter.ai/api/v1 \
        --api-key sk-or-v1-... --model google/gemini-2.5-flash
    python web_verify.py --batch data/still_missing_after_salt_fix.csv --limit 5
    # graded verdicts land in MongoDB during the run; if it is down they
    # are rescued to data/web_verified_interactions.csv — review that CSV
    # and load it with load_ddinter.py. --csv-out is an optional export.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

try:
    # Same canonicalisation as the loader and the engine (salt forms
    # stripped, Acetaminophen -> Paracetamol, ...), so a web-verified
    # row is stored under exactly the name the engine looks up.
    from load_ddinter import _canonical_drug
except ImportError:  # web_verify.py copied away from backend/
    def _canonical_drug(name: str) -> str:
        n = re.sub(r"\s+", " ", (name or "").strip().lower())
        return {"acetaminophen": "paracetamol",
                "acetylsalicylic acid": "aspirin"}.get(n, n)

TRUSTED_SOURCES = (
    "drugs.com",
    "fda.gov",
    "dailymed.nlm.nih.gov",
    "medlineplus.gov",
    "medscape.com",
    "ncbi.nlm.nih.gov",
)

SEVERITIES = ("Major", "Moderate", "Minor", "None")

# Where graded verdicts land when MongoDB is unreachable — the curated
# cache load_ddinter.py already reads, so nothing is lost and a human
# reviews + loads it once the database is back.
RESCUE_CSV = Path(__file__).resolve().parent / "data" / "web_verified_interactions.csv"

# Flash models, newest first. Google retires old models over time (a
# retired one answers 404 "no longer available" — gemini-2.0-flash was
# retired in Nov 2026), so generate() walks this list and uses the first
# model the API still serves. Update this tuple when models change; no
# other code names a model.
GEMINI_MODELS = (
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)

PROMPT_TEMPLATE = """Check the drug-drug interaction between "{a}" and "{b}".

Use Google Search and read ONLY these trusted sources, in this order:
1. Drugs.com interaction checker (drugs.com/drug-interactions)
2. FDA drug label / DailyMed (dailymed.nlm.nih.gov)
3. MedlinePlus / Medscape

Rules:
- If the source says there is no interaction, severity is "None".
- If it lists an interaction, give its severity exactly as the source
  grades it: "Major", "Moderate" or "Minor". Do not upgrade or
  downgrade it, and do not guess a severity if the source does not
  give one — say "Unknown" instead.
- Quote the mechanism in the source's own words, briefly.
- Cite the EXACT pair page you read, e.g.
  https://www.drugs.com/drug-interactions/amoxicillin-with-pantoprazole-187-0-1790-0.html
  Never cite the checker's homepage or index
  (https://www.drugs.com/drug-interactions) — a verdict whose only
  source is the index will be rejected, because the index page says
  nothing about this pair.
- Reply with JSON ONLY, no other text:
{{"drug_a": "{a}", "drug_b": "{b}", "severity": "Major|Moderate|Minor|None|Unknown",
  "mechanism": "...", "management": "...",
  "source": "https://... (the exact page you read)",
  "confidence": "high|medium|low"}}"""


def build_prompt(a: str, b: str) -> str:
    return PROMPT_TEMPLATE.format(a=a, b=b)


def _source_is_trusted(url: str) -> bool:
    host = re.sub(r"^https?://(www\.)?", "", (url or "").strip().lower())
    return any(host.startswith(s) or f".{s}" in host.split("/")[0]
               for s in TRUSTED_SOURCES)


def _source_is_pair_specific(url: str) -> bool:
    """The cited page must identify the pair, not just the source's
    homepage or checker index. The domain check alone cannot catch
    this: a model can cite https://www.drugs.com/drug-interactions
    (the checker index) and the verdict gets stored as if it were
    read from a pair page — that is how an ungrounded Moderate for
    Amoxicillin + Serratiopeptidase entered the database in Nov 2026
    (no Drugs.com pair page exists for it; the cited index says
    nothing about the pair). The rule: the URL path must have at
    least two non-empty segments — a real pair page
    (/drug-interactions/diclofenac-pantoprazole/) or a deep link
    (/druginfo/meds/a601...), never a bare index (/drug-interactions)
    or homepage (/). This is a URL-shape check, not a page-content
    check: Drugs.com answers 403 to plain fetches, so the page itself
    cannot be verified from here — the shape is the strongest check
    available, and a rejected verdict stays Unknown (safe) instead of
    stored."""
    try:
        path = urlparse(url).path
    except Exception:
        return False
    return len([p for p in path.split("/") if p]) >= 2


def parse_verdict(text: str) -> dict:
    """Parse Gemini's JSON reply. Raises ValueError on anything unusable
    — a bad reply must fail loudly, never be stored as a verdict."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object in model reply: {text[:200]!r}")
    data = json.loads(match.group(0))
    sev = str(data.get("severity", "")).strip()
    if sev not in SEVERITIES and sev != "Unknown":
        raise ValueError(f"Severity {sev!r} is not Major/Moderate/Minor/None/Unknown")
    source = str(data.get("source", "")).strip()
    if not source or not _source_is_trusted(source):
        raise ValueError(f"Source {source!r} is not a trusted source: {TRUSTED_SOURCES}")
    if not _source_is_pair_specific(source):
        raise ValueError(
            f"Source {source!r} is the checker's index/homepage, not the "
            "pair page — cite the exact pair page you read")
    return {
        "drug_a": str(data.get("drug_a", "")).strip(),
        "drug_b": str(data.get("drug_b", "")).strip(),
        "severity": sev,
        "mechanism": str(data.get("mechanism", "")).strip(),
        "management": str(data.get("management", "")).strip(),
        "source": source,
        "confidence": str(data.get("confidence", "low")).strip().lower(),
    }


def verify_pair(a: str, b: str, client=None, api_key: str | None = None,
                base_url: str | None = None, model: str | None = None) -> dict:
    """Verify one pair. `client` is an object with .generate(prompt)
    returning text (used by the tests); otherwise a backend is picked by
    _make_client (generic OpenAI-compatible provider or native Gemini)."""
    prompt = build_prompt(a, b)
    if client is None:
        client = _make_client(api_key, base_url, model)
    try:
        text = client.generate(prompt)
    except Exception as exc:
        # API-level failures (retired model, bad key, quota, outage) print
        # as FAILED rows just like parse errors — never stored as verdicts.
        raise SystemExit(f"LLM API call failed: {exc}") from exc
    verdict = parse_verdict(text)
    verdict["drug_a"] = a
    verdict["drug_b"] = b
    return verdict


def _load_dotenv(path: Path | None = None) -> None:
    """Read backend/.env (same file as MONGO_URI) into os.environ.
    No dependency needed: KEY=VALUE lines only, # comments skipped,
    existing environment variables win. The .env file is never
    committed — see .env.example."""
    env_path = path or Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def _make_client(api_key: str | None, base_url: str | None = None,
                 model: str | None = None):
    """Pick the backend. A generic OpenAI-compatible provider (base_url +
    model, from flags or LLM_* env) wins when configured — it exists
    precisely to bypass Gemini (e.g. its free quota is exhausted).
    Native grounded Gemini is the default otherwise."""
    _load_dotenv()
    base_url = base_url or os.environ.get("LLM_BASE_URL", "")
    model = model or os.environ.get("LLM_MODEL", "")
    if base_url or model:
        key = (api_key or os.environ.get("LLM_API_KEY")
               or os.environ.get("GEMINI_API_KEY") or "")
        missing = [name for name, value in (
            ("LLM_BASE_URL / --base-url", base_url),
            ("LLM_MODEL / --model", model),
            ("LLM_API_KEY / --api-key", key),
        ) if not value]
        if missing:
            raise SystemExit(
                "Generic provider half-configured, missing: " + ", ".join(missing))
        return _generic_client(base_url, key, model)
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "No API key. Set GEMINI_API_KEY (grounded Gemini search), or set "
            "LLM_BASE_URL + LLM_API_KEY + LLM_MODEL for any OpenAI-compatible "
            "provider, or pass --base-url/--model/--api-key.")
    return _gemini_client(key)


def _generic_client(base_url: str, api_key: str, model: str):
    """Client for any OpenAI-compatible provider: one stdlib-urllib POST
    to {base_url}/chat/completions — no SDK dependency, so every provider
    that speaks the protocol (OpenRouter, Groq, DeepSeek, Ollama,
    LM Studio, …) just works."""
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"

    class _Client:
        def generate(self, prompt: str) -> str:
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace").strip()[:300]
                raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
            except OSError as exc:  # URLError, timeout, refused connection
                raise RuntimeError(f"Could not reach {url}: {exc}") from exc
            try:
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(
                    f"Unusable reply from {url}: {json.dumps(body)[:300]}") from exc
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Empty reply from {url}: {json.dumps(body)[:300]}")
            return content

    return _Client()


def _gemini_client(api_key: str | None):
    _load_dotenv()
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "No Gemini API key. Set GEMINI_API_KEY or pass --api-key. "
            "This is the same key Phase 3 uses for handwriting OCR.")
    try:
        from google import genai
        from google.genai import errors as genai_errors
        from google.genai import types
    except ImportError:
        raise SystemExit(
            "google-genai is not installed. Run: pip install google-genai")
    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        response_mime_type="application/json",
    )

    class _Client:
        def generate(self, prompt: str) -> str:
            last_error: Exception | None = None
            for model in GEMINI_MODELS:
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    return response.text
                except genai_errors.ClientError as exc:
                    if exc.code != 404:  # anything but "model gone" — raise
                        raise
                    last_error = exc  # retired model — try the next one
            raise SystemExit(
                f"None of the Gemini models {GEMINI_MODELS} could be used: {last_error}"
            ) from last_error

    return _Client()


def _connect_db(mongo_uri: str | None, db_name: str | None):
    """Connect to MongoDB (URI/DB from flags or .env)."""
    _load_dotenv()
    if mongo_uri:
        uri = mongo_uri
    elif os.environ.get("MONGO_URI"):
        uri = os.environ["MONGO_URI"]
    else:
        uri = "mongodb://127.0.0.1:27017"
        print("  Note: no MONGO_URI in .env and no --mongo-uri — using local "
              "MongoDB 127.0.0.1:27017.")
    name = db_name or os.environ.get("MONGO_DB", "rxsentinel")
    try:
        from pymongo import MongoClient
    except ImportError:
        raise SystemExit("pymongo is not installed. Run: pip install pymongo")
    db = MongoClient(uri, serverSelectionTimeoutMS=5000)[name]
    db.interactions.create_index([("drug_a", 1), ("drug_b", 1)], unique=True)
    return db


def rescue_to_csv(verdict: dict, path: Path | None = None) -> bool:
    """MongoDB unreachable: append the graded verdict to the curated CSV
    cache instead of losing it; load it later with load_ddinter.py, which
    canonicalises, sorts and upserts exactly like a direct store would.
    Unknown verdicts are not rescued — there is nothing to cache — and a
    pair already in the CSV is not written twice."""
    if verdict["severity"] == "Unknown":
        return False
    path = Path(path) if path else RESCUE_CSV
    a, b = sorted([_canonical_drug(verdict["drug_a"]),
                   _canonical_drug(verdict["drug_b"])])
    existing_pairs = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as f:
            existing_pairs = [
                tuple(sorted([_canonical_drug(r.get("drug_a") or ""),
                              _canonical_drug(r.get("drug_b") or "")]))
                for r in csv.DictReader(f)]
    if (a, b) in existing_pairs:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "drug_a", "drug_b", "severity", "mechanism", "management", "source"])
        if not exists:
            w.writeheader()
        w.writerow({k: verdict[k] for k in
                    ["drug_a", "drug_b", "severity", "mechanism", "management", "source"]})
    return True


def lookup_db(a: str, b: str, db):
    """The stored row for this pair, canonicalised and sorted exactly
    like the engine's lookup (load_ddinter.load-style). None when the
    database has no row for the pair."""
    a, b = sorted([_canonical_drug(a), _canonical_drug(b)])
    return db.interactions.find_one({"drug_a": a, "drug_b": b})


def needs_web_search(existing) -> bool:
    """True only when the pair is missing or its stored verdict is
    ungraded Unknown. A graded row (Major/Moderate/Minor/None) answers
    the pair — no web call, no LLM cost."""
    return not (existing and existing.get("severity") in SEVERITIES)


def store_in_db(verdict: dict, db) -> bool:
    """Store one verified verdict straight into the `interactions`
    collection. Returns True if stored, False if deliberately skipped.
    Skips: Unknown verdicts (nothing to store) and pairs that already
    have a graded verdict (a web result never overwrites DDInter or a
    curated row). Only a missing or ungraded-Unknown row is filled in.
    Names are canonicalised (salt forms stripped, synonyms mapped) and
    sorted, exactly like load_ddinter.py, so the engine finds the row."""
    a, b = sorted([_canonical_drug(verdict["drug_a"]),
                   _canonical_drug(verdict["drug_b"])])
    if verdict["severity"] == "Unknown":
        print(f"  {a} + {b}: Unknown — not stored.")
        return False
    if a == b:
        print(f"  {a} + {b}: same drug twice — not stored.")
        return False
    existing = lookup_db(a, b, db)
    if existing and existing.get("severity") in SEVERITIES:
        print(f"  Kept existing {existing['severity']} verdict "
              f"({existing.get('source', 'DDInter 2.0')}) — web result not stored over it.")
        return False
    doc = {
        "drug_a": a, "drug_b": b, "severity": verdict["severity"],
        "mechanism": verdict["mechanism"],
        "management": verdict["management"],
        "source": verdict["source"],
        "verified_via": "web",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    db.interactions.update_one({"drug_a": a, "drug_b": b},
                               {"$set": doc}, upsert=True)
    was = existing["severity"] if existing else "no record"
    print(f"  Stored in MongoDB: {a} + {b} = {verdict['severity']} (was: {was})")
    return True


def web_verify_pair(drugdb, a: str, b: str, client=None,
                    api_key: str | None = None, base_url: str | None = None,
                    model: str | None = None) -> dict | None:
    """The adapter the safety engine calls for one pair it cannot
    answer. Database-first: a graded stored row answers with no web
    call; only a missing or ungraded-Unknown pair reaches the LLM.
    A graded verdict is stored (never overwriting a graded row) and
    the stored row is returned, so the engine reports it like any
    database row. Returns None when the pair could not be verified —
    the engine then keeps its Unknown finding. Never raises: a failed
    check (no key, quota, outage, unreachable DB) must not break an
    assessment. `drugdb` is a DrugDB (its .db) or a raw pymongo DB."""
    db = getattr(drugdb, "db", drugdb)
    try:
        existing = lookup_db(a, b, db)
        if not needs_web_search(existing):
            return existing
        verdict = verify_pair(a, b, client=client, api_key=api_key,
                              base_url=base_url, model=model)
        if not verdict or verdict.get("severity") not in SEVERITIES:
            return None
        store_in_db(verdict, db)
        return lookup_db(a, b, db)
    except (Exception, SystemExit) as exc:
        # verify_pair turns API errors into SystemExit (missing key,
        # retired model, quota); store/lookup failures land here too.
        # The outcome is printed (stderr, so it never pollutes the
        # report JSON on stdout): a silent Unknown hides whether the
        # web check never ran, failed, or found nothing — the user
        # needs to tell those apart. Wording is deliberate: this is
        # not an error, it means no trusted verdict exists and the
        # pair honestly stays Unknown.
        print(f"  Web check: no usable verdict for {a} + {b} ({exc}) — pair stays Unknown", file=sys.stderr)
        return None


def remove_pair_from_db(a: str, b: str, db) -> bool:
    """Delete one stored verdict — the correction path for a verdict
    that should never have been stored (e.g. cited only the checker's
    index, or a later check disproves it). The pair returns to
    missing, so the next assessment web-checks it again — which is
    why a removal is only durable together with the pair-specific
    source rule in parse_verdict. Returns True when a row existed."""
    a, b = sorted([_canonical_drug(a), _canonical_drug(b)])
    result = db.interactions.delete_one({"drug_a": a, "drug_b": b})
    if result.deleted_count:
        print(f"  Removed {a} + {b} from MongoDB.")
        return True
    print(f"  No stored row for {a} + {b} — nothing removed.")
    return False


def append_to_cache(verdict: dict, cache_path: Path) -> None:
    """Optional CSV export (only with --csv-out): same columns
    load_ddinter.py reads. Unknown verdicts are NOT written."""
    if verdict["severity"] == "Unknown":
        return
    exists = cache_path.is_file()
    with cache_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "drug_a", "drug_b", "severity", "mechanism", "management", "source"])
        if not exists:
            w.writeheader()
        w.writerow({k: verdict[k] for k in
                    ["drug_a", "drug_b", "severity", "mechanism", "management", "source"]})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", nargs=2, metavar=("DRUG_A", "DRUG_B"))
    ap.add_argument("--batch", help="CSV with drug_a,drug_b columns (e.g. still_missing_after_salt_fix.csv).")
    ap.add_argument("--limit", type=int, default=5, help="Pairs to check from --batch (default 5).")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--base-url", default="",
                    help="OpenAI-compatible base URL, e.g. https://openrouter.ai/api/v1 "
                         "(generic provider; needs --model). Beats LLM_BASE_URL.")
    ap.add_argument("--model", default="",
                    help="Model id for the generic provider, e.g. google/gemini-2.5-flash. "
                         "Beats LLM_MODEL.")
    ap.add_argument("--mongo-uri", default="",
                    help="MongoDB URI. Defaults to MONGO_URI from backend/.env, else local.")
    ap.add_argument("--db", default="",
                    help="Database name. Defaults to MONGO_DB from backend/.env, else rxsentinel.")
    ap.add_argument("--csv-out", default="",
                    help="Optional: also append graded verdicts to this CSV. Off by default — no CSV is created.")
    ap.add_argument("--remove-pair", nargs=2, metavar=("DRUG_A", "DRUG_B"),
                    action="append", default=[],
                    help="Delete a stored verdict (repeatable) — the "
                         "correction path for a row that should never have "
                         "been stored. The pair returns to missing and is "
                         "web-checked again on the next assessment.")
    ap.add_argument("--force-web", action="store_true",
                    help="Skip the MongoDB-first lookup and re-ask the LLM "
                         "(a graded DB row is still never overwritten).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the prompt that would be sent, verify nothing, store nothing.")
    args = ap.parse_args()

    if args.remove_pair:
        # Removal needs no --pair/--batch and no LLM call.
        db = _connect_db(args.mongo_uri or None, args.db or None)
        for a, b in args.remove_pair:
            remove_pair_from_db(a, b, db)
        if not args.pair and not args.batch:
            return 0

    pairs: list[tuple[str, str]] = []
    if args.pair:
        pairs = [tuple(args.pair)]
    elif args.batch:
        with Path(args.batch).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pairs.append((row["drug_a"], row["drug_b"]))
                if len(pairs) >= args.limit:
                    break
    else:
        ap.error("give --pair A B or --batch <csv>")

    db = None
    from_db = stored = skipped = rescued = failed = 0
    if not args.dry_run:
        # Connect before any API call: a dead MongoDB must be known up
        # front, not after verdicts were paid for.
        try:
            db = _connect_db(args.mongo_uri or None, args.db or None)
        except Exception as exc:
            print(f"MongoDB unreachable: {exc}\n"
                  f"  Verdicts will be rescued to {RESCUE_CSV} — review and "
                  "load with load_ddinter.py.\n")

    for a, b in pairs:
        print(f"{a} + {b}:")
        if args.dry_run:
            print(build_prompt(a, b))
            continue
        # Database first: a graded verdict already stored answers the
        # pair instantly, with no web call. Only a missing or ungraded
        # pair reaches the LLM.
        if db is not None and not args.force_web:
            found = lookup_db(a, b, db)
            if not needs_web_search(found):
                print(f"  {found['severity']} — from MongoDB "
                      f"({found.get('source', 'DDInter 2.0')}), no web call.")
                if found.get("mechanism"):
                    print(f"  {str(found['mechanism'])[:200]}")
                from_db += 1
                continue
        try:
            verdict = verify_pair(a, b, api_key=args.api_key or None,
                                  base_url=args.base_url or None,
                                  model=args.model or None)
        except (ValueError, SystemExit) as exc:
            print(f"  FAILED: {exc}")
            failed += 1
            continue
        print(f"  {verdict['severity']} (confidence {verdict['confidence']}) — {verdict['source']}")
        print(f"  {verdict['mechanism'][:200]}")
        if args.csv_out:
            append_to_cache(verdict, Path(args.csv_out))
        if db is None:
            # MongoDB is down: rescue a graded verdict to the CSV. An
            # Unknown verdict is a legitimate answer, not a failure, and
            # a pair already in the CSV is already safe — neither counts
            # as failed.
            if verdict["severity"] == "Unknown":
                skipped += 1
            elif rescue_to_csv(verdict):
                rescued += 1
                print(f"  Rescued to {RESCUE_CSV}")
            else:
                skipped += 1
            continue
        try:
            if store_in_db(verdict, db):
                stored += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"  FAILED to store in MongoDB: {exc}")
            if verdict["severity"] == "Unknown":
                skipped += 1
            elif rescue_to_csv(verdict):
                rescued += 1
                print(f"  Rescued to {RESCUE_CSV}")
            else:
                failed += 1

    if not args.dry_run:
        print(f"\nStored in MongoDB: {stored} · answered from DB: {from_db} · "
              f"kept/skipped: {skipped} · rescued to CSV: {rescued} · "
              f"failed: {failed}")
        if rescued:
            print(f"Review {RESCUE_CSV}, then load: python load_ddinter.py "
                  f"--input {RESCUE_CSV} --mongo-uri \"mongodb+srv://...\"")
        if db is None:
            print("MongoDB was unreachable this run — set MONGO_URI in .env "
                  "(copy it from env.example) or pass --mongo-uri.")
    return 1 if (failed or rescued) else 0


if __name__ == "__main__":
    raise SystemExit(main())
