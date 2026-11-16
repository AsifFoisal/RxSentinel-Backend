# RxSentinel — Backend (Phase 1: Database + Import · Phase 2: Safety Engine)

## What Phase 1 is
Load the Global Medicine Directory dataset (Aug 2026, 26,205 brands, with the
`ingredients` column already added from `Generic_Title`) into MongoDB.
No UI yet — this is the foundation Phase 2's safety engine builds on.
Phase 2 (`safety_engine.py`) is documented at the bottom of this file.

## Database: MongoDB
Picked at Phase 1 because `ingredients[]` is a native array and each CSV row
maps cleanly to one document. If you prefer PostgreSQL later, the schema below
is the spec to port — the import script would be the only thing to rewrite.

## Collections

### `generics` — one document per generic
Deduped by **normalised `Generic_Title`**, not by `Generic_Row_ID` — the
dataset reuses the same id for different generics (e.g. id 10 is both
"Aceclofenac" and "Aloe + Black Pepper + Henbane"). Keying on the id merges
different generics and keeps the wrong ingredients. The id is stored as a
field only.
| Field | Notes |
|---|---|
| `generic_id` | `Generic_Row_ID`, e.g. `"779"` (null if missing) |
| `title` | `Generic_Title`, kept as-is for display |
| `ingredients` | JSON array, e.g. `["Paracetamol", "Caffeine"]` |
| `generic_url` | MedEx generic page URL |
| `indications`, `pharmacology`, `dosage_administration`, `interaction`, `side_effects`, `pregnancy_lactation`, `precautions_warnings`, `use_in_special_populations`, `therapeutic_class`, `storage_conditions` | Monograph text from the dataset |
| `source`, `price_as_of` | e.g. `"MedEx"`, `"2026-08-21"` |

Group titles are kept exactly as in the dataset — e.g.
`["10 Vitamin & 6 Mineral [Pregnancy and Breast Feeding Formula]"]` stays
ONE ingredient string. Nothing is split, nothing is skipped, nothing is
invented. A group name is not a list of ingredients.

### `brands` — one document per brand row
| Field | Notes |
|---|---|
| `brand_name`, `generic_id`, `generic_title` | |
| `ingredients` | Copied from the generic, so brand-level checks need no join |
| `dosage_form`, `strength_raw`, `strength_parts` | e.g. `"500 mg + 65 mg"` → `["500 mg", "65 mg"]` |
| `company`, `medicine_type` | |
| `price_raw`, `unit_price_bdt`, `pack_price_bdt`, `price_as_of` | Dataset price + date. **Never presented as live.** Check the current price via `medex_url`. |
| `source`, `medex_url` | |

### `interactions` — empty for now
DDInter 2.0 pairs land here in Phase 2.

### `recalls` — empty for now
DGDA recall rows land here later.

### `aliases` — seeded
Verified aliases only. Currently: Paracetamol = Acetaminophen. Add more only
with a source.

## Setup
```bash
# 1. MongoDB running — local, or Atlas (free tier works)
#    Local: install MongoDB Community and start it (default port 27017).
#    Atlas: create a free cluster, copy its connection string.

# 2. Install the driver
pip install -r requirements.txt

# 3. Dry run first — parses everything, writes nothing
python import_data.py --input <your dataset.csv> --dry-run

# 4. Real import
python import_data.py --input <your dataset.csv>

# Atlas example:
python import_data.py --input <your dataset.csv> --mongo-uri "mongodb+srv://<user>:<pass>@<cluster>/"
```

## Useful flags
| Flag | What it does |
|---|---|
| `--dry-run` | Parse + summary only, no writes |
| `--limit N` | Import only the first N rows (testing) |
| `--drop` | Drop the collections before importing |
| `--db NAME` | Database name (default `rxsentinel`) |
| `--price-as-of YYYY-MM-DD` | Date the dataset prices are from (default `2026-08-21`) |
| `--mongo-uri URI` | Connection string (default local `mongodb://127.0.0.1:27017`) |

Re-running the import is safe: rows are upserted by brand + generic + strength,
so it updates instead of duplicating.

## Ingredient conflicts report
If different rows for one generic disagree on `ingredients`, the import keeps
the first value and writes a report next to your input file:
`<input-stem>_ingredient_conflicts.csv` — one row per conflicted generic,
with the kept value, the other values, example brands, and a verdict:

| Verdict | Meaning | Action |
|---|---|---|
| `order_or_case_only` | Same ingredients, different order/case | Harmless, ignore |
| `strength_in_name_only` | e.g. `Paracetamol` vs `Paracetamol 500 mg` | Harmless, ignore |
| `overlapping_but_different` | Share some ingredients, differ on others | Look at it |
| `different_ingredients` | No overlap at all | Check before Phase 2 — the kept value may be wrong |

The report is also written on a `--dry-run`, so you can inspect conflicts
without touching the database: `python import_data.py --input <csv> --dry-run`.
Override the path with `--conflicts-report <path>`.

## What "done" looks like for Phase 1
- `generics` ≈ 1,650 documents, `brands` ≈ 26,205 documents
- `db.brands.find({ingredients: "Paracetamol"})` returns every brand containing paracetamol — including Napa Extra, via its ingredients array
- A group generic (e.g. id 779) stores its full title as one ingredient string

---

# Phase 2 — Safety Engine (no UI)

Pure rule-based checks in `safety_engine.py`. Same medicines always give the
same findings — no LLM judgement here. The LLM/RAG layer (Phase 4) only
explains these findings afterwards.

## The four checks
1. **Duplicate ingredient** — same active ingredient in 2+ medicines,
   including hidden in combos (Napa + Napa Extra). Uses `ingredients[]`,
   with aliases canonicalised (Acetaminophen = Paracetamol).
2. **Drug-drug interaction** — every ingredient pair looked up in the
   `interactions` collection (DDInter 2.0), severity Major / Moderate /
   Minor. A pair with **no record is reported as Unknown — never as safe**.
   The one exception is severity **"None"**: a curated, source-labelled
   verdict that no known interaction exists — checked-and-clear, no finding.
3. **Total daily dose load** — per ingredient, mg per dose × doses per day,
   summed across medicines, vs `data/max_daily_dose.json` (reference adult
   maximums, clearly marked as such — a pharmacist must verify). Over the
   max is High; 75–100% is a Warning. Drugs with no maximum on file are
   grouped into one Unknown note, not one finding each.
4. **Expiry & batch** — when a medicine carries an `expiry` (from the pack
   or receipt, Phase 3), an expired medicine is High and one expiring
   within 30 days is a Warning. Accepted formats: `2027-03`, `03/2027`,
   `2027-03-15`, `Mar 2027`, `15 Mar 2027`; a bare month means the end of
   that month. An optional `batch` is carried into the finding. Medicines
   with no expiry supplied are counted in coverage as not checked — never
   assumed fresh. An expiry value that cannot be parsed is an Unknown
   finding, flagged rather than dropped.

## Risk levels
`High` (any Major interaction, dose over max, or expired medicine) →
`Warning` (any Moderate interaction, duplicate, near max, or expiring
within 30 days) → `Unknown` (no data to check against) → `Low` (no
findings, everything checkable). A Major interaction counts as High risk.

## Interactions data
`load_ddinter.py` loads pairs into the `interactions` collection (upsert by
drug pair, safe to re-run). Start with the curated demo seed (7 well-known
pairs) so the engine works today:
```bash
python load_ddinter.py --input data/sample_interactions.csv
```
For the full data, download the DDInter 2.0 export from the Download page
on https://ddinter2.scbdd.com/download/ (free, CC BY-NC-SA 4.0; 8 per-ATC
CSVs, ~13 MB total as listed on the page) and load each file the same way —
the loader auto-detects the export's columns and writes in bulk_write()
batches of 2,000 (use `--batch-size` to change this), so 302,516 pairs
take dozens of round-trips instead of one per pair:
```bash
python load_ddinter.py --input <ddinter-export.csv> --mongo-uri "mongodb+srv://..."
```
Then load the verified no-interaction pairs **after** the DDInter files —
DDInter lists these three pairs as ungraded Unknown, and loading the
curated file last is what keeps the verified verdict:
```bash
python load_ddinter.py --input data/curated_no_interaction.csv --mongo-uri "mongodb+srv://..."
```
`data/curated_no_interaction.csv` holds 3 pairs (Caffeine+Paracetamol,
Caffeine+Omeprazole, Omeprazole+Paracetamol), each checked against the
Drugs.com interaction checker on 2026-11-14 ("no drug interactions found",
with the source URL on the row). They are the three pairs from the
Napa Extra + Seclo test — with them loaded, that check comes back Low
with all 3 pairs counted as checked, instead of 3× Unknown. Do not add a
pair here without a named source: "None" means verified clear, and a
wrong "None" would hide a real interaction.

## Try it
```bash
# Demo cases, no MongoDB needed:
python safety_engine.py --demo

# Run the tests (offline):
python test_safety_engine.py

# Check a real prescription against your MongoDB:
python safety_engine.py --mongo-uri "mongodb+srv://..." --input prescription.json
# prescription.json: {"medicines": [{"brand": "Napa Extra", "doses_per_day": 2},
#                                    {"brand": "Napa", "doses_per_day": 2}]}
# Pack expiry/batch are optional per medicine:
#   {"brand": "Napa", "doses_per_day": 2, "expiry": "2027-03", "batch": "B12345"}
```
Unknown brand names are reported as unresolved and never guessed; the check
continues with the rest.

---

# Phase 4 — API, safety report PDF, document chat

`api.py` (FastAPI, port 8000) serves the Next.js frontend: `POST
/api/scan`, `GET/POST /api/admin/medicine-requests/...`,
`POST /api/report` (fpdf2, `report_pdf.py`), `POST /api/chat`
(ChromaDB, `doc_chat.py`). Windows run instructions:
`PHASE4_RUN.md`.

## Admin password — demo-grade, not production auth
The review-queue endpoints need header `X-Admin-Password` matching
`ADMIN_PASSWORD` in `.env`. That is a single shared password with no
per-user accounts and no rate limiting — fine for a course project,
not production auth. The API fails closed with 401 when
`ADMIN_PASSWORD` is unset. Never reuse a real password here.

## Document-chat embeddings — which path applies
`doc_chat.py` resolves embeddings in this order: (1) an injected
function (tests); (2) the OpenAI-compatible `/embeddings` endpoint at
`LLM_BASE_URL`, model from `EMBEDDING_MODEL` in `.env` — the
configured path when a provider is set; (3) a local
sentence-transformers model, **only if installed** (`pip install
sentence-transformers`, heavy). If none applies, `/api/chat` returns
a clear 503 instead of a made-up answer. The chat LLM is the same
provider as web_verify (`web_verify._make_client`).

## The one engine change (§5)
`safety_engine.expand_medicine`: a medicine carrying an explicit
`ingredients` list (the API supplies it from a source-verified web
draft, for one assessment only) is expanded directly — mg per
ingredient from `strength_parts` via `parse_strength_mg`, missing or
unparseable part gives mg None and the dose check skips it. Nothing
else in the engine changed; web-sourced compositions are labelled
`web-unverified` with their source and never presented as database
fact, and nothing is written to `brands`/`generics` at scan time —
only the admin approve click writes, via `add_medicine.add_entry`.
