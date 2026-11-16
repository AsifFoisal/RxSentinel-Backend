# RxSentinel — Phase 4 run guide (Windows PowerShell)

Backend: FastAPI on port 8000 (`backend/api.py`). Frontend: Next.js
in `frontend/` (see the frontend README / PHASE4_SPEC.md §4).

## Backend — first time

```powershell
cd <repo>\rxsentinel\backend

# 1. Virtual environment (Python 3.11+)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If PowerShell blocks the activation script, run once as admin:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 2. Dependencies
pip install -r requirements.txt

# 3. Environment — copy the example and fill in your values
Copy-Item .env.example .env
notepad .env
```

Fill in `.env` at minimum:

| Key | What it is |
| --- | --- |
| `MONGO_URI` / `MONGO_DB` | Your MongoDB (local `mongodb://127.0.0.1:27017` or Atlas; database `rxsentinel`) |
| `GEMINI_API_KEY` **or** `LLM_BASE_URL` + `LLM_API_KEY` + `LLM_MODEL` | The LLM used for web-verify and chat (see below) |
| `ADMIN_PASSWORD` | Password for the admin review queue (sent as the `X-Admin-Password` header). **Demo-grade protection for a course project only** — a single shared password, no per-user accounts. Pick a throwaway value, never reuse a real password, and change it from `change-me`. The API fails closed with 401 when this is unset. |
| `EMBEDDING_MODEL` | Embedding model for the document chat, served by the same OpenAI-compatible provider at `LLM_BASE_URL` (e.g. `openai/text-embedding-3-small` on OpenRouter, or `nomic-embed-text` on a local Ollama server). If `LLM_BASE_URL` + `EMBEDDING_MODEL` are unset, chat falls back to a local sentence-transformers model **only if installed** (`pip install sentence-transformers` — heavy); otherwise `/api/chat` returns a clear 503 instead of a made-up answer. |

The chat LLM is the same provider as web_verify (`_make_client` in
`web_verify.py`): a generic OpenAI-compatible provider wins when
`LLM_BASE_URL`/`LLM_MODEL` are set, otherwise native Gemini
(`GEMINI_API_KEY`).

## Backend — every run

```powershell
cd <repo>\rxsentinel\backend
.\.venv\Scripts\Activate.ps1
uvicorn api:app --port 8000
```

Check it: open http://127.0.0.1:8000/api/health — `{"ok": true}`.
Uploads land in `backend/uploads/` (gitignored). `.env` is never
committed (see `backend/.gitignore`).

## Frontend

```powershell
cd <repo>\rxsentinel\frontend
npm install
npm run dev
```

Open http://localhost:3000. The frontend talks to the API at
`NEXT_PUBLIC_API_URL` (default `http://127.0.0.1:8000`).
Production build check: `npm run build`.

On `/admin`, enter the `ADMIN_PASSWORD` from `.env` once — the
frontend keeps it in sessionStorage and sends it as the
`X-Admin-Password` header.

## Tests (backend)

```powershell
cd <repo>\rxsentinel\backend
.\.venv\Scripts\Activate.ps1
python test_safety_engine.py
python test_intake.py
python test_web_verify.py
python test_lookup_medicine.py
python test_enrich_mechanisms.py
python test_add_medicine.py
python test_api.py
python test_report_pdf.py
python test_doc_chat.py
```

`test_api.py`, `test_report_pdf.py` and `test_doc_chat.py` need the
Phase 4 deps (`fastapi`, `fpdf2`, `chromadb` via `pip install -r
requirements.txt`); the Phase 1–3 suites stay offline and need
nothing extra.
