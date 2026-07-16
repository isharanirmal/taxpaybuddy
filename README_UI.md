# TaxPayBuddy UI — setup

Two pieces: a FastAPI backend that wraps the existing `RouterAgent`, and a
React (Vite) frontend chat UI.

## 1. Backend

1. Copy `app.py` into the **root** of your `TaxPayBuddy` project — the same
   folder as `src/`, `requirements.txt`, and `.env`. It imports from `src...`
   so it must run from that root.
2. `fastapi` and `uvicorn` are already in `requirements.txt`, so just:
   ```
   pip install -r requirements.txt
   ```
3. Run it from the project root:
   ```
   uvicorn app:app --reload --port 8000
   ```
4. Check it's alive: open `http://localhost:8000/api/health` — should say
   `{"status": "ok"}`.

## 2. Frontend

1. Copy the whole `frontend/` folder anywhere convenient (e.g. next to your
   `TaxPayBuddy/` project folder, not inside `src/`).
2. Install and run:
   ```
   cd frontend
   npm install
   npm run dev
   ```
3. Open the URL Vite prints (usually `http://localhost:5173`).

The frontend talks to `http://localhost:8000` by default. If you run the
backend on a different port/host, copy `.env.example` to `.env` inside
`frontend/` and change `VITE_API_URL`.

## What you get

- A chat interface styled like a tax ledger (teal/gold, serif header).
- Every answer shows a small stamp: which specialist agent handled the
  query (TIN / Individual Income Tax / Corporate Income Tax / Withholding
  Tax / General fallback) — this doubles as a nice visual for demoing the
  multi-agent routing in your presentation.
- A collapsible "Sources" list under each answer, showing which PDF chunks
  were retrieved.

## Notes / things to double check before demo day

- CORS in `app.py` only allows `localhost:5173` right now. If you deploy
  the frontend somewhere else, add that origin to `allow_origins`.
- The backend keeps one shared `RouterAgent` (and one Gemini client) alive
  for the whole server process — fine for a demo/single evaluator, but if
  multiple people hit it at once, requests will queue since the Gemini key
  is called once per question per your existing design.
- If several teammates need to demo separately, each just needs their own
  `GEMINI_API_KEY` in `.env` and to run `uvicorn` locally — no shared
  server needed.
