# Zedny Agent Demo — Gemma-4-31B Multimodal Chat

Production-ready chat system for **Gemma-4-31B-it** running on **Modal** (A100-80GB),
fronted by a **FastAPI** backend and a **Streamlit** UI.

Features:
- Text + image chat (multimodal)
- Real token-by-token streaming (SSE)
- Per-request latency, input/output token counts, USD cost
- JSON request logs, in-process LRU cache, per-IP rate limit
- Configurable pricing (`config/pricing.yaml`)

---

## Architecture

```
Streamlit  ──HTTP/SSE──▶  FastAPI  ──modal.Cls──▶  Modal A100-80GB (Gemma4Chat)
frontend/                 backend/                   services/modal_app.py
```

The Modal app (`services/modal_app.py`) deploys as a **separate** Modal app named
`gemma4-chat` and reuses the existing `gemma4-weights` Modal Volume. The original
`gemma4-31b` app at `../modal-test/gemma4_modal.py` is left untouched.

---

## Prerequisites

1. Python ≥ 3.11
2. A Modal account with billing for A100-80GB
3. Modal CLI authenticated:
   ```
   pip install modal
   modal setup
   ```
4. HuggingFace secret created in Modal:
   ```
   modal secret create huggingface-secret HF_TOKEN=hf_xxxx
   ```
5. (Optional — `python run.py` will do this for you) Model weights cached in
   the `gemma4-weights` Volume. To do it manually:
   ```
   modal run services/modal_app.py::download
   ```
   This runs an in-project downloader and is idempotent (fast no-op if the
   weights are already cached).

---

## Run everything (one command)

```
cp .env.example .env
python run.py
```

That's it. `run.py` will:

1. Create `.venv/` and install all dependencies
2. Verify Modal CLI auth
3. Deploy `services/modal_app.py` if it isn't deployed yet (use `--deploy` to force redeploy)
4. **Ensure the model weights are in the `gemma4-weights` Volume** (idempotent — first run takes 30-60 min, subsequent runs skip via a marker)
5. Start the FastAPI backend (port 8000) and wait for `/health`
6. Start the Streamlit frontend (port 8501)
7. Open the browser

Press **Ctrl+C** to stop both processes.

Useful flags:
```
python run.py --deploy          # force redeploy the Modal app
python run.py --skip-deploy     # skip both deploy and download
python run.py --download-model  # force re-run the downloader
python run.py --skip-download   # skip the weights check (model must already be cached)
python run.py --setup-only      # install deps and exit
python run.py --no-browser      # don't auto-open the browser
python run.py --reinstall       # reinstall deps even if already installed
python run.py --backend-port 9000 --frontend-port 9501
```

---

## Run manually (3 terminals)

If you'd rather run each piece yourself:

```
make deploy-modal      # one-time: modal deploy services/modal_app.py
make smoke-modal       # optional: test the Modal app directly
make run-backend       # uvicorn backend.main:app --reload --port 8000
make run-frontend      # streamlit run frontend/app.py
```

Endpoints:
- `POST /chat` — JSON, returns full reply + metrics
- `POST /chat/stream` — JSON, SSE stream of tokens then a final `done` event
- `POST /chat-with-image` — multipart (`messages` JSON + `image` file)
- `POST /chat-with-image/stream` — multipart, SSE
- `GET  /health`
- `GET  /metrics` — aggregate counters

Try it:
```
curl -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

---

## Config

- `.env` — runtime knobs (Modal app name, rate limit, cache size, backend URL)
- `config/pricing.yaml` — `$/1K tokens` rates used for cost estimation

---

## Project layout

```
backend/    FastAPI app
frontend/   Streamlit UI
services/   Modal app + async Modal client
schemas/    Pydantic models
utils/      cost, metrics, cache, JSON logger
config/     settings + pricing
```

---

## Notes

- The Modal class uses `allow_concurrent_inputs=4` so a single warm container
  can serve multiple chats in parallel.
- The first request after a deploy pays cold-start (~30–60s for model load).
  The FastAPI lifespan triggers a tiny warmup call to amortize this.
- Streaming uses `transformers.TextIteratorStreamer` driven by a background
  thread inside the Modal container; tokens are yielded back to the FastAPI
  process via Modal's generator-method protocol (`.remote_gen.aio`).
