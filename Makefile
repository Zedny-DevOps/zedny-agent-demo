# ── Convenience commands ──────────────────────────────────────────────────────
# Usage: make <target>

PY ?= python

.PHONY: run setup deploy-modal smoke-modal run-backend run-frontend lint

run:
	$(PY) run.py

setup:
	$(PY) run.py --setup-only

deploy-modal:
	modal deploy services/modal_app.py

smoke-modal:
	modal run services/modal_app.py::smoke

run-backend:
	uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	streamlit run frontend/app.py

lint:
	ruff check .
