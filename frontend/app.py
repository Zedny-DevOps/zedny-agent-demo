"""Streamlit chat UI for Gemma-4-31B.

Run with:
    streamlit run frontend/app.py

Talks to the FastAPI backend at $BACKEND_URL (default http://localhost:8000).
"""
from __future__ import annotations

import json
import os
from typing import Generator

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Gemma-4 Chat", page_icon="💬", layout="wide")

# ── session state ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "metrics"?}
if "totals" not in st.session_state:
    st.session_state.totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "latency_ms": 0,
        "turns": 0,
    }
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None  # (filename, bytes)


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    backend = st.text_input("Backend URL", value=BACKEND_URL)
    st.caption("Generation parameters")
    max_new_tokens = st.slider("max_new_tokens", 16, 4096, 1024, step=16)
    temperature = st.slider("temperature", 0.0, 2.0, 1.0, step=0.05)
    top_p = st.slider("top_p", 0.0, 1.0, 0.95, step=0.01)
    top_k = st.slider("top_k", 0, 200, 64, step=1)
    enable_thinking = st.toggle("enable_thinking", value=False)
    use_streaming = st.toggle("Streaming", value=True)

    st.divider()
    st.subheader("📊 Session totals")
    t = st.session_state.totals
    c1, c2 = st.columns(2)
    c1.metric("Turns", t["turns"])
    c2.metric("Cost (USD)", f"${t['cost_usd']:.4f}")
    c1.metric("In tokens", t["input_tokens"])
    c2.metric("Out tokens", t["output_tokens"])
    c1.metric("Latency (ms, sum)", t["latency_ms"])

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "latency_ms": 0,
            "turns": 0,
        }
        st.rerun()


# ── chat history render ──────────────────────────────────────────────────────
st.title("💬 Gemma-4-31B Chat")
st.caption("Multimodal chat • token-by-token streaming • per-message metrics")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("metrics"):
            mt = m["metrics"]
            st.caption(
                f"⏱️ {mt['latency_ms']} ms · "
                f"🔢 in {mt['input_tokens']} / out {mt['output_tokens']} tokens · "
                f"💰 ${mt['cost_usd']:.6f}"
                + ("  ·  cache hit" if mt.get("cache_hit") else "")
            )


# ── input area ───────────────────────────────────────────────────────────────
img_file = st.file_uploader(
    "Attach an image (optional)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
)
if img_file is not None:
    st.session_state.pending_image = (img_file.name, img_file.getvalue())
    st.caption(f"📎 attached: {img_file.name} ({len(img_file.getvalue())} bytes)")

prompt = st.chat_input("Ask anything…")


# ── helpers ──────────────────────────────────────────────────────────────────
def _params_dict() -> dict:
    return {
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "enable_thinking": bool(enable_thinking),
    }


def _record_metrics(metrics: dict) -> None:
    t = st.session_state.totals
    t["turns"] += 1
    t["input_tokens"] += int(metrics.get("input_tokens", 0))
    t["output_tokens"] += int(metrics.get("output_tokens", 0))
    t["cost_usd"] += float(metrics.get("cost_usd", 0.0))
    t["latency_ms"] += int(metrics.get("latency_ms", 0))


def _stream_text_response(history: list[dict]) -> tuple[Generator[str, None, None], dict]:
    """Open the SSE text stream and yield token chunks. Captures final metrics."""
    captured: dict = {}

    def gen():
        url = f"{backend}/chat/stream"
        body = {"messages": history, "params": _params_dict()}
        with httpx.stream("POST", url, json=body, timeout=None) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload:
                    continue
                evt = json.loads(payload)
                if evt.get("type") == "token":
                    yield evt.get("text", "")
                elif evt.get("type") == "done":
                    captured.update(evt.get("metrics") or {})
                elif evt.get("type") == "error":
                    yield f"\n\n[error] {evt.get('error')}"

    return gen(), captured


def _stream_image_response(
    history: list[dict],
    image_name: str,
    image_bytes: bytes,
) -> tuple[Generator[str, None, None], dict]:
    captured: dict = {}

    def gen():
        url = f"{backend}/chat-with-image/stream"
        files = {"image": (image_name, image_bytes)}
        data = {"messages": json.dumps(history), "params": json.dumps(_params_dict())}
        with httpx.stream("POST", url, data=data, files=files, timeout=None) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload:
                    continue
                evt = json.loads(payload)
                if evt.get("type") == "token":
                    yield evt.get("text", "")
                elif evt.get("type") == "done":
                    captured.update(evt.get("metrics") or {})
                elif evt.get("type") == "error":
                    yield f"\n\n[error] {evt.get('error')}"

    return gen(), captured


def _blocking_text(history: list[dict]) -> dict:
    url = f"{backend}/chat"
    body = {"messages": history, "params": _params_dict()}
    r = httpx.post(url, json=body, timeout=None)
    r.raise_for_status()
    return r.json()


def _blocking_image(history: list[dict], image_name: str, image_bytes: bytes) -> dict:
    url = f"{backend}/chat-with-image"
    files = {"image": (image_name, image_bytes)}
    data = {"messages": json.dumps(history), "params": json.dumps(_params_dict())}
    r = httpx.post(url, data=data, files=files, timeout=None)
    r.raise_for_status()
    return r.json()


# ── handle submit ────────────────────────────────────────────────────────────
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        if st.session_state.pending_image is not None:
            st.image(st.session_state.pending_image[1], width=240)

    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]
    image = st.session_state.pending_image
    st.session_state.pending_image = None

    with st.chat_message("assistant"):
        try:
            if use_streaming:
                if image is not None:
                    gen, captured = _stream_image_response(history, image[0], image[1])
                else:
                    gen, captured = _stream_text_response(history)
                final_text = st.write_stream(gen)
                metrics = captured
            else:
                if image is not None:
                    payload = _blocking_image(history, image[0], image[1])
                else:
                    payload = _blocking_text(history)
                final_text = payload["message"]["content"]
                metrics = payload["metrics"]
                st.markdown(final_text)
        except httpx.HTTPError as exc:
            st.error(f"Backend error: {exc}")
            final_text = f"[error] {exc}"
            metrics = {}

        if metrics:
            st.caption(
                f"⏱️ {metrics.get('latency_ms', 0)} ms · "
                f"🔢 in {metrics.get('input_tokens', 0)} / out {metrics.get('output_tokens', 0)} tokens · "
                f"💰 ${metrics.get('cost_usd', 0.0):.6f}"
            )
            _record_metrics(metrics)

    st.session_state.messages.append(
        {"role": "assistant", "content": final_text, "metrics": metrics}
    )
    st.rerun()
