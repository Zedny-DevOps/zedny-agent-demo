"""
Modal app — Gemma-4-31B chat (text + image, with streaming and token counts).

Deploys as a SEPARATE Modal app named "gemma4-chat" so the existing
"gemma4-31b" app at ../modal-test/gemma4_modal.py is left untouched.

The class reuses the same `gemma4-weights` Volume populated by the original
downloader, so no re-download is needed.

Methods (all called via .remote.aio / .remote_gen.aio from the FastAPI backend):
  - chat(messages, **gen_kwargs) -> {text, input_tokens, output_tokens}
  - chat_stream(messages, **gen_kwargs) -> generator of stream events
  - chat_multimodal(messages, images_b64, **gen_kwargs) -> same dict
  - chat_multimodal_stream(messages, images_b64, **gen_kwargs) -> generator

Setup (one-time):
  modal setup
  modal secret create huggingface-secret HF_TOKEN=hf_xxxx
  # populate the volume by running the original downloader once:
  modal run ../modal-test/gemma4_modal.py
  # then deploy this app:
  modal deploy services/modal_app.py
"""
from __future__ import annotations

import modal

# ── App ──────────────────────────────────────────────────────────────────────
app = modal.App("gemma4-chat")

# ── Persistent volume (populated by `download_model` below) ──────────────────
MODEL_VOLUME = modal.Volume.from_name("gemma4-weights", create_if_missing=True)
MODEL_DIR = "/models/gemma-4-31b-it"
MODEL_ID = "google/gemma-4-31B-it"

# ── Container image (matches the original) ───────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "torchvision",
        "transformers>=4.51.0",
        "accelerate>=0.30.0",
        "huggingface_hub>=0.23.0",
        "sentencepiece",
        "protobuf",
        "pillow",
    )
)


# ── Downloader (idempotent — safe to call any number of times) ───────────────
@app.function(
    image=image,
    volumes={"/models": MODEL_VOLUME},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    cpu=4,
    memory=32768,
)
def download_model(force: bool = False) -> bool:
    """Download Gemma-4-31B-it weights into the Modal Volume.

    Idempotent: skips files that are already present unless force=True.
    Returns True if a download (re)ran, False if the model was already cached.
    """
    import os
    from pathlib import Path

    from huggingface_hub import snapshot_download

    config = Path(MODEL_DIR) / "config.json"
    if config.exists() and not force:
        print(f"✓ model already present at {MODEL_DIR} — skipping")
        return False

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN missing — make sure the 'huggingface-secret' Modal secret "
            "holds a real HuggingFace token (https://huggingface.co/settings/tokens)."
        )

    print(f"Downloading {MODEL_ID} → {MODEL_DIR} (this can take 30-60 minutes) ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        token=token,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )
    MODEL_VOLUME.commit()
    print(f"✓ model saved to {MODEL_DIR}")
    return True


# ── Inference class ──────────────────────────────────────────────────────────
@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes={"/models": MODEL_VOLUME},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=4)
class Gemma4Chat:

    @modal.enter()
    def load(self) -> None:
        """Load model once per container."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        print("Loading Gemma-4-31B-it from volume ...")
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print("✅ Model ready on GPU")

    # ── helpers (run inside the container) ───────────────────────────────────
    def _decode_images(self, images_b64: list[str] | None):
        if not images_b64:
            return []
        import base64
        from io import BytesIO

        from PIL import Image

        out = []
        for b64 in images_b64:
            raw = base64.b64decode(b64)
            img = Image.open(BytesIO(raw)).convert("RGB")
            out.append(img)
        return out

    def _build_inputs(
        self,
        messages: list[dict],
        images: list,
        enable_thinking: bool,
    ):
        """Apply chat template and tokenize. If images are provided, attach them
        to the LAST user message in Gemma's content-list format."""
        if images:
            multimodal_messages: list[dict] = []
            user_seen = 0
            last_user_idx = max(
                (i for i, m in enumerate(messages) if m["role"] == "user"),
                default=-1,
            )
            for i, m in enumerate(messages):
                if i == last_user_idx and m["role"] == "user":
                    content_list = [{"type": "image", "image": img} for img in images]
                    content_list.append({"type": "text", "text": m["content"]})
                    multimodal_messages.append({"role": "user", "content": content_list})
                    user_seen += 1
                else:
                    # System/assistant/earlier-user messages stay text-only.
                    multimodal_messages.append(
                        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                    )
            text = self.processor.apply_chat_template(
                multimodal_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            inputs = self.processor(
                text=text,
                images=images,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        return inputs

    def _post_process(self, raw: str) -> str:
        """Strip thinking tags / special tokens like the original does."""
        parser = getattr(self.processor, "parse_response", None)
        if callable(parser):
            try:
                return parser(raw)
            except Exception:
                pass
        return raw.replace("<eos>", "").strip()

    # ── 1. Non-streaming text chat ───────────────────────────────────────────
    @modal.method()
    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        enable_thinking: bool = False,
    ) -> dict:
        import torch

        inputs = self._build_inputs(messages, images=[], enable_thinking=enable_thinking)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
            )

        new_token_ids = output_ids[0][input_len:]
        output_tokens = int(new_token_ids.shape[-1])
        raw = self.processor.decode(new_token_ids, skip_special_tokens=False)
        text = self._post_process(raw)
        return {"text": text, "input_tokens": int(input_len), "output_tokens": output_tokens}

    # ── 2. Streaming text chat ───────────────────────────────────────────────
    @modal.method()
    def chat_stream(
        self,
        messages: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        enable_thinking: bool = False,
    ):
        yield from self._stream_impl(
            messages=messages,
            images_b64=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
        )

    # ── 3. Multimodal chat (non-streaming) ───────────────────────────────────
    @modal.method()
    def chat_multimodal(
        self,
        messages: list[dict],
        images_b64: list[str],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        enable_thinking: bool = False,
    ) -> dict:
        import torch

        images = self._decode_images(images_b64)
        inputs = self._build_inputs(messages, images=images, enable_thinking=enable_thinking)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
            )

        new_token_ids = output_ids[0][input_len:]
        output_tokens = int(new_token_ids.shape[-1])
        raw = self.processor.decode(new_token_ids, skip_special_tokens=False)
        text = self._post_process(raw)
        return {"text": text, "input_tokens": int(input_len), "output_tokens": output_tokens}

    # ── 4. Multimodal chat (streaming) ───────────────────────────────────────
    @modal.method()
    def chat_multimodal_stream(
        self,
        messages: list[dict],
        images_b64: list[str],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        enable_thinking: bool = False,
    ):
        yield from self._stream_impl(
            messages=messages,
            images_b64=images_b64,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
        )

    # ── shared streaming implementation ──────────────────────────────────────
    def _stream_impl(
        self,
        messages: list[dict],
        images_b64: list[str] | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        enable_thinking: bool,
    ):
        """Drive model.generate in a background thread, yielding token chunks
        as they're produced via TextIteratorStreamer."""
        import threading

        from transformers import TextIteratorStreamer

        images = self._decode_images(images_b64)
        inputs = self._build_inputs(messages, images=images, enable_thinking=enable_thinking)
        input_len = int(inputs["input_ids"].shape[-1])

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            streamer=streamer,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        output_text_parts: list[str] = []
        try:
            for chunk in streamer:
                if not chunk:
                    continue
                output_text_parts.append(chunk)
                yield {"type": "token", "text": chunk}
        finally:
            thread.join()

        # Approximate output token count from the produced text. Exact counts
        # would need a second tokenize pass; this stays accurate enough for
        # cost reporting and avoids re-tokenizing on every stream.
        full_text = "".join(output_text_parts)
        try:
            output_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
        except Exception:
            output_tokens = max(1, len(full_text) // 4)

        yield {
            "type": "done",
            "input_tokens": input_len,
            "output_tokens": int(output_tokens),
        }


# ── Local entrypoints ────────────────────────────────────────────────────────
@app.local_entrypoint()
def download(force: bool = False):
    """Populate the gemma4-weights Volume.

    Run with:  modal run services/modal_app.py::download
    Force redownload:  modal run services/modal_app.py::download --force
    Idempotent — fast no-op if the model is already present.
    """
    ran = download_model.remote(force=force)
    if ran:
        print("✓ download complete — Volume populated")
    else:
        print("✓ Volume already had the model — nothing to do")


@app.local_entrypoint()
def smoke():
    """Quick end-to-end sanity check. Run with:  modal run services/modal_app.py::smoke"""
    print("── chat() ────────────────────────────────────────")
    result = Gemma4Chat().chat.remote(
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        max_new_tokens=64,
    )
    print(result)

    print("── chat_stream() ─────────────────────────────────")
    for evt in Gemma4Chat().chat_stream.remote_gen(
        messages=[{"role": "user", "content": "Count to 5."}],
        max_new_tokens=64,
    ):
        print(evt)
