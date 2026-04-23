#!/usr/bin/env python
"""run.py — bootstrap and launch the entire Gemma-4 chat stack in one shot.

What it does (in order):
  1. Verifies Python ≥ 3.11
  2. Creates .venv/ if missing
  3. Installs dependencies (parsed from pyproject.toml) into the venv
  4. Verifies Modal CLI auth and deploys services/modal_app.py if not already deployed
  5. Starts the FastAPI backend (uvicorn)
  6. Waits for /health to respond
  7. Starts the Streamlit frontend
  8. Opens the browser
  9. Forwards both processes' stdout/stderr with [backend]/[frontend] prefixes
 10. Cleanly tears everything down on Ctrl+C

Usage:
    python run.py                   # default: setup + deploy-if-missing + run
    python run.py --deploy          # force redeploy the Modal app
    python run.py --skip-deploy     # never deploy (assume already deployed)
    python run.py --setup-only      # install deps and exit
    python run.py --no-browser      # don't auto-open browser
    python run.py --backend-port 9000 --frontend-port 9501
"""
from __future__ import annotations

import argparse
import atexit
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
IS_WINDOWS = platform.system() == "Windows"


# ── pretty printing ──────────────────────────────────────────────────────────
def step(msg: str) -> None:
    print(f"\n\033[1;36m▶ {msg}\033[0m", flush=True)


def info(msg: str) -> None:
    print(f"  \033[90m{msg}\033[0m", flush=True)


def warn(msg: str) -> None:
    print(f"  \033[33m! {msg}\033[0m", flush=True)


def ok(msg: str) -> None:
    print(f"  \033[32m✓ {msg}\033[0m", flush=True)


def fatal(msg: str, code: int = 1) -> None:
    print(f"\n\033[1;31m✗ {msg}\033[0m", file=sys.stderr, flush=True)
    sys.exit(code)


# ── venv helpers ─────────────────────────────────────────────────────────────
def venv_python() -> Path:
    if IS_WINDOWS:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def venv_bin(name: str) -> Path:
    suffix = ".exe" if IS_WINDOWS else ""
    folder = "Scripts" if IS_WINDOWS else "bin"
    return VENV_DIR / folder / f"{name}{suffix}"


def ensure_venv() -> Path:
    py = venv_python()
    if py.exists():
        info(f"venv exists at {VENV_DIR}")
        return py
    step(f"Creating virtualenv at {VENV_DIR}")
    venv.EnvBuilder(with_pip=True, upgrade_deps=False).create(str(VENV_DIR))
    if not py.exists():
        fatal(f"venv creation failed; expected {py}")
    ok("virtualenv created")
    return py


def install_deps(py: Path, force: bool) -> None:
    """Install project dependencies parsed from pyproject.toml into the venv."""
    marker = VENV_DIR / ".deps-installed"
    if marker.exists() and not force:
        info("dependencies already installed (delete .venv/.deps-installed to reinstall)")
        return

    step("Installing dependencies")
    with open(ROOT / "pyproject.toml", "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]

    subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    subprocess.check_call([str(py), "-m", "pip", "install", "--quiet", *deps])
    marker.write_text("ok\n", encoding="utf-8")
    ok(f"installed {len(deps)} packages")


# ── Modal helpers ────────────────────────────────────────────────────────────
def modal_cli() -> Path:
    candidate = venv_bin("modal")
    if candidate.exists():
        return candidate
    found = shutil.which("modal")
    if found:
        return Path(found)
    fatal("`modal` CLI not found in venv or PATH (did pip install fail?)")


def modal_authenticated(modal_bin: Path) -> bool:
    """Modal writes credentials to ~/.modal.toml on `modal setup`.
    The env-var pair MODAL_TOKEN_ID / MODAL_TOKEN_SECRET also works."""
    if (Path.home() / ".modal.toml").exists():
        return True
    if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
        return True
    return False


def modal_app_deployed(modal_bin: Path, app_name: str) -> bool:
    try:
        out = subprocess.check_output(
            [str(modal_bin), "app", "list"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        warn(f"`modal app list` failed: {getattr(e, 'output', '').strip() or e}")
        return False
    return any(app_name in line for line in out.splitlines())


def deploy_modal(modal_bin: Path) -> None:
    step("Deploying Modal app (services/modal_app.py)")
    subprocess.check_call(
        [str(modal_bin), "deploy", "services/modal_app.py"],
        cwd=str(ROOT),
    )
    ok("Modal app deployed")


def ensure_model_weights(modal_bin: Path, force: bool) -> None:
    """Run the in-project downloader. Idempotent — fast no-op when present."""
    marker = VENV_DIR / ".model-downloaded"
    if marker.exists() and not force:
        info("model weights marker present (delete .venv/.model-downloaded to recheck)")
        return

    step("Ensuring Gemma-4-31B weights are in the gemma4-weights Volume")
    info("(idempotent — first run takes 30-60 min; subsequent runs are fast)")
    cmd = [str(modal_bin), "run", "services/modal_app.py::download"]
    if force:
        cmd += ["--force"]
    subprocess.check_call(cmd, cwd=str(ROOT))
    marker.write_text("ok\n", encoding="utf-8")
    ok("model weights ready")


# ── child process management ─────────────────────────────────────────────────
_children: list[subprocess.Popen] = []
_shutting_down = False


def spawn(name: str, cmd: list[str], env: dict | None = None) -> subprocess.Popen:
    info(f"spawn [{name}]: {' '.join(cmd)}")
    creationflags = 0
    if IS_WINDOWS:
        # Allow us to send CTRL_BREAK_EVENT to terminate the group.
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env or os.environ.copy(),
        creationflags=creationflags,
    )
    _children.append(proc)
    threading.Thread(target=_pump, args=(name, proc), daemon=True).start()
    return proc


def _pump(name: str, proc: subprocess.Popen) -> None:
    color = {"backend": "32", "frontend": "35"}.get(name, "37")
    prefix = f"\033[1;{color}m[{name}]\033[0m "
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(prefix + line)
        sys.stdout.flush()


def kill_all() -> None:
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    for p in _children:
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
    deadline = time.time() + 5
    for p in _children:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except Exception:
                pass


atexit.register(kill_all)


def wait_healthy(url: str, timeout: float) -> bool:
    info(f"waiting for {url} (≤ {int(timeout)}s)")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap and launch the full Gemma-4 chat stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--deploy", action="store_true", help="Force redeploy the Modal app")
    g.add_argument("--skip-deploy", action="store_true", help="Skip Modal deploy entirely")
    parser.add_argument("--setup-only", action="store_true", help="Install deps then exit")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser")
    parser.add_argument("--reinstall", action="store_true", help="Reinstall deps even if marker exists")
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Force re-run the model downloader (idempotent on the Volume)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the model-weights ensure step entirely",
    )
    parser.add_argument("--backend-port", default="8000")
    parser.add_argument("--frontend-port", default="8501")
    parser.add_argument(
        "--app-name",
        default=os.getenv("MODAL_APP_NAME", "gemma4-chat"),
        help="Modal app name to check/deploy",
    )
    args = parser.parse_args()

    if sys.version_info < (3, 11):
        fatal(f"Python ≥ 3.11 required, got {sys.version.split()[0]}")
    info(f"Python {sys.version.split()[0]} on {platform.system()}")

    py = ensure_venv()
    install_deps(py, force=args.reinstall)

    if args.setup_only:
        ok("Setup complete (--setup-only).")
        return

    modal_bin = modal_cli()

    if args.skip_deploy:
        info("skipping Modal deploy step (--skip-deploy)")
    else:
        if not modal_authenticated(modal_bin):
            fatal(
                "Modal is not authenticated.\n"
                f"  Run:  {modal_bin} setup\n"
                "  Then create the HuggingFace secret:\n"
                f"  {modal_bin} secret create huggingface-secret HF_TOKEN=hf_xxxx"
            )
        ok("Modal authenticated")

        already = modal_app_deployed(modal_bin, args.app_name)
        if args.deploy or not already:
            if not already:
                warn(f"Modal app '{args.app_name}' not deployed — deploying now")
            deploy_modal(modal_bin)
        else:
            ok(f"Modal app '{args.app_name}' already deployed")

        if args.skip_download:
            info("skipping model-weights check (--skip-download)")
        else:
            ensure_model_weights(modal_bin, force=args.download_model)

    env = os.environ.copy()
    env["BACKEND_URL"] = f"http://localhost:{args.backend_port}"
    env["MODAL_APP_NAME"] = args.app_name
    # Ensure backend can find sibling packages when launched from ROOT
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    step("Starting FastAPI backend")
    spawn(
        "backend",
        [
            str(py), "-m", "uvicorn", "backend.main:app",
            "--host", "0.0.0.0",
            "--port", args.backend_port,
        ],
        env=env,
    )

    if not wait_healthy(f"http://localhost:{args.backend_port}/health", timeout=30):
        fatal("Backend did not become healthy within 30s — check the [backend] logs above")
    ok("backend healthy")

    step("Starting Streamlit frontend")
    spawn(
        "frontend",
        [
            str(py), "-m", "streamlit", "run", "frontend/app.py",
            "--server.port", args.frontend_port,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        env=env,
    )

    frontend_url = f"http://localhost:{args.frontend_port}"
    if not args.no_browser:
        time.sleep(2.5)
        try:
            webbrowser.open(frontend_url)
        except Exception:
            pass

    step(f"All services running — open {frontend_url}")
    print("  Press Ctrl+C to stop everything.\n", flush=True)

    try:
        while True:
            for p in _children:
                rc = p.poll()
                if rc is not None:
                    fatal(f"a child process exited with code {rc} — see logs above")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n\033[1;33m⏹  interrupted — shutting down\033[0m", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        kill_all()
