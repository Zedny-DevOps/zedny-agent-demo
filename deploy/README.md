# Deploying the Zedny Chat Stack to an Ubuntu/Debian VM

This guide turns `python run.py` into a **systemd-managed background service**
on a Linux VM, manageable with `start` / `stop` / `restart` / `status` / `logs`
commands.

What you'll have at the end:
- A `zedny` system user owning the app at `/opt/zedny-agent-demo`
- A `zedny-chat.service` systemd unit that auto-restarts on failure
- A `manage.sh` wrapper for one-line ops
- Logs streamed into `journalctl` with `[backend]` / `[frontend]` prefixes

---

## 0. Prerequisites

- An Ubuntu 22.04 / 24.04 (or recent Debian) VM with SSH access
- At least 1 vCPU / 2 GB RAM (the heavy lifting is on Modal — the VM only
  hosts the FastAPI + Streamlit processes)
- Ports `22` (SSH) reachable, plus `8000` (backend) and `8501` (frontend)
  reachable from wherever you'll consume the chat (or behind a reverse proxy)
- An HuggingFace access token with read access to `google/gemma-4-31B-it`
- A Modal account with billing enabled for A100-80GB GPUs

---

## 1. Transfer the project to the VM (from your dev machine)

From PowerShell on Windows (using rsync via WSL or Git Bash):

```bash
rsync -avz --delete \
  --exclude .venv --exclude __pycache__ --exclude .git \
  "/d/Zedny INC/01-production/services/zedny-agent-demo/" \
  user@VM_IP:/tmp/zedny-agent-demo/
```

Or with `scp`:

```bash
scp -r "D:/Zedny INC/01-production/services/zedny-agent-demo" user@VM_IP:/tmp/
```

Or `git clone` if you've pushed to a remote — clone into `/tmp/zedny-agent-demo`.

Then on the VM, move it into place:

```bash
sudo mv /tmp/zedny-agent-demo /opt/
```

(Don't worry about ownership — the installer fixes it.)

---

## 2. Run the installer (once, as root)

```bash
sudo bash /opt/zedny-agent-demo/deploy/install.sh
```

This is idempotent. It will:
1. Verify Debian/Ubuntu
2. Install Python ≥ 3.11 (using deadsnakes PPA on Ubuntu 22.04 if needed)
3. Create the `zedny` system user
4. `chown -R zedny:zedny /opt/zedny-agent-demo`
5. Create the venv via `run.py --setup-only`
6. Install the systemd unit (but **does not start it yet**)
7. Open ports 8000 and 8501 in `ufw` (if active)
8. Print the remaining manual steps

---

## 3. One-time interactive bootstrap (as the `zedny` user)

These three steps need a human (browser-based auth, dashboard token edit, and
a long-running download) so they aren't part of `install.sh`.

### 3a. Authenticate Modal

```bash
sudo -u zedny -H /opt/zedny-agent-demo/.venv/bin/modal setup
```

A URL is printed — open it in any browser and log in. The token is written
to `/home/zedny/.modal.toml`.

### 3b. Confirm the HuggingFace secret

The Modal app needs a secret called `huggingface-secret` containing
`HF_TOKEN=hf_...` with read access to `google/gemma-4-31B-it`.

If you've never created it:

```bash
sudo -u zedny -H /opt/zedny-agent-demo/.venv/bin/modal secret create \
  huggingface-secret HF_TOKEN=hf_yourRealToken
```

If it already exists with the wrong value, the easiest fix is the dashboard:

> https://modal.com/secrets → `huggingface-secret` → edit `HF_TOKEN` → save

### 3c. Download the model weights into the Modal Volume

This populates the `gemma4-weights` Volume (~60 GB, takes 30–60 minutes the
first time, idempotent on re-runs):

```bash
sudo -u zedny -H bash -c \
  "cd /opt/zedny-agent-demo && .venv/bin/python run.py --download-model --skip-deploy"
```

### 3d. Deploy the Modal inference app

```bash
sudo -u zedny -H bash -c \
  "cd /opt/zedny-agent-demo && .venv/bin/modal deploy services/modal_app.py"
```

You should see something like `✓ Created class Gemma4Chat`.

---

## 4. Start the service

```bash
sudo /opt/zedny-agent-demo/deploy/manage.sh start
sudo /opt/zedny-agent-demo/deploy/manage.sh status
```

Expected: `Active: active (running)`.

Smoke-test the backend:

```bash
sudo /opt/zedny-agent-demo/deploy/manage.sh health
# → {"status":"ok"}
#   ✓ backend is healthy
```

Smoke-test inference end-to-end:

```bash
curl -sS -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}]}' | jq
```

Expected: a JSON `ChatResponse` with non-zero `input_tokens` / `output_tokens` / `cost_usd`.

Open the Streamlit UI from your browser:

```
http://VM_IP:8501
```

---

## 5. Day-to-day management

```bash
# Status / control
sudo /opt/zedny-agent-demo/deploy/manage.sh start
sudo /opt/zedny-agent-demo/deploy/manage.sh stop
sudo /opt/zedny-agent-demo/deploy/manage.sh restart
sudo /opt/zedny-agent-demo/deploy/manage.sh status

# Auto-start on boot (run once if you want it)
sudo /opt/zedny-agent-demo/deploy/manage.sh enable
sudo /opt/zedny-agent-demo/deploy/manage.sh disable

# Logs
sudo /opt/zedny-agent-demo/deploy/manage.sh logs            # follow live
sudo /opt/zedny-agent-demo/deploy/manage.sh logs-tail       # last 200 lines
sudo /opt/zedny-agent-demo/deploy/manage.sh logs-backend    # only [backend] lines
sudo /opt/zedny-agent-demo/deploy/manage.sh logs-frontend   # only [frontend] lines

# Quick health check
sudo /opt/zedny-agent-demo/deploy/manage.sh health
```

Underneath, this is just `systemctl` and `journalctl` — you can use those
directly if you prefer:

```bash
sudo systemctl status zedny-chat
sudo journalctl -u zedny-chat -f --output=cat
```

---

## 6. Updating the code

From your dev machine, rsync the new code over (same command as step 1),
then on the VM:

```bash
sudo chown -R zedny:zedny /opt/zedny-agent-demo
sudo /opt/zedny-agent-demo/deploy/manage.sh restart
sudo /opt/zedny-agent-demo/deploy/manage.sh status
```

If you changed `pyproject.toml` (added a dep), force a reinstall first:

```bash
sudo -u zedny rm /opt/zedny-agent-demo/.venv/.deps-installed
sudo -u zedny -H bash -c \
  "cd /opt/zedny-agent-demo && .venv/bin/python run.py --setup-only"
sudo /opt/zedny-agent-demo/deploy/manage.sh restart
```

If you changed `services/modal_app.py`, redeploy the Modal app:

```bash
sudo -u zedny -H bash -c \
  "cd /opt/zedny-agent-demo && .venv/bin/modal deploy services/modal_app.py"
```

(No service restart needed — the FastAPI side just calls into the deployed
class by name; Modal serves the new version on next call.)

---

## 7. (Optional) nginx reverse proxy with TLS

If you want to expose the UI on a real domain over HTTPS rather than
`http://VM_IP:8501`, drop this into `/etc/nginx/sites-available/zedny-chat`:

```nginx
server {
    listen 80;
    server_name chat.example.com;

    # API
    location /api/ {
        proxy_pass         http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_buffering    off;            # important for SSE streaming
        proxy_read_timeout 600s;
    }

    # Streamlit UI (and its websocket)
    location / {
        proxy_pass         http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 600s;
    }
}
```

Then:

```bash
sudo ln -s /etc/nginx/sites-available/zedny-chat /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo ufw deny 8000/tcp && sudo ufw deny 8501/tcp   # only nginx is public
sudo ufw allow 'Nginx Full'

# TLS via Let's Encrypt:
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d chat.example.com
```

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `manage.sh status` shows `failed` | systemd unit can't start `run.py` | `manage.sh logs-tail` to see the error |
| Health endpoint returns 502 | Modal app not deployed, or HF secret holds placeholder token | Re-do steps 3b–3d |
| First chat request hangs ~60 s | Cold start of the Modal A100 container | Normal once, then warm for `scaledown_window=120` s |
| `Permission denied` reading `~/.modal.toml` | Modal auth was done as a different user | `sudo -u zedny ... modal setup` |
| `8501` reachable from VM but not externally | Firewall blocking | `sudo ufw allow 8501/tcp` (or use the nginx setup above) |
| `journalctl` shows `ImportError: tomllib` | Python < 3.11 was picked | Re-run `install.sh`; verify `head /etc/systemd/system/zedny-chat.service` shows the right Python path |

For a deeper crash dump:

```bash
sudo journalctl -u zedny-chat -n 500 --no-pager --output=cat
```
