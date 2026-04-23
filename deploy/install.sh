#!/usr/bin/env bash
# install.sh — one-shot server-side installer for the Zedny chat stack.
#
# Run as root on a fresh Ubuntu/Debian VM, AFTER the project tree has been
# placed at /opt/zedny-agent-demo (e.g. via rsync from your dev machine):
#
#   sudo bash /opt/zedny-agent-demo/deploy/install.sh
#
# Idempotent — safe to re-run.
#
# What it does:
#   1. Verifies Debian/Ubuntu + root.
#   2. Installs system packages (python3.11+, venv, pip, ufw, curl, git).
#   3. Creates the `zedny` system user.
#   4. Hands ownership of /opt/zedny-agent-demo to zedny.
#   5. Bootstraps the venv via `run.py --setup-only`.
#   6. Installs and reloads the systemd unit (does not start it).
#   7. Opens UFW for ports 8000 and 8501 if UFW is active.
#   8. Prints the remaining manual steps (Modal auth, weights download, deploy).
set -euo pipefail

APP_DIR="/opt/zedny-agent-demo"
SERVICE_NAME="zedny-chat"
SERVICE_USER="zedny"
SERVICE_HOME="/home/${SERVICE_USER}"
UNIT_SRC="${APP_DIR}/deploy/${SERVICE_NAME}.service"
UNIT_DST="/etc/systemd/system/${SERVICE_NAME}.service"

# ── pretty printing ──────────────────────────────────────────────────────────
c_cyan='\033[1;36m'; c_grey='\033[90m'; c_yellow='\033[33m'; c_green='\033[32m'; c_red='\033[1;31m'; c_off='\033[0m'
step() { printf "\n${c_cyan}▶ %s${c_off}\n" "$*"; }
info() { printf "  ${c_grey}%s${c_off}\n" "$*"; }
warn() { printf "  ${c_yellow}! %s${c_off}\n" "$*"; }
ok()   { printf "  ${c_green}✓ %s${c_off}\n" "$*"; }
fatal(){ printf "\n${c_red}✗ %s${c_off}\n" "$*" >&2; exit 1; }

# ── 1. Pre-flight ────────────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] || fatal "Run as root (try: sudo bash $0)"
[[ -f /etc/os-release ]] || fatal "Cannot detect OS — /etc/os-release missing"
. /etc/os-release
case "${ID:-}${ID_LIKE:-}" in
  *debian*|*ubuntu*) ok "OS: ${PRETTY_NAME}" ;;
  *) fatal "Unsupported OS '${PRETTY_NAME}' — this installer targets Debian/Ubuntu" ;;
esac
[[ -d "$APP_DIR" ]] || fatal "$APP_DIR not found — copy the project there first (see deploy/README.md)"
[[ -f "$APP_DIR/run.py" ]] || fatal "$APP_DIR/run.py missing — incomplete copy?"
[[ -f "$UNIT_SRC" ]] || fatal "$UNIT_SRC missing — incomplete copy?"

# ── 2. System packages ───────────────────────────────────────────────────────
step "Installing base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y -qq
apt-get install -y -qq curl git ufw software-properties-common ca-certificates

# Pick a Python ≥ 3.11. Try the system one first; fall back to deadsnakes on Ubuntu 22.04.
PYTHON_BIN=""
for cand in python3.13 python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    ver=$("$cand" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || true)
    if dpkg --compare-versions "$ver" ge "3.11" 2>/dev/null; then
      PYTHON_BIN="$(command -v "$cand")"
      info "Found ${cand} (${ver}) at ${PYTHON_BIN}"
      break
    fi
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  warn "No system Python ≥ 3.11 found — adding deadsnakes PPA and installing python3.11"
  if [[ "${ID:-}" == "ubuntu" ]]; then
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -y -qq
    apt-get install -y -qq python3.11 python3.11-venv python3.11-distutils
    PYTHON_BIN="/usr/bin/python3.11"
  else
    fatal "Debian without Python ≥ 3.11 — install one manually and re-run"
  fi
fi

# Always install matching venv/pip support for whichever python we chose.
PY_PKG="$(basename "$PYTHON_BIN")"
apt-get install -y -qq "${PY_PKG}-venv" || true
apt-get install -y -qq python3-pip || true
ok "Python ready: ${PYTHON_BIN}"

# ── 3. Service user ──────────────────────────────────────────────────────────
step "Ensuring service user '${SERVICE_USER}' exists"
if id -u "$SERVICE_USER" >/dev/null 2>&1; then
  info "user '${SERVICE_USER}' already exists"
else
  useradd --system --create-home --home-dir "$SERVICE_HOME" \
          --shell /usr/sbin/nologin "$SERVICE_USER"
  ok "created user '${SERVICE_USER}' with home ${SERVICE_HOME}"
fi
[[ -d "$SERVICE_HOME" ]] || install -d -o "$SERVICE_USER" -g "$SERVICE_USER" "$SERVICE_HOME"

# ── 4. Ownership ─────────────────────────────────────────────────────────────
step "Setting ownership of ${APP_DIR} to ${SERVICE_USER}"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "$APP_DIR"
chmod +x "$APP_DIR/deploy/manage.sh" 2>/dev/null || true
chmod +x "$APP_DIR/deploy/install.sh" 2>/dev/null || true
ok "ownership set"

# ── 5. Bootstrap venv via run.py ─────────────────────────────────────────────
step "Bootstrapping venv + dependencies (sudo -u ${SERVICE_USER} python run.py --setup-only)"
sudo -u "$SERVICE_USER" -H bash -c "cd '$APP_DIR' && '$PYTHON_BIN' run.py --setup-only"
ok "venv ready at ${APP_DIR}/.venv"

# ── 6. Install systemd unit ──────────────────────────────────────────────────
step "Installing systemd unit ${UNIT_DST}"
# Patch the ExecStart to use the python we picked, in case it's not /usr/bin/python3
install -m 0644 "$UNIT_SRC" "$UNIT_DST"
if [[ "$PYTHON_BIN" != "/usr/bin/python3" ]]; then
  sed -i "s|^ExecStart=/usr/bin/python3 |ExecStart=${PYTHON_BIN} |" "$UNIT_DST"
  info "ExecStart pinned to ${PYTHON_BIN}"
fi
systemctl daemon-reload
ok "unit installed (not started yet)"

# ── 7. Firewall ──────────────────────────────────────────────────────────────
step "Configuring firewall (ufw)"
if ufw status 2>/dev/null | grep -q "Status: active"; then
  ufw allow 8000/tcp >/dev/null && info "allowed 8000/tcp"
  ufw allow 8501/tcp >/dev/null && info "allowed 8501/tcp"
  ok "ufw rules added"
else
  info "ufw not active — skipping (open ports 8000/8501 manually if firewalled)"
fi

# ── 8. Next steps ────────────────────────────────────────────────────────────
cat <<EOF

${c_green}✓ Install complete.${c_off}

${c_cyan}Next steps (one-time, must be done as the '${SERVICE_USER}' user):${c_off}

  1. Authenticate with Modal (opens a browser link in your terminal):
       sudo -u ${SERVICE_USER} -H ${PYTHON_BIN} -m modal setup

  2. Make sure the 'huggingface-secret' Modal secret holds a real HF token
     with read access to google/gemma-4-31B-it. Easiest path:
       https://modal.com/secrets  →  huggingface-secret  →  set HF_TOKEN

  3. Populate the gemma4-weights Volume (long-running, ~30-60 min the first time):
       sudo -u ${SERVICE_USER} -H bash -c \\
         "cd ${APP_DIR} && ${PYTHON_BIN} run.py --download-model --skip-deploy"

  4. Deploy the Modal inference app (one-time):
       sudo -u ${SERVICE_USER} -H bash -c \\
         "cd ${APP_DIR} && .venv/bin/modal deploy services/modal_app.py"

  5. Start the service and confirm it's healthy:
       sudo ${APP_DIR}/deploy/manage.sh start
       sudo ${APP_DIR}/deploy/manage.sh status
       curl http://localhost:8000/health

  6. (Optional) auto-start on boot:
       sudo ${APP_DIR}/deploy/manage.sh enable

${c_grey}Tail logs:        sudo ${APP_DIR}/deploy/manage.sh logs${c_off}
${c_grey}Stop / restart:   sudo ${APP_DIR}/deploy/manage.sh stop | restart${c_off}
EOF
