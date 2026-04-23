#!/usr/bin/env bash
# manage.sh — thin wrapper around systemctl / journalctl for the zedny-chat service.
#
# Usage:
#   sudo ./manage.sh <command>
#
# Commands:
#   start           Start the service
#   stop            Stop the service
#   restart         Restart the service (graceful: stop then start)
#   status          Show systemd status (units, last 10 log lines)
#   enable          Enable auto-start on boot
#   disable         Disable auto-start on boot
#   logs            Tail journal output (Ctrl+C to detach)
#   logs-tail       Show the last 200 journal lines and exit
#   logs-backend    Tail only [backend]-prefixed lines
#   logs-frontend   Tail only [frontend]-prefixed lines
#   health          curl http://localhost:8000/health (sanity check)
set -euo pipefail

SERVICE="zedny-chat"

require_root() {
  if [[ $EUID -ne 0 ]]; then
    echo "✗ This command needs root. Re-run with: sudo $0 $*" >&2
    exit 1
  fi
}

usage() {
  cat <<EOF
usage: $0 <command>

  start | stop | restart | status
  enable | disable
  logs | logs-tail | logs-backend | logs-frontend
  health
EOF
  exit "${1:-1}"
}

cmd="${1:-}"
case "$cmd" in
  start|stop|restart|enable|disable)
    require_root "$cmd"
    systemctl "$cmd" "$SERVICE"
    systemctl --no-pager status "$SERVICE" | head -n 5 || true
    ;;
  status)
    systemctl --no-pager status "$SERVICE"
    ;;
  logs)
    journalctl -u "$SERVICE" -f --output=cat
    ;;
  logs-tail)
    journalctl -u "$SERVICE" -n 200 --no-pager --output=cat
    ;;
  logs-backend)
    journalctl -u "$SERVICE" -f --output=cat | grep --line-buffered '\[backend\]'
    ;;
  logs-frontend)
    journalctl -u "$SERVICE" -f --output=cat | grep --line-buffered '\[frontend\]'
    ;;
  health)
    if curl -fsS --max-time 3 http://localhost:8000/health; then
      echo
      echo "✓ backend is healthy"
    else
      echo "✗ backend not reachable on :8000" >&2
      exit 1
    fi
    ;;
  -h|--help|help|"")
    usage 0
    ;;
  *)
    echo "✗ unknown command: $cmd" >&2
    usage 1
    ;;
esac
