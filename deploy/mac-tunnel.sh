#!/usr/bin/env bash
# Run the reporter stack on this Mac and publish it through a Cloudflare quick
# tunnel (no account; the URL is random and changes every time this runs).
#
#   bash deploy/mac-tunnel.sh          # start containers, keep-awake, tunnel; print the URL
#   bash deploy/mac-tunnel.sh stop     # stop the tunnel and keep-awake (containers keep running)
#
# Docker Desktop restarts the containers by itself (restart: unless-stopped);
# cloudflared and caffeinate do not survive a reboot, so rerun this after one.
set -euo pipefail
cd "$(dirname "$0")"

LOG="$HOME/.hedge-fund/cloudflared.log"
mkdir -p "$HOME/.hedge-fund"

if [ "${1:-}" = "stop" ]; then
  pkill -f 'cloudflared tunnel --url http://localhost:8788' && echo "tunnel stopped" || echo "tunnel was not running"
  pkill -x caffeinate && echo "keep-awake stopped" || true
  exit 0
fi

command -v cloudflared >/dev/null || { echo "brew install cloudflared first"; exit 1; }
[ -f .env ] || { echo "deploy/.env missing (copy .env.example)"; exit 1; }
[ -f codex-home/auth.json ] || { echo "copy ~/.codex/auth.json to deploy/codex-home/"; exit 1; }

docker compose up -d
pgrep -x caffeinate >/dev/null || { nohup caffeinate -s >/dev/null 2>&1 & }
pkill -f 'cloudflared tunnel --url http://localhost:8788' 2>/dev/null || true
nohup cloudflared tunnel --url "http://localhost:${REPORTER_PORT:-8788}" --no-autoupdate > "$LOG" 2>&1 &

for _ in $(seq 1 30); do
  url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" | head -1 || true)
  [ -n "$url" ] && break
  sleep 1
done
echo "dashboard: ${url:-<tunnel URL not found yet; see $LOG>}"
echo "user: $(grep '^REPORTER_USER=' .env | cut -d= -f2)   password: in deploy/.env (REPORTER_PASSWORD)"
