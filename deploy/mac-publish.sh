#!/usr/bin/env bash
# Run the reporter on this Mac and mirror its dashboard to Firebase Hosting.
#
#   bash deploy/mac-publish.sh          # wrapper container up, keep-awake, reporter on
#                                       # 127.0.0.1:8788, publish to Firebase after every cycle
#   bash deploy/mac-publish.sh stop     # stop the reporter and keep-awake (wrapper keeps running)
#   bash deploy/mac-publish.sh log      # tail the reporter log
#   bash deploy/mac-publish.sh publish  # export + deploy the current briefs once, now
#
# The reporter runs from this checkout's venv (not the reporter container) so
# the `firebase` CLI signed in on this Mac can upload the export. Only the
# Codex wrapper runs in Docker. The local dashboard is unauthenticated and
# bound to localhost; the public copy on Firebase is read-only.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
LOG="$HOME/.hedge-fund/reporter.log"
PID="$HOME/.hedge-fund/reporter.pid"
mkdir -p "$HOME/.hedge-fund"

case "${1:-}" in
  stop)
    if [ -f "$PID" ] && kill "$(cat "$PID")" 2>/dev/null; then echo "reporter stopped"; else echo "reporter was not running"; fi
    rm -f "$PID"
    pkill -x caffeinate && echo "keep-awake stopped" || true
    exit 0 ;;
  log) exec tail -n 50 -f "$LOG" ;;
esac

command -v firebase >/dev/null || { echo "npm install -g firebase-tools, then firebase login"; exit 1; }
[ -f .env ] || { echo "deploy/.env missing (copy .env.example)"; exit 1; }
[ -f codex-home/auth.json ] || { echo "copy ~/.codex/auth.json to deploy/codex-home/"; exit 1; }

set -a; source .env; set +a
if [ "${1:-}" != "publish" ]; then
  # Codex rotates its refresh token whenever the CLI on this Mac refreshes, which
  # strands the copy the container holds (401 "workspace routing discovery").
  # Start from whichever copy is newer.
  if [ ~/.codex/auth.json -nt codex-home/auth.json ]; then
    cp -p ~/.codex/auth.json codex-home/auth.json && echo "codex-home/auth.json refreshed from ~/.codex"
  fi
  # Compose first: it insists on the .env values the host reporter overrides below.
  docker compose up -d codex-wrapper
fi
export CODEX_WRAPPER_URL="http://127.0.0.1:${CODEX_WRAPPER_PORT:-8020}/v1"
export CODEX_WRAPPER_API_KEY="$PROXY_API_KEY"
export REPORTER_LLM_TIMEOUT="${CODEX_TIMEOUT:-900}"
export REPORTER_PASSWORD=""                      # localhost only: no sign-in
export REPORTER_PUBLISH_DIR="$ROOT/deploy/firebase/site"
export REPORTER_PUBLISH_CMD="firebase deploy --only hosting --non-interactive"
export HEDGE_FUND_DATA_SOURCE="${HEDGE_FUND_DATA_SOURCE:-yfinance}"

if [ "${1:-}" = "publish" ]; then
  exec "$ROOT/venv/bin/python" -m hedge_fund.run reporter --publish
fi

pgrep -x caffeinate >/dev/null || { nohup caffeinate -s >/dev/null 2>&1 & }
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "reporter already running (pid $(cat "$PID"))"
else
  nohup "$ROOT/venv/bin/python" -m hedge_fund.run reporter --host 127.0.0.1 --port "${REPORTER_PORT:-8788}" >> "$LOG" 2>&1 &
  echo $! > "$PID"
  echo "reporter started (pid $!), log: $LOG"
fi
echo "local dashboard: http://127.0.0.1:${REPORTER_PORT:-8788}"
echo "public mirror:   https://$(python3 -c "import json;print(json.load(open('firebase/.firebaserc'))['projects']['default'])").web.app"
