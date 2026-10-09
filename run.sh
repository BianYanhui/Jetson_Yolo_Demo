#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Broken LAN proxy breaks pip/curl; keep runtime clean.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing venv. Create it first (see README)." >&2
  exit 1
fi

# Prefer video0 (UGREEN USB camera capture node)
CAMERA="${CAMERA:-0}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

exec "$ROOT/.venv/bin/python" -m app.stream_server \
  --host "$HOST" \
  --port "$PORT" \
  --camera "$CAMERA" \
  --weights "$ROOT/weights/yolov6n.pt" \
  --width 1280 \
  --height 720 \
  --jpeg-quality 70 \
  "$@"
