#!/usr/bin/env bash
# Start the local stack: ollama, llama-server (if a model exists), ntfy (if installed), hub.
# Logs and PID files go to ./logs. Everything binds to loopback.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

HUB_PORT="${SAIR_PORT:-8787}"
WRITER_MODEL="${WRITER_MODEL_PATH:-models/sair-writer-q4_k_m.gguf}"
[ -f "$WRITER_MODEL" ] || WRITER_MODEL="models/qwen3.5-4b-q4_k_m.gguf"

start() { # name, command...
  local name="$1"; shift
  if [ -f "logs/$name.pid" ] && kill -0 "$(cat "logs/$name.pid")" 2>/dev/null; then
    echo "$name: already running"; return
  fi
  nohup "$@" >"logs/$name.log" 2>&1 &
  echo $! >"logs/$name.pid"
  echo "$name: started (pid $(cat "logs/$name.pid"))"
}

if command -v ollama >/dev/null; then
  OLLAMA_DEBUG=0 OLLAMA_HOST=127.0.0.1:11434 start ollama ollama serve
else
  echo "ollama: not installed, skipped"
fi

if command -v llama-server >/dev/null && [ -f "$WRITER_MODEL" ]; then
  start llama-server llama-server -m "$WRITER_MODEL" --host 127.0.0.1 --port 8091 -c 4096
else
  echo "llama-server: skipped (binary or model file missing)"
fi

if command -v ntfy >/dev/null; then
  start ntfy ntfy serve --listen-http 127.0.0.1:8090
else
  echo "ntfy: not installed, skipped"
fi

start hub uv run uvicorn hub.app:app --host 127.0.0.1 --port "$HUB_PORT"
