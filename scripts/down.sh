#!/usr/bin/env bash
# Stop everything started by scripts/up.sh (PID files in ./logs).
set -euo pipefail
cd "$(dirname "$0")/.."
for name in hub ntfy llama-server ollama; do
  pidfile="logs/$name.pid"
  if [ -f "$pidfile" ]; then
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "$name: stopped (pid $pid)"
    else
      echo "$name: not running"
    fi
    rm -f "$pidfile"
  else
    echo "$name: no pid file"
  fi
done
