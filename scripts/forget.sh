#!/usr/bin/env bash
# FR-O5: wipe all local Sair state. Run by the owner only.
#
# Lists the targets (resolved by hub.settings, the same source the hub uses), asks y/N, stops
# the hub (its PID file, or its LaunchAgent), then calls hub.db.forget and prints only how many
# paths were removed. Files are deleted, never opened or printed.
#
# SAIR_FORGET_ROOT (optional): refuse to run unless every target resolves inside this dir.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

PY=(uv run --quiet python)

# 1. Resolve and list the targets (and enforce SAIR_FORGET_ROOT) before asking anything.
"${PY[@]}" - <<'PYEOF'
import os
import sys
from pathlib import Path

from hub.settings import get_settings

s = get_settings()
db = s.db_path
targets = [
    db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm"),
    s.tmp_dir, s.audio_dir, s.audio_cache_dir, s.checkpoint_dir, s.log_dir,
]
root = os.environ.get("SAIR_FORGET_ROOT")
if root:
    r = Path(root).expanduser().resolve()
    bad = [t for t in targets if not t.resolve().is_relative_to(r)]
    if bad:
        for t in bad:
            print(f"refusing: {t} is outside SAIR_FORGET_ROOT", file=sys.stderr)
        sys.exit(2)
print("This will permanently delete:")
for t in targets:
    print(f"  [{'exists' if t.exists() else 'absent'}]  {t}")
print("  (mission audio is the audio dir; model-server logs are the files under the log dir;")
print("   Ollama's own logs live outside this repo)")
PYEOF

read -r -p "Wipe all of the above? [y/N] " ans || ans=""
case "$ans" in
  y|Y) ;;
  *) echo "Aborted."; exit 0 ;;
esac

# 2. Stop the hub (only the hub) so it cannot recreate the DB or audio mid-wipe.
LOG_DIR="${SAIR_LOG_DIR:-logs}"
LOG_DIR="${LOG_DIR/#\~/$HOME}"
pidfile="$LOG_DIR/hub.pid"
if [ -f "$pidfile" ]; then
  pid="$(cat "$pidfile")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null \
      && ps -p "$pid" -o command= | grep -q "hub"; then
    kill "$pid"
    for _ in $(seq 1 100); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
    echo "hub: stopped (pid $pid)"
  else
    echo "hub: not running"
  fi
  rm -f "$pidfile"
fi
# The LaunchAgent hub writes its logs to this repo's logs/; only touch it for that log dir.
if [ "$(cd "$LOG_DIR" 2>/dev/null && pwd -P)" = "$(cd "$REPO/logs" 2>/dev/null && pwd -P)" ] \
    && command -v launchctl >/dev/null \
    && launchctl print "gui/$(id -u)/com.sair.hub" >/dev/null 2>&1; then
  launchctl bootout "gui/$(id -u)/com.sair.hub" 2>/dev/null || true
  echo "hub: LaunchAgent stopped (run 'make install-agents' to start it again)"
fi

# 3. Wipe through the tested code path (hub.db.forget). Prints a count only.
"${PY[@]}" -c '
from hub.db import forget
from hub.settings import get_settings
print(f"removed {len(forget(get_settings()))} paths")
'
echo "Done."
