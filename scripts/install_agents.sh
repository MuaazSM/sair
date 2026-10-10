#!/usr/bin/env bash
# Install / remove the Sair user LaunchAgents (NFR-R2).
#   scripts/install_agents.sh              render and install, then print status
#   scripts/install_agents.sh --dry-run [DIR]  render only (to DIR, or stdout)
#   scripts/install_agents.sh --uninstall  boot out and remove
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${SAIR_DATA_DIR:-$HOME/.sair}"
DATA="${DATA/#\~/$HOME}"
SRC="$REPO/ops/launchd"
DEST="$HOME/Library/LaunchAgents"
UID_N="$(id -u)"
LABELS=(ollama llama-server ntfy hub keepawake)

WRITER="$REPO/models/sair-writer-q4_k_m.gguf"
[ -f "$WRITER" ] || WRITER="$REPO/models/qwen3.5-4b-q4_k_m.gguf"

render() { # label -> stdout
  sed -e "s#__REPO__#$REPO#g" -e "s#__DATA__#$DATA#g" -e "s#__WRITER__#$WRITER#g" \
    "$SRC/com.sair.$1.plist"
}

mode="install"
out=""
case "${1:-}" in
  --dry-run) mode="dry"; out="${2:-}" ;;
  --uninstall) mode="uninstall" ;;
  "") ;;
  *) echo "usage: $0 [--dry-run [DIR] | --uninstall]" >&2; exit 2 ;;
esac

if [ "$mode" = "dry" ]; then
  for l in "${LABELS[@]}"; do
    if [ -n "$out" ]; then
      mkdir -p "$out"; render "$l" >"$out/com.sair.$l.plist"; echo "rendered $out/com.sair.$l.plist"
    else
      echo "<!-- com.sair.$l -->"; render "$l"
    fi
  done
  exit 0
fi

if [ "$mode" = "uninstall" ]; then
  for l in "${LABELS[@]}"; do
    launchctl bootout "gui/$UID_N/com.sair.$l" 2>/dev/null || true
    rm -f "$DEST/com.sair.$l.plist"
    echo "removed com.sair.$l"
  done
  exit 0
fi

mkdir -p "$DEST" "$REPO/logs" "$DATA/ntfy"
for l in "${LABELS[@]}"; do
  dst="$DEST/com.sair.$l.plist"
  render "$l" >"$dst"
  plutil -lint "$dst" >/dev/null
  launchctl bootout "gui/$UID_N/com.sair.$l" 2>/dev/null || true
  launchctl bootstrap "gui/$UID_N" "$dst"
  launchctl kickstart "gui/$UID_N/com.sair.$l" 2>/dev/null || true
done
for l in "${LABELS[@]}"; do
  pid="$(launchctl print "gui/$UID_N/com.sair.$l" 2>/dev/null | awk '/^[[:space:]]*pid =/ {print $3}')"
  echo "com.sair.$l: ${pid:+running (pid $pid)}${pid:-not running}"
done
