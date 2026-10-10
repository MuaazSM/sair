#!/usr/bin/env bash
# Wipe all local Sair state. Lists targets and asks for confirmation. Run by the owner only.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${SAIR_DATA_DIR:-$HOME/.sair}"
DB_PATH="${SAIR_DB_PATH:-$DATA_DIR/sair.db}"
TMP_DIR="${SAIR_TMP_DIR:-$DATA_DIR/tmp}"
AUDIO_CACHE="${SAIR_AUDIO_CACHE_DIR:-$DATA_DIR/cache}"
LOG_DIR="${SAIR_LOG_DIR:-./logs}"
CKPT_DIR="${SAIR_CHECKPOINT_DIR:-$DATA_DIR/checkpoints}"

targets=(
  "$DB_PATH" "$DB_PATH-wal" "$DB_PATH-shm"
  "$TMP_DIR"
  "$AUDIO_CACHE"
  "$CKPT_DIR"
  "$LOG_DIR"
)

echo "This will permanently delete:"
for t in "${targets[@]}"; do
  [ -e "$t" ] && echo "  [exists]  $t" || echo "  [absent]  $t"
done
echo "  (model-server logs are the files under $LOG_DIR; Ollama's own logs live outside this repo)"
read -r -p "Wipe all of the above? [y/N] " ans
case "$ans" in
  y|Y) ;;
  *) echo "Aborted."; exit 0 ;;
esac

for t in "${targets[@]}"; do
  [ -e "$t" ] && rm -rf -- "$t" && echo "removed $t"
done
echo "Done."
