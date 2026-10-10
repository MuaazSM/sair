#!/usr/bin/env bash
# Pre-fetch every model/voice/data file. Idempotent: each step skips if already present.
# Nothing may download at runtime (PRD 11.1). Usage: scripts/models.sh [--with-whisper]
set -euo pipefail
cd "$(dirname "$0")/.."

WITH_WHISPER=0
[ "${1:-}" = "--with-whisper" ] && WITH_WHISPER=1

say() { printf '==> %s\n' "$*"; }

# Run a huggingface_hub CLI command. Credentials (if any) come from .env.training,
# sourced silently in a subshell; nothing from it is printed.
hf_cli() {
  (
    if [ -f ./.env.training ]; then set -a; . ./.env.training; set +a; fi
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    if uv run --with huggingface_hub which hf >/dev/null 2>&1; then
      uv run --with huggingface_hub hf "$@"
    else
      uv run --with huggingface_hub huggingface-cli "$@"
    fi
  )
}

# 1. brew packages
brew_need() {
  if brew list --formula "$1" >/dev/null 2>&1; then say "brew: $1 present"
  else say "brew: installing $1"; brew install "$1"; fi
}
brew_need espeak-ng
brew_need ntfy
[ "$WITH_WHISPER" = 1 ] && brew_need whisper-cpp

# 2. Ollama + gemma4:e2b
if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  say "ollama: starting a temporary 'ollama serve'"
  mkdir -p logs
  OLLAMA_DEBUG=0 nohup ollama serve >logs/ollama-models.log 2>&1 &
  OLLAMA_TMP_PID=$!
  for _ in $(seq 1 30); do
    curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
    sleep 1
  done
fi
if ollama list | awk 'NR>1{print $1}' | grep -qx 'gemma4:e2b'; then
  say "ollama: gemma4:e2b present"
else
  say "ollama: pulling gemma4:e2b"; ollama pull gemma4:e2b
fi
[ -n "${OLLAMA_TMP_PID:-}" ] && { say "ollama: stopping temporary server"; kill "$OLLAMA_TMP_PID" || true; }

# 3. Kokoro-82M weights
mkdir -p models
if [ -f models/kokoro/config.json ] && ls models/kokoro/voices/*.pt >/dev/null 2>&1; then
  say "kokoro: models/kokoro present"
else
  say "kokoro: downloading hexgrad/Kokoro-82M"
  hf_cli download hexgrad/Kokoro-82M --local-dir models/kokoro
fi

# 4. Kokoro python deps + spaCy model (misaki needs en_core_web_sm)
say "tts extra: uv sync --extra tts"
uv sync --all-extras
if uv run python -c "import en_core_web_sm" >/dev/null 2>&1; then
  say "spacy: en_core_web_sm present"
else
  say "spacy: installing en_core_web_sm 3.8.0"
  uv pip install "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
fi
say "note: a later plain 'uv sync' removes en_core_web_sm; use 'uv sync --inexact' or 'uv sync --extra tts' + this script"

# 5. Stock Qwen3.5-4B Q4_K_M GGUF (P0 check, baseline C)
GGUF=models/qwen3.5-4b-q4_k_m.gguf
if [ -f "$GGUF" ]; then
  say "gguf: $GGUF present"
else
  say "gguf: downloading unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf"
  hf_cli download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir models/_gguf_tmp
  mv models/_gguf_tmp/Qwen3.5-4B-Q4_K_M.gguf "$GGUF"
  rm -rf models/_gguf_tmp
fi

# 6. Qwen3.5-4B HF bf16 snapshot (for the P6 merge); verify only when already cached
HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
if ls "$HF_HOME_DIR"/hub/models--Qwen--Qwen3.5-4B/snapshots/*/*.safetensors >/dev/null 2>&1; then
  say "hf: Qwen/Qwen3.5-4B weights present in cache"
else
  say "hf: downloading Qwen/Qwen3.5-4B (bf16)"
  hf_cli download Qwen/Qwen3.5-4B
fi

# 7. llama.cpp checkout + its own venv for convert_hf_to_gguf.py
mkdir -p vendor
if [ -d vendor/llama.cpp/.git ]; then
  say "llama.cpp: vendor/llama.cpp present"
else
  say "llama.cpp: cloning (depth 1)"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp vendor/llama.cpp
fi
if [ -x vendor/llama.cpp/.venv/bin/python ]; then
  say "llama.cpp venv present"
else
  say "llama.cpp: creating venv + installing converter requirements"
  uv venv --python 3.12 vendor/llama.cpp/.venv
  uv pip install --python vendor/llama.cpp/.venv/bin/python \
    -r vendor/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
fi

say "done"
