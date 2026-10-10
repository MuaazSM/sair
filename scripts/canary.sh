#!/usr/bin/env bash
# Privacy canary against the running local model servers (PRD §11.1; NFR-P3, NFR-P4).
#
# 1. Eval settings via env vars only (no env file is read), data in a scratch dir, never ~/.sair.
# 2. scripts/canary_driver.py pushes a marker transcript through the journal + mission graphs
#    with the REAL writer/planner (stub transcriber), capturing Sentry (fake transport), ntfy
#    (mock transport), outbound sockets and the eval DB, and writes a counts-only summary.
# 3. This script greps the driver's stdout/stderr, our log files and the Ollama/llama-server log
#    files for the marker. It prints file names, counts and PASS/FAIL only, never log content.
# 4. meta.last_canary_result (+ last_canary_date) is written to the eval DB (SAIR_DB_PATH).
set -uo pipefail
cd "$(dirname "$0")/.."

MARKER="${CANARY_MARKER:-zebra-ultramarine-4471}"
SCRATCH="${SAIR_CANARY_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/sair-canary.XXXXXX")}"
case "$(cd "$SCRATCH" 2>/dev/null && pwd -P)" in
  "$HOME/.sair"|"$HOME/.sair/"*) echo "canary: refusing to use ~/.sair"; exit 2 ;;
esac
mkdir -p "$SCRATCH"

export SAIR_ENV=eval
export SAIR_DATA_DIR="$SCRATCH/data"
export SAIR_DB_PATH="${SAIR_DB_PATH:-$SCRATCH/data/sair-eval.db}"
export SAIR_TMP_DIR="$SCRATCH/data/tmp"
export SAIR_AUDIO_DIR="$SCRATCH/data/audio"
export SAIR_AUDIO_CACHE_DIR="$SCRATCH/data/cache"
export SAIR_CHECKPOINT_DIR="$SCRATCH/data/checkpoints"
export SAIR_LOG_DIR="$SCRATCH/logs"
export SENTRY_DSN=""            # the driver installs a fake transport on a loopback DSN
export CANARY_MARKER="$MARKER"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
case "$SAIR_DB_PATH" in
  "$HOME/.sair"|"$HOME/.sair/"*) echo "canary: refusing a DB under ~/.sair"; exit 2 ;;
esac

up() { curl -s -o /dev/null -m 3 -w '%{http_code}' "$1" 2>/dev/null | grep -q '^200$'; }
OLLAMA_UP=no;  up "http://127.0.0.1:11434/api/version" && OLLAMA_UP=yes
LLAMA_UP=no;   up "http://127.0.0.1:8091/health" && LLAMA_UP=yes
echo "canary: scratch=$SCRATCH"
echo "canary: ollama=$OLLAMA_UP llama-server=$LLAMA_UP"

# Log files to grep: ours (./logs, the scratch log dir), the model servers' (./logs from
# up.sh/launchd, ~/.ollama/logs) and whatever files the running servers hold as stdout/stderr.
LOGS_LIST="$SCRATCH/logfiles.txt"
: >"$LOGS_LIST"
for f in logs/*.log logs/*.out logs/*.err "$HOME"/.ollama/logs/*.log "$SAIR_LOG_DIR"/*; do
  [ -f "$f" ] && echo "$f" >>"$LOGS_LIST"
done
for pid in $(pgrep -f 'ollama serve|ollama runner|llama-server' 2>/dev/null); do
  lsof -a -p "$pid" -d 1,2 -Fn 2>/dev/null | sed -n 's/^n//p' | while read -r f; do
    [ -f "$f" ] && echo "$f" >>"$LOGS_LIST"
  done
done

OUT="$SCRATCH/driver.out"; ERR="$SCRATCH/driver.err"; SUMMARY="$SCRATCH/summary.json"
echo "canary: running driver (real writer/planner; this can take a few minutes)"
uv run --quiet python scripts/canary_driver.py run --summary "$SUMMARY" >"$OUT" 2>"$ERR"
DRIVER_RC=$?

for f in "$SCRATCH"/logs/*; do [ -f "$f" ] && echo "$f" >>"$LOGS_LIST"; done
sort -u "$LOGS_LIST" -o "$LOGS_LIST"

FAIL=0
count() { grep -c -F -- "$MARKER" "$1" 2>/dev/null || true; }

n=$(count "$OUT"); echo "stdout (driver): marker_count=$n"; [ "${n:-0}" -eq 0 ] || FAIL=1
n=$(count "$ERR"); echo "stderr (driver): marker_count=$n"; [ "${n:-0}" -eq 0 ] || FAIL=1
echo "driver exit=$DRIVER_RC"; [ "$DRIVER_RC" -eq 0 ] || FAIL=1

NLOG=0
while read -r f; do
  [ -n "$f" ] || continue
  NLOG=$((NLOG + 1))
  n=$(count "$f")
  echo "log $f: marker_count=${n:-0}"
  [ "${n:-0}" -eq 0 ] || FAIL=1
done <"$LOGS_LIST"
echo "log files checked: $NLOG"
[ "$OLLAMA_UP" = yes ] || echo "note: ollama not running; its log was checked as it is on disk"
[ "$LLAMA_UP" = yes ] || echo "note: llama-server not running; writer ran on Gemma (Ollama)"

if [ -f "$SUMMARY" ] && [ "$(count "$SUMMARY")" -eq 0 ]; then
  echo "summary: $(cat "$SUMMARY")"
  uv run --quiet python - "$SUMMARY" <<'PY' || FAIL=1
import json, sys
s = json.load(open(sys.argv[1]))
bad = []
if s["sentry_hits"]: bad.append("sentry")
if s["ntfy_hits"]: bad.append("ntfy")
if s["outbound_non_loopback"]: bad.append("outbound")
if s["db_disallowed"]: bad.append("db:" + ",".join(sorted(s["db_disallowed"])))
if not s["ollama_up"]: bad.append("ollama_down")
if s["walks_done"] < 1: bad.append("no_walk_done")
if not s["sentry_envelopes"]: bad.append("sentry_sink_not_exercised")
if not s["ntfy_requests"]: bad.append("ntfy_sink_not_exercised")
print("summary checks:", "ok" if not bad else "FAILED " + " ".join(bad))
sys.exit(1 if bad else 0)
PY
else
  echo "summary: missing or contains the marker"; FAIL=1
fi

RESULT=passed; [ "$FAIL" -eq 0 ] || RESULT=failed
uv run --quiet python scripts/canary_driver.py write-meta "$RESULT" >/dev/null 2>&1 \
  && echo "meta.last_canary_result=$RESULT written to the eval DB" \
  || { echo "could not write meta"; FAIL=1; }
if [ "$FAIL" -eq 0 ]; then echo "CANARY PASS"; exit 0; fi
echo "CANARY FAIL"; exit 1
