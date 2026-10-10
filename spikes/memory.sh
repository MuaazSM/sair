#!/usr/bin/env bash
# P0 memory spike (PRD 18.3, NFR-R1): Ollama (gemma4:e2b loaded) + llama-server (Qwen GGUF)
# + Kokoro in its own process, all co-resident. Prints ollama ps, vm_stat-derived memory,
# the GPU wired limit and per-process RSS. Assumes ollama serve and llama-server are already
# running on 127.0.0.1 (scripts/up.sh or the spike commands); starts and stops Kokoro itself.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "== load gemma4:e2b (tiny request, keep_alive 30m)"
curl -fsS 127.0.0.1:11434/api/generate -d '{"model":"gemma4:e2b","prompt":"ok","stream":false,"think":false,"keep_alive":"30m","options":{"num_predict":1}}' >/dev/null
echo "== warm llama-server (1 token)"
curl -fsS 127.0.0.1:8091/completion -d '{"prompt":"ok","n_predict":1}' >/dev/null
echo "== start kokoro holder"
KOUT="$(mktemp)"
uv run python spikes/kokoro_hold.py 300 >"$KOUT" 2>/dev/null & KPID=$!
for _ in $(seq 1 90); do grep -q rendered "$KOUT" && break; sleep 1; done
cat "$KOUT"; rm -f "$KOUT"
sleep 3

echo "== ollama ps"; ollama ps
echo "== GPU wired limit"
LIMIT_MB="$(sysctl -n iogpu.wired_limit_mb)"
RAM_MB=$(( $(sysctl -n hw.memsize) / 1048576 ))
echo "iogpu.wired_limit_mb=$LIMIT_MB  hw.memsize_mb=$RAM_MB"
if [ "$LIMIT_MB" = 0 ]; then
  # 0 = macOS default. Metal reports it as recommendedMaxWorkingSetSize; llama-server logs it at start.
  echo "(0 = OS default; Metal recommendedMaxWorkingSetSize follows)"
fi
printf 'import Metal\nlet d = MTLCreateSystemDefaultDevice()!\nprint("metal recommendedMaxWorkingSetSize_mb=\\(d.recommendedMaxWorkingSetSize / 1048576) currentAllocatedSize_mb=\\(d.currentAllocatedSize / 1048576)")\n' > /tmp/sair_metal_ws.swift
swift /tmp/sair_metal_ws.swift 2>/dev/null || echo "(swift Metal query unavailable)"
rm -f /tmp/sair_metal_ws.swift
echo "== vm_stat"
vm_stat | python3 -c '
import re,sys
t=sys.stdin.read(); ps=int(re.search(r"page size of (\d+)",t).group(1))
g=lambda k:int(re.search(k+r":\s+(\d+)",t).group(1))*ps/2**30
act,ina,spec,wired,comp,free=g("Pages active"),g("Pages inactive"),g("Pages speculative"),g("Pages wired down"),g("Pages occupied by compressor"),g("Pages free")
print(f"page={ps} free={free:.2f}G active={act:.2f}G inactive={ina:.2f}G speculative={spec:.2f}G wired={wired:.2f}G compressor={comp:.2f}G")
print(f"used(app+wired+compressed)={act+wired+comp:.2f}G")'
echo "== per-process RSS (MB)"
ps -axo rss=,pid=,comm= | awk '/ollama|llama-server|python/ {printf "%8.1f MB  %6s  %s\n", $1/1024, $2, $3}' | sort -rn | head -12
for p in $(pgrep -f 'spikes/kokoro_hold.py'); do echo "kokoro_hold pid=$p rss_mb=$(( $(ps -o rss= -p "$p") / 1024 ))"; done
echo "== phys_footprint (incl. compressed; RSS undercounts under memory pressure)"
for p in $( (pgrep -x llama-server; pgrep -f 'kokoro_hold.py') | sort -u); do
  footprint -p "$p" 2>/dev/null | grep -E 'phys_footprint(_peak)?:' | tr -s ' ' | tr '\n' ' ' | sed "s/^/pid $p $(ps -o comm= -p $p | sed "s:.*/::"): /"; echo
done
echo "== top 8 RSS on the machine (context: other apps share the 18 GB)"
ps -axo rss=,comm= | sort -rn | head -8 | awk '{n=$2; sub(".*/","",n); printf "%8.1f MB  %s\n", $1/1024, n}'
echo "== ioreg GPU in-use (if exposed)"
ioreg -r -d 1 -w 0 -c AGXAccelerator 2>/dev/null | grep -o '"In use system memory"=[0-9]*\|"Alloc system memory"=[0-9]*\|"In use system memory (driver)"=[0-9]*' | head -3 || true
kill "$KPID" 2>/dev/null || true; pkill -f 'spikes/kokoro_hold.py' 2>/dev/null || true
echo "== kokoro holder stopped"
