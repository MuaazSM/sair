"""P0 end-to-end latency spike (PRD 18.4, NFR-L2): mock journal pipeline on a synthetic 60 s note.

    uv run python spikes/e2e.py [--note spikes/out/notes/en_60s.webm] [--runs 5]

decode (ffmpeg -> 16 kHz mono) -> VAD gate -> transcribe (Gemma 4 E2B, Ollama /api/chat, temp 0)
-> stand-in writer (stock Qwen3.5-4B, llama-server /completion + draft GBNF)
-> stand-in check (span substring only) -> ONE forced regeneration (always) -> check -> done.
Run 1 is cold for Gemma (unloaded first with keep_alive 0) and has llama-server's prompt cache off.
Prints per-step latency only (no transcript or entry text).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402
import transcribe  # noqa: E402
import vad  # noqa: E402
import writer  # noqa: E402

VAD_DB, MIN_VOICED_S = -45.0, 1.0
GROUP, MISSION = "Mom and Dad", "Listen for the koel near the temple. Is it there again?"


def check(content: str, transcript: str) -> int:
    """Stand-in for PRD 8 check 1 only: count facts whose span is a normalised substring.
    Returns -1 when the output is not valid JSON (e.g. cut at n_predict) -> template path."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return -1
    t = c.norm_text(transcript)
    return sum(c.norm_text(f["span"]) in t for f in obj.get("facts", []))


def run_once(note: Path, grammar: str, cl: httpx.Client, cold: bool) -> dict[str, float]:
    if cold:
        cl.post(
            f"{c.OLLAMA_HOST}/api/generate",
            json={"model": c.OLLAMA_MODEL, "keep_alive": 0},
            timeout=60,
        )
        time.sleep(2)
    st: dict[str, float] = {}
    t0 = time.perf_counter()
    pcm = c.decode_16k(note)
    st["decode"] = time.perf_counter() - t0
    t = time.perf_counter()
    voiced = vad.voiced_seconds(vad.frame_db(pcm), VAD_DB)
    st["vad"] = time.perf_counter() - t
    if voiced < MIN_VOICED_S:
        raise SystemExit("vad rejected the note")
    t = time.perf_counter()
    text, _ = transcribe.gemma_transcribe(pcm, cl)
    st["transcribe"] = time.perf_counter() - t
    prompt = writer.build_prompt(GROUP, MISSION, text)
    t = time.perf_counter()
    j = cl.post(
        f"{c.WRITER_URL}/completion",
        timeout=120,
        json={
            "prompt": prompt,
            "grammar": grammar,
            "n_predict": 400,
            "temperature": 0,
            "cache_prompt": not cold,
        },
    ).json()
    st["write1"] = time.perf_counter() - t
    st["write1_tok"] = j["timings"]["predicted_n"]
    t = time.perf_counter()
    ok1 = check(j["content"], text)
    st["check1"] = time.perf_counter() - t
    # forced regeneration: the real path re-prompts with failure codes; we always do it once
    regen = prompt.replace(
        "<|im_end|>\n<|im_start|>assistant",
        "\nPrevious attempt failed checks: SPAN_NOT_IN_TRANSCRIPT. "
        "Copy spans exactly.<|im_end|>\n<|im_start|>assistant",
        1,
    )
    t = time.perf_counter()
    j2 = cl.post(
        f"{c.WRITER_URL}/completion",
        timeout=120,
        json={
            "prompt": regen,
            "grammar": grammar,
            "n_predict": 400,
            "temperature": 0,
            "cache_prompt": True,
        },
    ).json()
    st["write2"] = time.perf_counter() - t
    st["write2_tok"] = j2["timings"]["predicted_n"]
    t = time.perf_counter()
    ok2 = check(j2["content"], text)
    st["check2"] = time.perf_counter() - t
    st["total"] = time.perf_counter() - t0
    st["ok1"], st["ok2"] = float(ok1), float(ok2)
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default=str(c.OUT / "notes" / "en_60s.webm"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--grammar", default=str(Path(writer.__file__).with_name("entry.draft.gbnf")))
    a = ap.parse_args()
    note = Path(a.note)
    if not note.exists():
        c.encode(note.with_suffix(".wav"), note, "webm")
    grammar = Path(a.grammar).read_text()
    print(
        f"grammar={Path(a.grammar).name} note={note.name} dur={len(c.decode_16k(note)) / 16000:.1f}s"
    )
    keys = ["decode", "vad", "transcribe", "write1", "check1", "write2", "check2", "total"]
    print(
        f"{'run':5}"
        + "".join(f"{k:>11}" for k in keys)
        + f"{'tok1':>6}{'tok2':>6}{'span1':>6}{'span2':>6}"
    )
    totals = []
    with httpx.Client() as cl:
        for i in range(a.runs):
            st = run_once(note, grammar, cl, cold=(i == 0))
            totals.append(st["total"])
            print(
                f"{('cold' if i == 0 else f'warm{i}'):5}"
                + "".join(f"{st[k]:11.3f}" for k in keys)
                + f"{int(st['write1_tok']):6d}{int(st['write2_tok']):6d}"
                + f"{int(st['ok1']):6d}{int(st['ok2']):6d}",
                flush=True,
            )
    arr = np.array(totals)
    print(
        f"total_s all={np.round(arr, 2).tolist()} p50={np.percentile(arr, 50):.2f} "
        f"p95={np.percentile(arr, 95):.2f} (numpy linear, n={len(arr)})"
    )
    w = arr[1:]
    if len(w):
        print(
            f"warm only: p50={np.percentile(w, 50):.2f} p95={np.percentile(w, 95):.2f} n={len(w)}"
        )
    print("span1/span2 = facts passing the substring check; -1 = invalid JSON (cut at n_predict)")
    print("NFR-L2 target: p50 <= 15 s, p95 <= 30 s")


if __name__ == "__main__":
    main()
