"""P0 transcription spike (PRD 18.1, FR-J2): Gemma 4 E2B audio via Ollama, temperature 0.

    uv run python spikes/transcribe.py FILE... [--chunk-s N] [--ref] [--quiet]

API used (Ollama 0.35.1): POST {OLLAMA_HOST}/api/chat with the 16 kHz mono WAV, base64-encoded,
in the message's `images` array; `think: false`; options.temperature = 0.
--chunk-s N  split on VAD silences into <= N s pieces, no overlap, joined in order with one space.
--ref        compare with <file>.txt next to the input (WER / CER / coverage).
--quiet      don't print transcripts (only metrics).
Only for synthetic or the owner's own test recordings; prints to this terminal only.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402
import vad  # noqa: E402

SR = 16000
PROMPT = (
    "Transcribe this audio verbatim, in the original language and script. "
    "Output only the transcript, with no commentary."
)


def wav_bytes(pcm: np.ndarray) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, pcm, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def gemma_transcribe(
    pcm: np.ndarray, client: httpx.Client, timeout: float = 120.0
) -> tuple[str, dict]:
    body = {
        "model": c.OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": 1024},
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [base64.b64encode(wav_bytes(pcm)).decode()],
            }
        ],
    }
    r = client.post(f"{c.OLLAMA_HOST}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    meta = {
        k: j.get(k)
        for k in (
            "prompt_eval_count",
            "eval_count",
            "done_reason",
            "load_duration",
            "total_duration",
        )
    }
    return j["message"]["content"].strip(), meta


def transcribe_file(path: str | Path, chunk_s: float, thr: float, client: httpx.Client) -> dict:
    pcm = c.decode_16k(path)
    dur = len(pcm) / SR
    spans = vad.split_on_silence(pcm, chunk_s, thr) if chunk_s > 0 else [(0, len(pcm))]
    t0 = time.perf_counter()
    texts, metas = [], []
    for a, b in spans:
        t, m = gemma_transcribe(pcm[a:b], client)
        texts.append(t)
        metas.append(m)
    return {
        "dur": dur,
        "latency": time.perf_counter() - t0,
        "text": " ".join(texts),
        "chunks": [(round(a / SR, 2), round(b / SR, 2)) for a, b in spans],
        "meta": metas,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--chunk-s", type=float, default=0.0)
    ap.add_argument("--vad-db", type=float, default=-45.0)
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    with httpx.Client() as cl:
        for f in a.files:
            r = transcribe_file(f, a.chunk_s, a.vad_db, cl)
            line = (
                f"{Path(f).name:26} dur={r['dur']:5.1f}s lat={r['latency']:5.2f}s "
                f"chunks={len(r['chunks'])} out_tok={sum(m['eval_count'] or 0 for m in r['meta'])}"
            )
            ref = Path(f).with_suffix(".txt")
            if a.ref and ref.exists():
                rt = ref.read_text(encoding="utf-8")
                line += (
                    f" WER={c.wer(rt, r['text']):.3f} CER={c.cer(rt, r['text']):.3f} "
                    f"cov={c.coverage(rt, r['text']):.2f} skelCER={c.skel_cer(rt, r['text']):.3f} "
                    f"ref_w={len(c.norm_text(rt).split())} hyp_w={len(c.norm_text(r['text']).split())}"
                )
            print(line, flush=True)
            if not a.quiet:
                print(f"    {r['text']}", flush=True)


if __name__ == "__main__":
    main()
