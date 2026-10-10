"""P0 VAD spike (FR-J2): energy gate over 30 ms frames; RMS stats per file.

    uv run python spikes/vad.py spikes/out/notes/*.wav [--thresholds -50,-45,-40,-35]

Units: dBFS of frame RMS on 16 kHz mono float PCM (full scale = 1.0).
Gate rule proposed for the hub: speech_ratio = fraction of 30 ms frames with
RMS >= threshold; accept iff speech_ratio >= MIN_SPEECH_RATIO (and >= MIN_SPEECH_S of voiced audio).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

SR = 16000
FRAME = int(0.030 * SR)


def frame_db(pcm: np.ndarray) -> np.ndarray:
    n = len(pcm) // FRAME
    if n == 0:
        return np.full(1, -120.0)
    f = pcm[: n * FRAME].reshape(n, FRAME)
    rms = np.sqrt(np.mean(f**2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-6))


def voiced_seconds(db: np.ndarray, thr: float) -> float:
    return float((db >= thr).sum()) * FRAME / SR


def split_on_silence(
    pcm: np.ndarray, max_s: float, thr: float, min_gap_s: float = 0.25
) -> list[tuple[int, int]]:
    """Split into contiguous, non-overlapping [start, end) sample ranges <= max_s each,
    cutting at the middle of the longest silence run available before the limit.
    Falls back to a hard cut at max_s if no silence is found."""
    db = frame_db(pcm)
    silent = db < thr
    # silence runs as (start_frame, end_frame)
    runs, i = [], 0
    while i < len(silent):
        if silent[i]:
            j = i
            while j < len(silent) and silent[j]:
                j += 1
            if (j - i) * FRAME / SR >= min_gap_s:
                runs.append((i, j))
            i = j
        else:
            i += 1
    cuts_s = [((a + b) / 2) * FRAME / SR for a, b in runs]
    total = len(pcm) / SR
    out, start = [], 0.0
    while total - start > max_s:
        cands = [x for x in cuts_s if start + 1.0 < x <= start + max_s]
        cut = max(cands) if cands else start + max_s
        out.append((int(start * SR), int(cut * SR)))
        start = cut
    out.append((int(start * SR), len(pcm)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--thresholds", default="-60,-55,-50,-45,-40,-35")
    a = ap.parse_args()
    thrs = [float(x) for x in a.thresholds.split(",")]
    head = " ".join(f"{'>=' + str(int(t)):>6}" for t in thrs)
    print(f"{'file':28} {'dur':>5} {'min':>6} {'p10':>6} {'med':>6} {'p90':>6} {'max':>6} | {head}")
    for f in a.files:
        pcm = c.decode_16k(f)
        db = frame_db(pcm)
        p = np.percentile(db, [10, 50, 90])
        fr = " ".join(f"{(db >= t).mean():6.2f}" for t in thrs)
        print(
            f"{Path(f).name:28} {len(pcm) / SR:5.1f} {db.min():6.1f} {p[0]:6.1f} {p[1]:6.1f} "
            f"{p[2]:6.1f} {db.max():6.1f} | {fr}"
        )


if __name__ == "__main__":
    main()
