"""P0 TTS spike (PRD 18.5): render one mission in English and Hindi with Kokoro, offline.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python spikes/tts.py [--guard]

--guard installs a socket guard that refuses every non-loopback connect and DNS lookup,
so any hidden download attempt fails loudly and is counted.
Outputs: spikes/out/tts_<lang>_<voice>.mp3 (gitignored).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

MISSION_EN = "Last time you heard a koel near the temple. Is it there again?"
MISSION_HI = "पिछली बार आपने मंदिर के पास कोयल सुनी थी। क्या वह आज फिर है?"
EN_VOICES = [
    "af_heart",
    "af_bella",
    "am_michael",
    "bf_emma",
    "bm_george",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
]
HI_VOICES = ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"]
SR = 24000

_PIPES: dict[str, object] = {}
_MODEL: object = None


def load(lang: str) -> object:
    """Kokoro pipeline for `lang` ('a' US English, 'b' UK English, 'h' Hindi), local files only."""
    global _MODEL
    from kokoro import KModel, KPipeline

    if _MODEL is None:
        _MODEL = KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(c.KOKORO_DIR / "config.json"),
            model=str(c.KOKORO_DIR / "kokoro-v1_0.pth"),
        ).eval()
    if lang not in _PIPES:
        _PIPES[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=_MODEL)
    return _PIPES[lang]


def render(text: str, voice: str, lang: str, speed: float = 1.0):  # -> np.ndarray
    import numpy as np

    pipe = load(lang)
    vpath = str(c.KOKORO_DIR / "voices" / f"{voice}.pt")
    parts = [
        r.audio.numpy()
        for r in pipe(text, voice=vpath, speed=speed)  # type: ignore[operator]
        if r.audio is not None
    ]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def main() -> None:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--guard", action="store_true")
    args = ap.parse_args()
    print(
        f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} "
        f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')} guard={args.guard}"
    )
    if args.guard:
        c.install_loopback_guard()
    warnings.filterwarnings("ignore")
    c.OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    load("a")
    print(f"kokoro model load: {time.perf_counter() - t0:.2f}s from {c.KOKORO_DIR}")
    jobs = [("en", "a", v, MISSION_EN) for v in EN_VOICES]
    jobs += [("en", "b", v, MISSION_EN) for v in ("bf_emma", "bm_george")]
    jobs = [j for j in jobs if not (j[1] == "a" and j[2].startswith("b"))]
    jobs += [("hi", "h", v, MISSION_HI) for v in HI_VOICES]
    print(f"{'file':34} {'dur_s':>6} {'rms_dbfs':>8} {'render_s':>8}  status")
    for tag, lang, voice, text in jobs:
        name = f"tts_{tag}_{voice}"
        try:
            t = time.perf_counter()
            pcm = render(text, voice, lang)
            dt = time.perf_counter() - t
            rms = float(np.sqrt(np.mean(pcm**2))) if pcm.size else 0.0
            db = 20 * np.log10(rms) if rms > 0 else -120.0
            wav = c.OUT / f"{name}.wav"
            c.write_wav(wav, pcm, SR)
            c.encode(wav, c.OUT / f"{name}.mp3", "mp3")
            status = "ok" if pcm.size > SR and db > -40 else "SUSPECT"
            print(f"{name + '.mp3':34} {pcm.size / SR:6.2f} {db:8.1f} {dt:8.2f}  {status}")
        except Exception as e:  # noqa: BLE001
            print(f"{name + '.mp3':34} {'-':>6} {'-':>8} {'-':>8}  FAIL {type(e).__name__}: {e}")
    print(f"blocked non-loopback attempts: {len(c.BLOCKED)} {sorted(set(c.BLOCKED))}")


if __name__ == "__main__":
    main()
