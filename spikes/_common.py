"""Shared helpers for the P0 spikes (synthetic audio only).

Nothing here logs or prints model output on its own; callers decide what to print.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "spikes" / "out"
KOKORO_DIR = ROOT / "models" / "kokoro"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")
WRITER_URL = os.environ.get("WRITER_URL", "http://127.0.0.1:8091")

# --------------------------------------------------------------------------- network guard

BLOCKED: list[str] = []


def install_loopback_guard() -> None:
    """Refuse any outbound socket connect that is not loopback (records attempts)."""
    real_connect = socket.socket.connect

    def guarded(self: socket.socket, address: object) -> object:
        host = address[0] if isinstance(address, tuple) else str(address)
        if self.family == socket.AF_UNIX or host in ("127.0.0.1", "::1", "localhost"):
            return real_connect(self, address)  # type: ignore[arg-type]
        BLOCKED.append(str(host))
        raise OSError(f"loopback guard: blocked connect to {host}")

    socket.socket.connect = guarded  # type: ignore[method-assign]
    real_gai = socket.getaddrinfo

    def guarded_gai(host: object, *a: object, **k: object) -> object:
        if host not in (None, "127.0.0.1", "::1", "localhost"):
            BLOCKED.append(f"dns:{host}")
            raise OSError(f"loopback guard: blocked DNS for {host}")
        return real_gai(host, *a, **k)  # type: ignore[arg-type]

    socket.getaddrinfo = guarded_gai  # type: ignore[assignment]


# --------------------------------------------------------------------------- audio io


def decode_16k(path: Path | str, sr: int = 16000) -> np.ndarray:
    """ffmpeg -> mono float32 PCM at `sr`."""
    raw = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def write_wav(path: Path | str, pcm: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(str(path), pcm, sr, subtype="PCM_16")


def encode(src: Path | str, dst: Path | str, kind: str) -> None:
    args = {
        "mp3": ["-c:a", "libmp3lame", "-b:a", "64k"],
        "webm": ["-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "16k"],
        "m4a": ["-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "24k"],
    }[kind]
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), *args, str(dst)], check=True
    )


# --------------------------------------------------------------------------- WER / CER


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = s.replace("\u0964", " ").replace("\u0965", " ")  # danda, double danda
    s = re.sub(r"[^\w\s\u0900-\u097F]", " ", s)  # keep Devanagari letters, matras, virama
    return re.sub(r"\s+", " ", s).strip()


def _edit(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = norm_text(ref).split(), norm_text(hyp).split()
    return _edit(r, h) / max(1, len(r))


def cer(ref: str, hyp: str) -> float:
    r, h = list(norm_text(ref).replace(" ", "")), list(norm_text(hyp).replace(" ", ""))
    return _edit(r, h) / max(1, len(r))


def coverage(ref: str, hyp: str) -> float:
    """Share of reference words (in order-insensitive multiset terms) present in hyp."""
    from collections import Counter

    r, h = Counter(norm_text(ref).split()), Counter(norm_text(hyp).split())
    return sum(min(c, h[w]) for w, c in r.items()) / max(1, sum(r.values()))


# --------------------------------------------------------------------------- script-insensitive CER

_DEV = dict(
    zip("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह", "kkggnccjjnttddnttddnpfbbmyrlvsssh", strict=True)
)


def skeleton(s: str) -> str:
    """Crude consonant skeleton shared by Devanagari and Latin-script Hindi (Hinglish).
    Vowels/matras dropped, aspiration dropped, doubled letters collapsed. Spike metric only."""
    s = unicodedata.normalize("NFD", norm_text(s)).replace("़", "")  # drop nukta
    out = []
    for w in s.split():
        if any("ऀ" <= ch <= "ॿ" for ch in w):
            k = "".join(_DEV.get(ch, "n" if ch in "ंँ" else "") for ch in w)
        else:
            w = w.replace("ch", "c").replace("sh", "s").replace("ph", "f")
            w = re.sub(r"(?<=[kgcjtdpb])h", "", w)
            w = w.translate(str.maketrans("wzqx", "vjkk"))
            k = re.sub(r"[aeiou]", "", w)
        out.append(re.sub(r"(.)\1+", r"\1", k))
    return " ".join(x for x in out if x)


def skel_cer(ref: str, hyp: str) -> float:
    r, h = list(skeleton(ref).replace(" ", "")), list(skeleton(hyp).replace(" ", ""))
    return _edit(r, h) / max(1, len(r))
