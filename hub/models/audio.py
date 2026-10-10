"""Audio ingest helpers (FR-J1, FR-J2): decode, bounds, VAD gate, silence-aware chunking.

ffmpeg stderr is never captured or logged (NFR-P4). All DSP is deterministic.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from hub.models.errors import Permanent
from hub.schemas import ReasonCode
from hub.settings import Settings, get_settings

__all__ = [
    "DEFAULT_VAD_THRESHOLD_DB",
    "FRAME_S",
    "MIN_VOICED_S",
    "SAMPLE_RATE",
    "VadStats",
    "bounds_check",
    "chunk_files",
    "decode_to_wav16k",
    "duration_s",
    "split_on_silence",
    "vad_gate",
]

SAMPLE_RATE = 16000
FRAME_S = 0.030
FRAME = int(FRAME_S * SAMPLE_RATE)  # 480 samples
DEFAULT_VAD_THRESHOLD_DB = -45.0
MIN_VOICED_S = 1.0
DECODE_TIMEOUT_S = 60
_FLOOR_DB = -120.0


@dataclass(frozen=True)
class VadStats:
    total_s: float
    voiced_s: float
    frames: int
    voiced_frames: int
    threshold_db: float
    max_db: float
    passed: bool


def decode_to_wav16k(src: Path, dst_dir: Path) -> Path:
    """Decode any ffmpeg-readable audio to 16 kHz mono s16 WAV in ``dst_dir``."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{src.stem}.16k.wav"
    cmd = [
        "ffmpeg", "-nostdin", "-v", "quiet", "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst),
    ]  # fmt: skip
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=DECODE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        dst.unlink(missing_ok=True)
        raise Permanent(ReasonCode.AUDIO_UNDECODABLE, "audio") from None
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size <= 44:
        dst.unlink(missing_ok=True)
        raise Permanent(ReasonCode.AUDIO_UNDECODABLE, "audio")
    return dst


def duration_s(path: Path) -> float:
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError):
        raise Permanent(ReasonCode.AUDIO_UNDECODABLE, "audio") from None
    return float(info.frames) / float(info.samplerate)


def bounds_check(duration: float, settings: Settings | None = None) -> None:
    """FR-J1: AUDIO_MIN_S <= duration <= AUDIO_MAX_S (both inclusive)."""
    s = settings or get_settings()
    if duration < s.audio_min_s:
        raise Permanent(ReasonCode.AUDIO_TOO_SHORT, "audio")
    if duration > s.audio_max_s:
        raise Permanent(ReasonCode.AUDIO_TOO_LONG, "audio")


def _read_mono(path: Path) -> np.ndarray:
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, OSError):
        raise Permanent(ReasonCode.AUDIO_UNDECODABLE, "audio") from None
    if sr != SAMPLE_RATE:
        raise Permanent(ReasonCode.AUDIO_UNDECODABLE, "audio")
    return np.asarray(data.mean(axis=1), dtype=np.float32)


def _frame_db(pcm: np.ndarray) -> np.ndarray:
    n = len(pcm) // FRAME
    if n == 0:
        return np.full(1, _FLOOR_DB)
    frames = pcm[: n * FRAME].astype(np.float64).reshape(n, FRAME)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return np.asarray(20.0 * np.log10(np.maximum(rms, 1e-6)))


def vad_gate(
    wav: Path,
    settings: Settings | None = None,
    *,
    threshold_db: float | None = None,
    min_voiced_s: float = MIN_VOICED_S,
) -> VadStats:
    """FR-J2 energy gate: 30 ms frames with RMS >= threshold count as voiced.

    Raises ``Permanent(AUDIO_SILENT)`` if less than ``min_voiced_s`` is voiced.
    """
    s = settings or get_settings()
    thr = (
        threshold_db
        if threshold_db is not None
        else (
            s.vad_energy_threshold
            if s.vad_energy_threshold is not None
            else DEFAULT_VAD_THRESHOLD_DB
        )
    )
    pcm = _read_mono(wav)
    db = _frame_db(pcm)
    voiced = int((db >= thr).sum())
    voiced_s = voiced * FRAME_S
    stats = VadStats(
        total_s=len(pcm) / SAMPLE_RATE,
        voiced_s=voiced_s,
        frames=len(db),
        voiced_frames=voiced,
        threshold_db=float(thr),
        max_db=float(db.max()),
        passed=voiced_s >= min_voiced_s - 1e-9,
    )
    if not stats.passed:
        raise Permanent(ReasonCode.AUDIO_SILENT, "audio")
    return stats


def _plan_cuts(pcm: np.ndarray, chunk_s: float) -> list[tuple[int, int]]:
    """Contiguous, non-overlapping [start, end) sample ranges, each <= chunk_s.

    Each cut sits at the quietest 30 ms frame (latest on ties) within the last 30% of the
    window; deterministic.
    """
    max_n = int(chunk_s * SAMPLE_RATE)
    if max_n < FRAME * 4:
        raise ValueError("chunk_s too small")
    db = _frame_db(pcm)
    out: list[tuple[int, int]] = []
    start = 0
    while len(pcm) - start > max_n:
        lo_f = max((start + int(max_n * 0.7)) // FRAME, start // FRAME + 1)
        hi_f = (start + max_n) // FRAME - 1  # frame fully inside the window
        hi_f = min(hi_f, len(db) - 1)
        if hi_f < lo_f:
            cut = start + max_n
        else:
            window = db[lo_f : hi_f + 1]
            idx = int(len(window) - 1 - np.argmin(window[::-1]))  # latest minimum
            cut = (lo_f + idx) * FRAME + FRAME // 2
        out.append((start, cut))
        start = cut
    out.append((start, len(pcm)))
    return out


def split_on_silence(wav: Path, chunk_s: float, dst_dir: Path | None = None) -> list[Path]:
    """Write ordered chunk WAVs (no overlap) and return their paths.

    Prefer :func:`chunk_files`, which deletes the files for you. If ``dst_dir`` is None a new
    private temp dir next to ``wav`` is created and the caller must remove ``chunks[0].parent``.
    """
    pcm = _read_mono(wav)
    spans = _plan_cuts(pcm, chunk_s)
    out_dir = (
        dst_dir if dst_dir is not None else Path(tempfile.mkdtemp(prefix="chunks-", dir=wav.parent))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, (a, b) in enumerate(spans):
        p = out_dir / f"{wav.stem}.c{i:03d}.wav"
        sf.write(str(p), pcm[a:b], SAMPLE_RATE, subtype="PCM_16")
        paths.append(p)
    return paths


@contextlib.contextmanager
def chunk_files(wav: Path, chunk_s: float) -> Iterator[list[Path]]:
    """Context manager: yields chunk paths, deletes its temp dir on exit (FR-J6)."""
    tmp = Path(tempfile.mkdtemp(prefix="chunks-", dir=wav.parent))
    try:
        yield split_on_silence(wav, chunk_s, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
