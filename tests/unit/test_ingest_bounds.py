"""FR-J1: container decode (WebM/Opus, M4A/AAC) and 1 s .. 90 s duration bounds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hub.models import audio
from hub.models.errors import Permanent
from hub.schemas import ReasonCode
from hub.settings import Settings

FX = Path(__file__).parent.parent / "fixtures" / "audio"
S = Settings(_env_file=None)  # type: ignore[call-arg]


def _wav(path: Path, seconds: float) -> Path:
    sf.write(str(path), np.zeros(round(seconds * 16000), np.float32), 16000, subtype="PCM_16")
    return path


@pytest.mark.parametrize("ext", ["webm", "m4a"])
def test_containers_decode_to_16k_mono(tmp_path: Path, ext: str) -> None:
    wav = audio.decode_to_wav16k(FX / f"en_20s.{ext}", tmp_path)
    info = sf.info(str(wav))
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert 23.0 < audio.duration_s(wav) < 23.7


@pytest.mark.parametrize("ext", ["webm", "m4a"])
def test_short_rejected(tmp_path: Path, ext: str) -> None:
    wav = audio.decode_to_wav16k(FX / f"short_0_5s.{ext}", tmp_path)
    with pytest.raises(Permanent) as ei:
        audio.bounds_check(audio.duration_s(wav), S)
    assert ei.value.code is ReasonCode.AUDIO_TOO_SHORT


@pytest.mark.parametrize("ext", ["webm", "m4a"])
def test_long_rejected(tmp_path: Path, ext: str) -> None:
    wav = audio.decode_to_wav16k(FX / f"en_95s.{ext}", tmp_path)
    with pytest.raises(Permanent) as ei:
        audio.bounds_check(audio.duration_s(wav), S)
    assert ei.value.code is ReasonCode.AUDIO_TOO_LONG


def test_garbage_undecodable(tmp_path: Path) -> None:
    bad = tmp_path / "junk.webm"
    bad.write_bytes(b"\x00not audio at all\xff" * 100)
    with pytest.raises(Permanent) as ei:
        audio.decode_to_wav16k(bad, tmp_path / "out")
    assert ei.value.code is ReasonCode.AUDIO_UNDECODABLE
    assert not list((tmp_path / "out").glob("*.wav"))


def test_missing_file_undecodable(tmp_path: Path) -> None:
    with pytest.raises(Permanent) as ei:
        audio.decode_to_wav16k(tmp_path / "nope.webm", tmp_path)
    assert ei.value.code is ReasonCode.AUDIO_UNDECODABLE


def test_exact_boundaries_accepted(tmp_path: Path) -> None:
    for secs in (1.0, 90.0):
        d = audio.duration_s(_wav(tmp_path / f"{secs}.wav", secs))
        assert d == secs
        audio.bounds_check(d, S)


def test_just_outside_boundaries(tmp_path: Path) -> None:
    with pytest.raises(Permanent) as ei:
        audio.bounds_check(audio.duration_s(_wav(tmp_path / "a.wav", 1 - 1 / 16000)), S)
    assert ei.value.code is ReasonCode.AUDIO_TOO_SHORT
    with pytest.raises(Permanent) as ei:
        audio.bounds_check(audio.duration_s(_wav(tmp_path / "b.wav", 90 + 1 / 16000)), S)
    assert ei.value.code is ReasonCode.AUDIO_TOO_LONG


def test_error_messages_are_fixed() -> None:
    e = Permanent(ReasonCode.AUDIO_TOO_SHORT, "audio")
    assert str(e) == "audio permanent: audio_too_short"
