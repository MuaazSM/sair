"""FR-J2: energy VAD gate on 30 ms frames (-45 dBFS, >= 1.0 s voiced)."""

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


def _square(path: Path, db: float, seconds: float) -> Path:
    """Square wave: RMS equals its amplitude exactly, so frame dBFS == ``db``."""
    n = int(seconds * 16000)
    a = 10 ** (db / 20)
    x = np.where(np.arange(n) // 40 % 2 == 0, a, -a).astype(np.float32)
    sf.write(str(path), x, 16000, subtype="FLOAT")
    return path


def test_silent_fixture_rejected(tmp_path: Path) -> None:
    wav = audio.decode_to_wav16k(FX / "silent_5s.webm", tmp_path)
    with pytest.raises(Permanent) as ei:
        audio.vad_gate(wav, S)
    assert ei.value.code is ReasonCode.AUDIO_SILENT


@pytest.mark.parametrize("name", ["en_20s", "hinglish_20s"])
def test_speech_fixtures_accepted(tmp_path: Path, name: str) -> None:
    wav = audio.decode_to_wav16k(FX / f"{name}.m4a", tmp_path)
    st = audio.vad_gate(wav, S)
    assert st.passed and st.voiced_s >= 1.0 and st.threshold_db == -45.0


def test_threshold_boundary(tmp_path: Path) -> None:
    above = _square(tmp_path / "a.wav", -44.5, 2.0)
    below = _square(tmp_path / "b.wav", -45.5, 2.0)
    assert audio.vad_gate(above, S).voiced_s > 1.9
    with pytest.raises(Permanent):
        audio.vad_gate(below, S)


def test_threshold_from_settings(tmp_path: Path) -> None:
    quiet = _square(tmp_path / "q.wav", -50.0, 2.0)
    with pytest.raises(Permanent):
        audio.vad_gate(quiet, S)
    s2 = Settings(_env_file=None, vad_energy_threshold=-55.0)  # type: ignore[call-arg]
    assert audio.vad_gate(quiet, s2).passed


def test_min_voiced_boundary(tmp_path: Path) -> None:
    # 0.9 s of tone then silence -> rejected; 1.05 s -> accepted
    for secs, ok in ((0.9, False), (1.05, True)):
        tone = _square(tmp_path / "t.wav", -20.0, secs)
        x, _ = sf.read(str(tone), dtype="float32")
        sf.write(
            str(tmp_path / "m.wav"), np.concatenate([x, np.zeros(16000 * 2, np.float32)]), 16000
        )
        if ok:
            assert audio.vad_gate(tmp_path / "m.wav", S).passed
        else:
            with pytest.raises(Permanent):
                audio.vad_gate(tmp_path / "m.wav", S)
