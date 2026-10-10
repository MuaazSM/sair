"""FR-J2: silence-aware chunking without overlap, deterministic join, temp cleanup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hub.models import audio
from hub.models.ollama import join_transcripts

FX = Path(__file__).parent.parent / "fixtures" / "audio"
FRAME_S = audio.FRAME_S


@pytest.fixture(scope="module")
def wav95(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return audio.decode_to_wav16k(FX / "en_95s.webm", tmp_path_factory.mktemp("w95"))


def test_chunks_cover_file_without_overlap(wav95: Path, tmp_path: Path) -> None:
    total, _ = sf.read(str(wav95), dtype="float32")
    chunks = audio.split_on_silence(wav95, 30, tmp_path / "c")
    assert len(chunks) >= 4
    assert [c.name for c in chunks] == sorted(c.name for c in chunks)  # ordered names
    durs = [audio.duration_s(c) for c in chunks]
    assert all(d <= 30 + 1e-6 for d in durs)
    assert sum(durs) == pytest.approx(len(total) / 16000, abs=FRAME_S)
    # exact sample-level reassembly == no overlap, no gap
    joined = np.concatenate([sf.read(str(c), dtype="float32")[0] for c in chunks])
    assert joined.shape == total.shape
    assert np.allclose(joined, total, atol=1 / 32768)


def test_cuts_are_deterministic_and_quiet(wav95: Path, tmp_path: Path) -> None:
    a = audio.split_on_silence(wav95, 30, tmp_path / "a")
    b = audio.split_on_silence(wav95, 30, tmp_path / "b")
    assert [audio.duration_s(p) for p in a] == [audio.duration_s(p) for p in b]
    full = audio._read_mono(wav95)
    db = audio._frame_db(full)
    cut = 0
    for p in a[:-1]:
        cut += int(round(audio.duration_s(p) * 16000))
        # the cut frame is no louder than the median of its search window
        frame = db[cut // audio.FRAME]
        assert frame <= np.median(db[(cut - 9 * 16000) // audio.FRAME : cut // audio.FRAME + 1])


def test_short_file_is_one_chunk(tmp_path: Path) -> None:
    wav = audio.decode_to_wav16k(FX / "en_20s.m4a", tmp_path)
    chunks = audio.split_on_silence(wav, 30, tmp_path / "c")
    assert len(chunks) == 1


def test_context_manager_cleans_temp_chunks(wav95: Path) -> None:
    with audio.chunk_files(wav95, 30) as chunks:
        parent = chunks[0].parent
        assert all(c.is_file() for c in chunks)
    assert not parent.exists()


def test_context_manager_cleans_on_error(wav95: Path) -> None:
    holder: list[Path] = []
    with pytest.raises(RuntimeError), audio.chunk_files(wav95, 30) as chunks:
        holder.append(chunks[0].parent)
        raise RuntimeError("boom")
    assert not holder[0].exists()


def test_join_is_deterministic() -> None:
    parts = ["  hello there ", "", "second\n", " third"]
    assert join_transcripts(parts) == "hello there second third"
    assert join_transcripts(parts) == join_transcripts(list(parts))
    assert join_transcripts([]) == ""
