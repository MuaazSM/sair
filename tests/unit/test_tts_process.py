"""TTS child-process management (fake child) + one real Kokoro test (gpu)."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from hub.models import audio
from hub.models.errors import ModelError, Permanent, Transient
from hub.models.tts import TtsClient, apply_overrides, pick_lang
from hub.settings import Settings

FAKE = Path(__file__).parent / "_fake_tts_child.py"
MARK = "CANARY-MARKER-9c1d"


def settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[arg-type]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FAKE_TTS_LOG", str(tmp_path / "reqs.jsonl"))
    monkeypatch.setenv("FAKE_TTS_PIDFILE", str(tmp_path / "pids"))
    return tmp_path


@pytest.fixture
def client(env: Path) -> Iterator[TtsClient]:
    c = TtsClient(settings(), command=[sys.executable, str(FAKE)], timeout_s=20)
    yield c
    c.shutdown()


def test_lazy_start_and_reuse(client: TtsClient, env: Path) -> None:
    assert client.starts == 0 and not client.running
    out = client.render("Is the koel there again?", env / "a.mp3")
    assert client.starts == 1 and client.running
    client.render("Second mission.", env / "b.mp3")
    assert client.starts == 1
    assert len((env / "pids").read_text().split()) == 1
    for p in (out, env / "b.mp3"):
        assert p.stat().st_size > 0
        assert 0.9 < audio.duration_s(audio.decode_to_wav16k(p, env / "dec")) < 1.3
    assert not list(env.glob("*.wav")) and not list(env.glob("*.part"))


def test_shutdown_stops_child_and_restart_is_lazy(client: TtsClient, env: Path) -> None:
    client.render("One.", env / "a.mp3")
    client.shutdown()
    assert not client.running
    client.render("Two.", env / "b.mp3")
    assert client.starts == 2


def test_overrides_and_lang_selection(client: TtsClient, env: Path) -> None:
    client.render("Listen for the Koel today.", env / "a.mp3")
    client.render("क्या कोयल फिर है?", env / "b.mp3")
    reqs = [json.loads(x) for x in (env / "reqs.jsonl").read_text().splitlines()]
    assert reqs[0]["lang"] == "a" and "[koel](/kˈOɛl/)" in reqs[0]["text"]
    assert reqs[1]["lang"] == "h" and "[" not in reqs[1]["text"]
    assert reqs[0]["voice"] == "hf_alpha"


def test_pick_lang_and_overrides_unit() -> None:
    assert pick_lang("hello", "both") == "a" and pick_lang("नमस्ते", None) == "h"
    assert pick_lang("नमस्ते", "en") == "a" and pick_lang("hello", "hi") == "h"
    assert apply_overrides("koel koels", "h") == "koel koels"
    assert apply_overrides("a koel", "a") == "a [koel](/kˈOɛl/)"
    assert "koels" in apply_overrides("koels", "a")  # word boundary


def test_no_text_in_logs_or_errors(
    client: TtsClient,
    env: Path,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level(logging.DEBUG)
    client.render(f"say {MARK}", env / "a.mp3")
    out, err = capfd.readouterr()
    assert MARK not in caplog.text and MARK not in out and MARK not in err


def test_child_render_failure_is_permanent(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_TTS_FAIL", "1")
    c = TtsClient(settings(), command=[sys.executable, str(FAKE)], timeout_s=20)
    try:
        with pytest.raises(Permanent) as ei:
            c.render(f"x {MARK}", env / "a.mp3")
        assert MARK not in str(ei.value) and not (env / "a.mp3").exists()
    finally:
        c.shutdown()


def test_child_death_is_transient_and_recovers(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_TTS_DIE", "1")
    c = TtsClient(settings(), command=[sys.executable, str(FAKE)], timeout_s=20)
    try:
        with pytest.raises(Transient) as ei:
            c.render(f"x {MARK}", env / "a.mp3")
        assert MARK not in str(ei.value)
        monkeypatch.delenv("FAKE_TTS_DIE")
        c.render("again", env / "b.mp3")
        assert c.starts == 2
    finally:
        c.shutdown()


def test_timeout_is_transient(env: Path) -> None:
    c = TtsClient(
        settings(), command=[sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=0.3
    )
    import hub.models.tts as t

    old = t.START_TIMEOUT_S
    t.START_TIMEOUT_S = 0.3
    try:
        with pytest.raises(Transient):
            c.render("x", env / "a.mp3")
        assert not c.running
    finally:
        t.START_TIMEOUT_S = old
        c.shutdown()


def test_spawn_failure_is_transient(env: Path) -> None:
    c = TtsClient(settings(), command=["/nonexistent/binary"])
    with pytest.raises(Transient):
        c.render("x", env / "a.mp3")


def test_piper_without_voice_is_permanent(env: Path) -> None:
    c = TtsClient(settings(tts_engine="piper"))
    with pytest.raises(Permanent) as ei:
        c.render("x", env / "a.mp3")
    assert isinstance(ei.value, ModelError) and c.starts == 0


def test_empty_text_permanent(client: TtsClient, env: Path) -> None:
    with pytest.raises(Permanent):
        client.render("   ", env / "a.mp3")
    assert client.starts == 0


@pytest.mark.gpu
def test_real_kokoro_renders_english_and_hindi(tmp_path: Path) -> None:
    s = settings()
    if not (s.kokoro_model_dir / "kokoro-v1_0.pth").is_file():
        pytest.skip("kokoro weights not present")
    c = TtsClient(s, timeout_s=180)
    try:
        en = c.render(
            "Last time you heard a koel near the temple. Is it there again?", tmp_path / "en.mp3"
        )
        hi = c.render("पिछली बार आपने मंदिर के पास कोयल सुनी थी।", tmp_path / "hi.mp3")
        assert c.starts == 1
        for p in (en, hi):
            assert audio.duration_s(audio.decode_to_wav16k(p, tmp_path / "d")) > 1.5
    finally:
        c.shutdown()
