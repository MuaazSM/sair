"""§T FR-O5: ``scripts/forget.sh`` wipes every local store (incl. mission audio) via hub.db.forget.

Every path the script could touch is overridden into ``tmp_path`` (and HOME too), and the
script is told to refuse anything outside it (``SAIR_FORGET_ROOT``), so the live data dir is
never touched by these tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "forget.sh"


def _uv_cache_dir() -> str:
    out = subprocess.run(["uv", "cache", "dir"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _layout(tmp: Path) -> tuple[dict[str, str], list[Path]]:
    data = tmp / "data"
    dirs = {
        "SAIR_DATA_DIR": data,
        "SAIR_DB_PATH": data / "sair.db",
        "SAIR_TMP_DIR": data / "tmp",
        "SAIR_AUDIO_DIR": data / "audio",
        "SAIR_AUDIO_CACHE_DIR": data / "cache",
        "SAIR_LOG_DIR": tmp / "logs",
        "SAIR_CHECKPOINT_DIR": data / "checkpoints",
    }
    files = [
        data / "sair.db",
        data / "sair.db-wal",
        data / "sair.db-shm",
        data / "tmp" / "w1.webm",
        data / "audio" / ("a" * 32 + ".mp3"),
        data / "audio" / ("b" * 32 + ".mp3"),
        data / "cache" / "x.bin",
        data / "checkpoints" / "ck.json",
        data / "ntfy" / "cache.db",  # self-hosted ntfy message cache (com.sair.ntfy plist)
        tmp / "logs" / "llama-server.log",
        tmp / "logs" / "sub" / "ollama.log",
    ]
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"synthetic")
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp / "home"),  # any default under ~ lands in tmp_path
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or _uv_cache_dir(),
        "SAIR_ENV": "test",  # no env files are read
        "SAIR_FORGET_ROOT": str(tmp),
        **{k: str(v) for k, v in dirs.items()},
    }
    (tmp / "home").mkdir()
    return env, files


def _run(env: dict[str, str], answer: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        input=answer,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO,
    )


def test_yes_removes_everything_incl_mission_audio(tmp_path: Path) -> None:
    env, files = _layout(tmp_path)
    r = _run(env, "y\n")
    assert r.returncode == 0, r.stderr
    left = [f for f in files if f.exists()]
    assert left == []
    assert not (tmp_path / "data" / "audio").exists()
    assert not (tmp_path / "data" / "ntfy").exists()
    assert "ntfy" in r.stdout  # listed as a target before asking
    assert "removed" in r.stdout and "paths" in r.stdout
    assert "synthetic" not in r.stdout + r.stderr  # never file contents


def test_no_deletes_nothing(tmp_path: Path) -> None:
    env, files = _layout(tmp_path)
    r = _run(env, "n\n")
    assert r.returncode == 0
    assert "Aborted" in r.stdout
    assert all(f.is_file() for f in files)


def test_refuses_paths_outside_forget_root(tmp_path: Path) -> None:
    env, files = _layout(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    env["SAIR_AUDIO_DIR"] = str(outside)
    r = _run(env, "y\n")
    assert r.returncode != 0
    assert "outside" in r.stderr
    assert all(f.is_file() for f in files)


def test_stops_the_hub_from_its_pid_file_first(tmp_path: Path) -> None:
    env, _ = _layout(tmp_path)
    # A stand-in hub process whose command line names the hub, recorded where up.sh writes it.
    fake = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "hub"])
    try:
        (tmp_path / "logs" / "hub.pid").write_text(str(fake.pid))
        r = _run(env, "y\n")
        assert r.returncode == 0, r.stderr
        deadline = time.monotonic() + 5
        while fake.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert fake.poll() is not None, "hub was stopped before wiping"
        assert "hub: stopped" in r.stdout
    finally:
        if fake.poll() is None:
            fake.kill()
        fake.wait()


def test_stops_ntfy_from_its_pid_file_first(tmp_path: Path) -> None:
    env, _ = _layout(tmp_path)
    fake = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "ntfy"])
    try:
        (tmp_path / "logs" / "ntfy.pid").write_text(str(fake.pid))
        r = _run(env, "y\n")
        assert r.returncode == 0, r.stderr
        deadline = time.monotonic() + 5
        while fake.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert fake.poll() is not None, "ntfy was stopped before wiping its cache"
        assert "ntfy: stopped" in r.stdout
        assert not (tmp_path / "data" / "ntfy").exists()
    finally:
        if fake.poll() is None:
            fake.kill()
        fake.wait()
