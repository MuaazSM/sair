"""FR-O5: forget wipes DB, temp audio, caches, mission audio, model-server logs, checkpoints."""

from __future__ import annotations

from pathlib import Path

from hub import db
from hub.settings import Settings
from tests.unit._dbfix import add_walk


def test_forget_removes_everything(tmp_path: Path) -> None:
    data = tmp_path / "sair"
    logs = tmp_path / "logs"
    s = Settings(_env_file=None, sair_data_dir=data, sair_log_dir=logs)  # type: ignore[call-arg]
    conn = db.init_db(s.db_path)
    add_walk(conn, "w1", 1)
    conn.close()  # WAL/SHM files exist or are folded; either way they must be gone after
    for d in (s.tmp_dir, s.audio_dir, s.audio_cache_dir, s.checkpoint_dir, logs / "nested"):
        d.mkdir(parents=True)
    (s.tmp_dir / "walk.webm").write_bytes(b"\x00")
    (s.audio_dir / "m1.mp3").write_bytes(b"\x00")
    (s.audio_cache_dir / "x.wav").write_bytes(b"\x00")
    (s.checkpoint_dir / "c.db").write_bytes(b"\x00")
    (logs / "llama-server.log").write_text("fake")
    (logs / "nested" / "ollama.log").write_text("fake")
    s.db_path.with_name("sair.db-wal").write_bytes(b"\x00")
    s.db_path.with_name("sair.db-shm").write_bytes(b"\x00")
    keep = tmp_path / "unrelated.txt"
    keep.write_text("keep")

    removed = db.forget(s)

    assert removed
    assert not s.db_path.exists()
    assert not s.db_path.with_name("sair.db-wal").exists()
    assert not s.db_path.with_name("sair.db-shm").exists()
    for d in (s.tmp_dir, s.audio_dir, s.audio_cache_dir, s.checkpoint_dir):
        assert not d.exists()
    assert list(logs.rglob("*")) == []
    assert keep.exists()
    assert db.raw_audio_on_disk(s.tmp_dir) == 0


def test_forget_on_empty_state_is_noop(tmp_path: Path) -> None:
    s = Settings(_env_file=None, sair_data_dir=tmp_path / "none", sair_log_dir=tmp_path / "nologs")  # type: ignore[call-arg]
    assert db.forget(s) == []


def test_forget_removes_self_hosted_ntfy_cache(tmp_path: Path) -> None:
    """FR-O5: the ntfy LaunchAgent caches messages at <data_dir>/ntfy/cache.db; forget wipes it."""
    s = Settings(_env_file=None, sair_data_dir=tmp_path / "sair", sair_log_dir=tmp_path / "logs")  # type: ignore[call-arg]
    cache = s.data_dir / "ntfy" / "cache.db"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"\x00")
    (s.data_dir / "ntfy" / "cache.db-wal").write_bytes(b"\x00")
    assert s.ntfy_cache_dir == s.data_dir / "ntfy"

    removed = db.forget(s)

    assert str(s.ntfy_cache_dir) in removed
    assert not s.ntfy_cache_dir.exists()
