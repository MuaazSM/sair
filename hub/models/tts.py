"""Kokoro TTS in a separate, lazily started child process (FR-M8, NFR-R1).

The parent starts ``python -m hub.models._tts_child <model_dir>`` on first use, keeps it
alive, and talks JSON lines over stdin/stdout. The child's stderr goes to /dev/null. Mission
text crosses the pipe only; neither side logs it (NFR-P4). Output is mp3 via ffmpeg.

Piper is an interface stub: ``TTS_ENGINE=piper`` raises ``Permanent`` until implemented.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

from hub.models.errors import Permanent, Transient
from hub.settings import Settings, get_settings

__all__ = ["KOEL_OVERRIDES", "TtsClient", "apply_overrides", "pick_lang"]

DEFAULT_VOICE = "hf_alpha"  # provisional (P0); Mom and Dad choose
REQUEST_TIMEOUT_S = 120.0  # first call includes the ~7 s model load
START_TIMEOUT_S = 60.0
MP3_BITRATE = "64k"

# Inline phonetic overrides, English pipeline only (misaki G2P says "coal" for "koel").
KOEL_OVERRIDES: dict[str, str] = {"koel": "[koel](/kˈOɛl/)"}

_LANG_ALIASES = {"en": "a", "a": "a", "b": "b", "hi": "h", "h": "h"}
_DEVANAGARI = re.compile("[ऀ-ॿ]")


def apply_overrides(text: str, lang: str) -> str:
    if lang not in ("a", "b"):
        return text
    for word, repl in KOEL_OVERRIDES.items():
        text = re.sub(rf"\b{re.escape(word)}\b", repl, text, flags=re.IGNORECASE)
    return text


def pick_lang(text: str, setting: str | None) -> str:
    """KOKORO_LANG: en/hi/a/b/h fixed; ``both`` (or unset) picks Hindi iff Devanagari is present."""
    if setting and setting.lower() in _LANG_ALIASES:
        return _LANG_ALIASES[setting.lower()]
    return "h" if _DEVANAGARI.search(text) else "a"


class TtsClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        command: Sequence[str] | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> None:
        self._s = settings or get_settings()
        self._command = list(command) if command else None
        self._timeout = timeout_s
        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self.starts = 0  # how many child processes were spawned (tests, metrics)

    # ------------------------------------------------------------------ process

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        return env

    def _pump(self, stream: IO[str]) -> None:
        for line in stream:
            self._lines.put(line)
        self._lines.put(None)  # EOF

    def _read(self, timeout: float) -> dict[str, Any]:
        """Next JSON object from the child; ignores non-JSON lines. Fixed errors only."""
        try:
            while True:
                line = self._lines.get(timeout=timeout)
                if line is None:
                    raise Transient(None, "tts")
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    return obj
        except queue.Empty:
            self._kill()
            raise Transient(None, "tts") from None

    def _ensure(self) -> subprocess.Popen[str]:
        if self.running and self._proc is not None:
            return self._proc
        self._kill()
        self._lines = queue.Queue()
        cmd = self._command or [
            sys.executable,
            "-m",
            "hub.models._tts_child",
            str(self._s.kokoro_model_dir),
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=self._child_env(),
            )
        except OSError:
            raise Transient(None, "tts") from None
        assert proc.stdout is not None
        self._proc = proc
        self.starts += 1
        threading.Thread(target=self._pump, args=(proc.stdout,), daemon=True).start()
        atexit.register(self.shutdown)
        ready = self._read(START_TIMEOUT_S)
        if not ready.get("ready"):
            self._kill()
            raise Transient(None, "tts")
        return proc

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
        for f in (proc.stdin, proc.stdout):
            try:
                if f:
                    f.close()
            except OSError:
                pass

    def shutdown(self) -> None:
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is None and proc.stdin is not None:
                try:
                    proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    proc.stdin.flush()
                    proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._kill()

    # ------------------------------------------------------------------ render

    def render(
        self, text: str, out_path: Path, voice: str | None = None, lang: str | None = None
    ) -> Path:
        """Render ``text`` to an mp3 at ``out_path`` (atomic) and return it."""
        if self._s.tts_engine == "piper":
            # Interface only: no Piper voice wiring yet.
            raise Permanent(None, "tts")
        if not text.strip():
            raise Permanent(None, "tts")
        kokoro_lang = _LANG_ALIASES.get((lang or "").lower()) if lang else None
        kl = kokoro_lang or pick_lang(text, self._s.kokoro_lang)
        voice = voice or self._s.kokoro_voice or DEFAULT_VOICE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fd, wav_name = tempfile.mkstemp(suffix=".wav", dir=out_path.parent)
        os.close(fd)
        wav = Path(wav_name)
        part = out_path.with_name(out_path.name + ".part")
        try:
            with self._lock:
                proc = self._ensure()
                req = {
                    "text": apply_overrides(text, kl),
                    "voice": voice,
                    "lang": kl,
                    "out": str(wav),
                }
                try:
                    assert proc.stdin is not None
                    proc.stdin.write(json.dumps(req) + "\n")
                    proc.stdin.flush()
                except (OSError, ValueError):
                    self._kill()
                    raise Transient(None, "tts") from None
                resp = self._read(self._timeout)
            if not resp.get("ok"):
                raise Permanent(None, "tts")
            self._to_mp3(wav, part)
            part.replace(out_path)
            return out_path
        finally:
            wav.unlink(missing_ok=True)
            part.unlink(missing_ok=True)

    @staticmethod
    def _to_mp3(wav: Path, dst: Path) -> None:
        cmd = ["ffmpeg", "-nostdin", "-v", "quiet", "-y", "-i", str(wav),
               "-c:a", "libmp3lame", "-b:a", MP3_BITRATE, "-f", "mp3", str(dst)]  # fmt: skip
        try:
            rc = subprocess.run(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=60, check=False,
            ).returncode  # fmt: skip
        except (OSError, subprocess.TimeoutExpired):
            raise Permanent(None, "tts") from None
        if rc != 0 or not dst.is_file() or dst.stat().st_size == 0:
            raise Permanent(None, "tts")

    def health(self) -> bool:
        """True if the engine is usable in principle (does not start the child)."""
        return (
            self._s.tts_engine == "kokoro" and (self._s.kokoro_model_dir / "config.json").is_file()
        )
