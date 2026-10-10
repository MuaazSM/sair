"""Hub runtime settings (IMPLEMENTATION §0.4). The only place configuration is read.

Privacy and network rules enforced here at startup (a violation raises a pydantic
``ValidationError`` with a fixed message; the offending value is never echoed, NFR-P4):

* NFR-P2: ``SAIR_HOST`` must be loopback.
* NFR-P1/P3: in prod every model/notification URL and every allowed outbound host is loopback,
  and the HuggingFace offline flags stay on.
* NFR-P7: in prod ``SENTRY_DSN`` must be empty; Sentry is enabled only in dev/eval.

Env files: ``.env`` for prod; ``.env.eval`` is layered on top in dev/eval. ``SAIR_ENV=test``
loads no env file at all. Tests can also build ``Settings(_env_file=None, ...)``.
"""

from __future__ import annotations

import datetime as dt
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import dotenv_values
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from hub.schemas import WriterBackend

__all__ = [
    "LOOPBACK_HOSTS",
    "Settings",
    "apply_offline_env",
    "get_settings",
    "reset_settings",
]

LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})

# Fixed messages: never include the rejected value.
MSG_HOST = "SAIR_HOST must be a loopback address (127.0.0.1, ::1 or localhost)"
MSG_URL = "model and notification URLs must be loopback in prod"
MSG_SENTRY = "SENTRY_DSN must be empty in prod"
MSG_OUTBOUND = "ALLOWED_OUTBOUND_HOSTS may contain only loopback names in prod"
MSG_OFFLINE = "HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE must be on in prod"
MSG_TZ = "SAIR_TZ is not a known time zone"
MSG_GROUPS = "SAIR_DEVICE_GROUPS must look like device:Group Name,device:Group Name"
MSG_BLOCK = "MISSION_PREGEN_BLOCK must look like HH:MM-HH:MM"


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    return value


def _is_loopback_url(url: str) -> bool:
    from urllib.parse import urlparse

    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    return host is not None and host in LOOPBACK_HOSTS


def _parse_device_groups(value: object) -> object:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value
    out: dict[str, str] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        device, sep, group = part.partition(":")
        device, group = device.strip(), group.strip()
        if not sep or not device or not group:
            raise ValueError(MSG_GROUPS)
        out[device] = group
    return out


def _parse_block(value: object) -> object:
    if isinstance(value, str):
        start, sep, end = value.partition("-")
        if not sep:
            raise ValueError(MSG_BLOCK)
        try:
            return (dt.time.fromisoformat(start.strip()), dt.time.fromisoformat(end.strip()))
        except ValueError:
            raise ValueError(MSG_BLOCK) from None
    return value


class Settings(BaseSettings):
    """Typed hub configuration. Field names are the lowercase env var names."""

    model_config = SettingsConfigDict(
        extra="ignore",
        env_ignore_empty=True,
        hide_input_in_errors=True,
        case_sensitive=False,
        frozen=True,
    )

    # --- core
    sair_env: Literal["dev", "eval", "test", "prod"] = "prod"
    sair_host: str = "127.0.0.1"
    sair_port: int = Field(default=8787, ge=1, le=65535)
    sair_data_dir: Path = Path("~/.sair")
    sair_db_path: Path | None = None
    sair_tmp_dir: Path | None = None
    sair_tz: str = "Asia/Kolkata"
    sair_token_mom: SecretStr = SecretStr("")
    sair_token_dad: SecretStr = SecretStr("")
    sair_token_owner: SecretStr = SecretStr("")
    sair_device_groups: Annotated[dict[str, str], NoDecode] = Field(
        default_factory=lambda: {"mom": "Mom and Dad", "dad": "Mom and Dad"}
    )

    # --- extra directories (not in §0.4; read by scripts/forget.sh, see report)
    sair_audio_dir: Path | None = None  # mission audio: <audio_dir>/<mission_id>.mp3
    sair_audio_cache_dir: Path | None = None
    sair_log_dir: Path = Path("logs")
    sair_checkpoint_dir: Path | None = None

    # --- audio
    audio_min_s: float = 1
    audio_max_s: float = 90
    vad_energy_threshold: float | None = None
    transcribe_chunk_s: float = Field(default=0, ge=0)
    transient_retry_window_s: int = Field(default=600, ge=0)

    # --- models
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma4:e2b"
    ollama_keep_alive: str = "30m"
    ollama_debug: bool = False
    transcriber: Literal["gemma", "whisper"] = "gemma"
    whisper_model_path: Path | None = None

    writer_backend: WriterBackend = WriterBackend.GEMMA
    writer_url: str = "http://127.0.0.1:8091"
    writer_model_path: Path = Path("models/sair-writer-q4_k_m.gguf")
    writer_grammar: Path = Path("grammars/entry.gbnf")
    writer_json_schema: Path = Path("schemas/entry.schema.json")
    writer_max_tokens: int = Field(default=400, ge=1)
    # Timeouts/sampling read by hub/graphs (the model clients keep their own constants).
    ollama_transcribe_timeout_s: float = Field(default=180, gt=0)
    ollama_chat_timeout_s: float = Field(default=120, gt=0)
    writer_timeout_s: float = Field(default=120, gt=0)
    planner_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    tts_timeout_s: float = Field(default=120, gt=0)

    tts_engine: Literal["kokoro", "piper"] = "kokoro"
    kokoro_model_dir: Path = Path("models/kokoro")
    kokoro_voice: str | None = None
    kokoro_lang: str | None = None
    piper_voice_path: Path | None = None

    # --- missions
    mission_pregen_time: dt.time = dt.time(16, 30)
    mission_pregen_block: Annotated[tuple[dt.time, dt.time], NoDecode] = (
        dt.time(11, 30),
        dt.time(16, 0),
    )

    # --- offline hardening (PRD §11.1)
    hf_hub_offline: bool = True
    transformers_offline: bool = True
    hf_hub_disable_telemetry: bool = True

    # --- notifications
    ntfy_url: str = "http://127.0.0.1:8090"
    ntfy_topic: str | None = None
    ntfy_ios_upstream: Literal["off", "on"] = "off"

    # --- observability (dev/eval only)
    sentry_dsn: SecretStr = SecretStr("")
    sentry_environment: str | None = None
    sentry_traces_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    allowed_outbound_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["127.0.0.1", "localhost"]
    )
    canary_marker: str | None = None

    # ------------------------------------------------------------ validators

    @field_validator("sair_env", mode="before")
    @classmethod
    def _env_lower(cls, v: object) -> object:
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("sair_device_groups", mode="before")
    @classmethod
    def _groups(cls, v: object) -> object:
        return _parse_device_groups(v)

    @field_validator("allowed_outbound_hosts", mode="before")
    @classmethod
    def _hosts(cls, v: object) -> object:
        return _split_csv(v)

    @field_validator("mission_pregen_block", mode="before")
    @classmethod
    def _block(cls, v: object) -> object:
        return _parse_block(v)

    @field_validator("sair_host")
    @classmethod
    def _loopback_host(cls, v: str) -> str:
        if v.strip() not in LOOPBACK_HOSTS:
            raise ValueError(MSG_HOST)
        return v.strip()

    @field_validator("sair_tz")
    @classmethod
    def _tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError, OSError):
            raise ValueError(MSG_TZ) from None
        return v

    @field_validator(
        "sair_data_dir",
        "sair_db_path",
        "sair_tmp_dir",
        "sair_audio_dir",
        "sair_audio_cache_dir",
        "sair_log_dir",
        "sair_checkpoint_dir",
        "whisper_model_path",
        "writer_model_path",
        "writer_grammar",
        "writer_json_schema",
        "kokoro_model_dir",
        "piper_voice_path",
    )
    @classmethod
    def _expand(cls, v: Path | None) -> Path | None:
        return v.expanduser() if v is not None else None

    @model_validator(mode="after")
    def _cross_checks(self) -> Settings:
        if self.sair_env == "prod":
            if self.sentry_dsn.get_secret_value():
                raise ValueError(MSG_SENTRY)
            for url in (self.ollama_host, self.writer_url, self.ntfy_url):
                if not _is_loopback_url(url):
                    raise ValueError(MSG_URL)
            if any(h not in LOOPBACK_HOSTS for h in self.allowed_outbound_hosts):
                raise ValueError(MSG_OUTBOUND)
            if not (self.hf_hub_offline and self.transformers_offline):
                raise ValueError(MSG_OFFLINE)
        return self

    # ------------------------------------------------- derived paths and values

    @property
    def data_dir(self) -> Path:
        return self.sair_data_dir

    @property
    def db_path(self) -> Path:
        return self.sair_db_path or self.sair_data_dir / "sair.db"

    @property
    def tmp_dir(self) -> Path:
        """Raw audio only (FR-J6)."""
        return self.sair_tmp_dir or self.sair_data_dir / "tmp"

    @property
    def audio_dir(self) -> Path:
        """Rendered mission audio (FR-M8): ``audio_dir / f"{mission_id}.mp3"``."""
        return self.sair_audio_dir or self.sair_data_dir / "audio"

    @property
    def audio_cache_dir(self) -> Path:
        return self.sair_audio_cache_dir or self.sair_data_dir / "cache"

    @property
    def log_dir(self) -> Path:
        return self.sair_log_dir

    @property
    def checkpoint_dir(self) -> Path:
        return self.sair_checkpoint_dir or self.sair_data_dir / "checkpoints"

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.sair_tz)

    @property
    def device_groups(self) -> dict[str, str]:
        return dict(self.sair_device_groups)

    @property
    def sentry_enabled(self) -> bool:
        """NFR-P7: only dev/eval with a DSN."""
        return self.sair_env in ("dev", "eval") and bool(self.sentry_dsn.get_secret_value())

    def token_for(self, device: str) -> str:
        """Plain token for ``mom`` / ``dad`` / ``owner`` ('' if unset or unknown device)."""
        secret: SecretStr | None = {
            "mom": self.sair_token_mom,
            "dad": self.sair_token_dad,
            "owner": self.sair_token_owner,
        }.get(device)
        return secret.get_secret_value() if secret is not None else ""


# ------------------------------------------------------------------- loading


def _env_files() -> tuple[str, ...]:
    env = os.environ.get("SAIR_ENV")
    if env is None and Path(".env").is_file():
        env = dotenv_values(".env").get("SAIR_ENV")
    env = (env or "prod").strip().lower()
    if env == "test":
        return ()
    if env in ("dev", "eval"):
        return (".env", ".env.eval")
    return (".env",)


def load_settings() -> Settings:
    """Build settings fresh from the environment and the env files for ``SAIR_ENV``."""
    files = _env_files()
    return Settings(_env_file=files or None)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reset_settings() -> None:
    """Drop the cached settings (tests)."""
    get_settings.cache_clear()


def apply_offline_env(settings: Settings | None = None) -> None:
    """Export the HuggingFace offline flags to ``os.environ`` (call at hub startup)."""
    s = settings or get_settings()
    os.environ["HF_HUB_OFFLINE"] = "1" if s.hf_hub_offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if s.transformers_offline else "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1" if s.hf_hub_disable_telemetry else "0"
