"""Backend factory for live/cached eval runs (no network at import time).

URLs and model names come from ``hub/settings.py`` (Ollama host/model, Sair-Writer URL).
Two eval-only knobs are read from the environment, with loopback defaults:

* ``EVAL_QWEN_BASE_URL`` (default ``http://127.0.0.1:8092``): llama-server with the stock
  Qwen3.5-4B Q4_K_M (system C);
* ``EVAL_QWEN_BASE_ID`` / ``EVAL_SAIR_WRITER_ID``: model ids used in cache keys.

C and D need the training prompt renderer (``--renderer module:attr``, an object with
``render(messages) -> str``). In ``--cached`` mode a placeholder renderer is used: the
cache answers every call, and a call that reaches the placeholder raises.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

from eval.backends import (
    CachedPlanner,
    CachedWriter,
    CacheMode,
    LlamaServerWriter,
    OllamaPlanner,
    OllamaWriter,
    PlannerBackend,
    PromptRenderer,
    TemplateWriter,
    WriterBackend,
)
from hub.prompts.builders import ChatMessage
from hub.settings import LOOPBACK_HOSTS, get_settings

__all__ = [
    "ConfigError",
    "build_planner",
    "build_writers",
    "is_loopback_url",
    "load_renderer",
]


class ConfigError(RuntimeError):
    """Fixed-message configuration error."""


def is_loopback_url(url: str | None) -> bool:
    """True for None (in-process backend) or an http(s) URL whose host is loopback."""
    if url is None:
        return True
    try:
        parsed = urlparse(url)
        host = parsed.hostname
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and host is not None and host in LOOPBACK_HOSTS


class _NoRenderer:
    def render(self, messages: Sequence[ChatMessage]) -> str:
        raise ConfigError("no prompt renderer configured (pass --renderer module:attr)")


def load_renderer(spec: str | None) -> PromptRenderer:
    if not spec:
        return _NoRenderer()
    mod_name, _, attr = spec.partition(":")
    if not mod_name or not attr:
        raise ConfigError("--renderer must look like module:attr")
    obj = getattr(importlib.import_module(mod_name), attr)
    renderer = obj() if isinstance(obj, type) else obj
    if not hasattr(renderer, "render"):
        raise ConfigError("renderer object has no render(messages) method")
    return renderer  # type: ignore[no-any-return]


def build_writers(
    keys: Sequence[str],
    *,
    cache_dir: Path,
    mode: CacheMode,
    renderer_spec: str | None = None,
) -> dict[str, WriterBackend]:
    """Writer backends by key (template, gemma, qwen_base, sair_writer), cache-wrapped."""
    s = get_settings()
    renderer = load_renderer(renderer_spec)
    if (
        renderer_spec is None
        and mode not in (CacheMode.CACHED,)
        and ("qwen_base" in keys or "sair_writer" in keys)
    ):
        raise ConfigError("systems C/D need --renderer module:attr for live runs")
    out: dict[str, WriterBackend] = {}
    for key in dict.fromkeys(keys):
        inner: WriterBackend
        if key == "template":
            out[key] = TemplateWriter()
            continue
        if key == "gemma":
            inner = OllamaWriter(s.ollama_host, s.ollama_model)
        elif key == "qwen_base":
            inner = LlamaServerWriter(
                os.environ.get("EVAL_QWEN_BASE_URL", "http://127.0.0.1:8092"),
                renderer,
                model_id=os.environ.get("EVAL_QWEN_BASE_ID", "qwen3.5-4b-base-q4_k_m"),
            )
        elif key == "sair_writer":
            inner = LlamaServerWriter(
                s.writer_url,
                renderer,
                model_id=os.environ.get("EVAL_SAIR_WRITER_ID", "sair-writer-q4_k_m"),
            )
        else:
            raise ConfigError("unknown writer backend key")
        out[key] = CachedWriter(inner, cache_dir, mode)
    return out


def build_planner(*, cache_dir: Path, mode: CacheMode) -> PlannerBackend:
    s = get_settings()
    return CachedPlanner(OllamaPlanner(s.ollama_host, s.ollama_model), cache_dir, mode)
