"""Backend factory for live/cached eval runs (no network at import time).

URLs and model names come from ``hub/settings.py`` (Ollama host/model, Sair-Writer URL).
Two eval-only knobs are read from the environment, with loopback defaults:

* ``EVAL_QWEN_BASE_URL`` (default ``http://127.0.0.1:8092``): llama-server with the stock
  Qwen3.5-4B Q4_K_M (system C);
* ``EVAL_QWEN_BASE_ID`` / ``EVAL_SAIR_WRITER_ID``: model ids used in cache keys.

System D (Sair-Writer) is served at ``WRITER_URL`` (default ``http://127.0.0.1:8091``).

C and D each take their own training prompt renderer (an object with
``render(messages) -> str``): ``--renderer-c`` / ``--renderer-d`` (``module:attr``), with
``--renderer`` as a fallback for both. Unset, they default to
``training.renderer_runtime:few_shot_renderer`` (C) and ``training.renderer_runtime:renderer``
(D) when those load cleanly (C needs ``training/data/few_shot.json``). An explicit spec that
fails to load is an error; a default that fails leaves a placeholder that refuses live calls.
The renderer fingerprint is part of the C/D cache key (``eval/backends.py``), so in
``--cached`` mode the same renderer must load for the cache to answer.
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
    "DEFAULT_RENDERER_C",
    "DEFAULT_RENDERER_D",
    "build_writers",
    "is_loopback_url",
    "load_renderer",
    "resolve_renderer",
]

DEFAULT_RENDERER_C = "training.renderer_runtime:few_shot_renderer"
DEFAULT_RENDERER_D = "training.renderer_runtime:renderer"
DEFAULT_QWEN_BASE_URL = "http://127.0.0.1:8092"


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
        raise ConfigError("no prompt renderer configured (pass --renderer-c/--renderer-d)")

    def fingerprint(self) -> str:
        return "none"


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


def resolve_renderer(spec: str | None, default: str) -> tuple[PromptRenderer, bool]:
    """(renderer, ok). An explicit ``spec`` must load; the ``default`` may fail quietly
    (type name never needed: the placeholder refuses live calls with a fixed message)."""
    if spec:
        return load_renderer(spec), True
    try:
        return load_renderer(default), True
    except Exception:  # noqa: BLE001 - e.g. few_shot.json not frozen yet
        return _NoRenderer(), False


def build_writers(
    keys: Sequence[str],
    *,
    cache_dir: Path,
    mode: CacheMode,
    renderer_spec: str | None = None,
    renderer_c: str | None = None,
    renderer_d: str | None = None,
) -> dict[str, WriterBackend]:
    """Writer backends by key (template, gemma, qwen_base, sair_writer), cache-wrapped.

    ``renderer_c`` / ``renderer_d`` pick the C / D renderers; ``renderer_spec`` (legacy
    ``--renderer``) is the fallback for both."""
    s = get_settings()
    live = mode is not CacheMode.CACHED
    rend: dict[str, PromptRenderer] = {}
    for key, label, spec, default in (
        ("qwen_base", "C", renderer_c or renderer_spec, DEFAULT_RENDERER_C),
        ("sair_writer", "D", renderer_d or renderer_spec, DEFAULT_RENDERER_D),
    ):
        if key not in keys:
            continue
        r, ok = resolve_renderer(spec, default)
        if not ok and live:
            raise ConfigError(
                f"system {label} needs --renderer-{label.lower()} module:attr for live runs"
            )
        rend[key] = r
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
                os.environ.get("EVAL_QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL),
                rend[key],
                model_id=os.environ.get("EVAL_QWEN_BASE_ID", "qwen3.5-4b-base-q4_k_m"),
            )
        elif key == "sair_writer":
            inner = LlamaServerWriter(
                s.writer_url,
                rend[key],
                model_id=os.environ.get("EVAL_SAIR_WRITER_ID", "sair-writer-q4_k_m"),
            )
        else:
            raise ConfigError("unknown writer backend key")
        out[key] = CachedWriter(inner, cache_dir, mode)
    return out


def build_planner(*, cache_dir: Path, mode: CacheMode) -> PlannerBackend:
    s = get_settings()
    return CachedPlanner(OllamaPlanner(s.ollama_host, s.ollama_model), cache_dir, mode)
