"""Model backends for the eval harness: small protocols, thin lazy adapters, stubs, cache.

Every backend exposes:

* ``backend_id``: a stable string used in cache keys and reports (no content);
* ``base_url``: the server URL, or ``None`` for in-process backends (template, stub).
  ``eval/run.py --set real`` refuses unless every non-None ``base_url`` is loopback;
* ``last_call``: a :class:`CallStats` for the latest call (latency, tokens if known).

The HTTP adapters import ``httpx`` lazily and are only constructed for live runs; the
test-suite never touches them. :class:`CachedWriter` / :class:`CachedPlanner` wrap any
backend and store each call under ``<cache_dir>/<sha>.json`` keyed by
``(backend_id, grammar flag, prompt sha)`` plus, for backends that render their own
prompt string (C and D), the renderer fingerprint, so a changed renderer or few-shot block can
never replay stale outputs (backends without a renderer keep the original key); writes are
atomic so a killed run never leaves a half-written entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from hub.grounding import build_template_entry
from hub.prompts.builders import ChatMessage

__all__ = [
    "CacheMiss",
    "CacheMode",
    "CachedPlanner",
    "CachedWriter",
    "CallStats",
    "LlamaServerWriter",
    "OllamaPlanner",
    "OllamaWriter",
    "PlannerBackend",
    "PromptRenderer",
    "StubPlanner",
    "StubWriter",
    "TemplateWriter",
    "WriterBackend",
    "prompt_sha",
    "renderer_fingerprint",
]

DEFAULT_TIMEOUT_S = 120.0
MAX_TOKENS = 400  # PRD §9.1


@dataclass(frozen=True, slots=True)
class CallStats:
    latency_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    cached: bool = False


@runtime_checkable
class WriterBackend(Protocol):
    backend_id: str
    base_url: str | None
    last_call: CallStats | None

    def generate(self, messages: list[ChatMessage], *, grammar: bool) -> str: ...


@runtime_checkable
class PlannerBackend(Protocol):
    backend_id: str
    base_url: str | None
    last_call: CallStats | None

    def plan(self, messages: list[ChatMessage]) -> str: ...


class PromptRenderer(Protocol):
    """Renders chat messages to the raw prompt string used in training (P5 renderer)."""

    def render(self, messages: Sequence[ChatMessage]) -> str: ...


def renderer_fingerprint(renderer: object) -> str:
    """``renderer.fingerprint()`` when present, else a sha256 over the renderer's type and its
    instance state (stable across processes; never an object address)."""
    fp = getattr(renderer, "fingerprint", None)
    if callable(fp):
        return str(fp())
    cls = type(renderer)
    state = getattr(renderer, "__dict__", None)
    blob = json.dumps(
        {"type": f"{cls.__module__}.{cls.__qualname__}", "state": state},
        sort_keys=True,
        default=repr,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _as_dicts(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def prompt_sha(messages: Sequence[ChatMessage], *, grammar: bool | None = None) -> str:
    """sha256 over the canonical JSON of the messages (+ grammar flag when given)."""
    payload: dict[str, Any] = {"messages": _as_dicts(messages)}
    if grammar is not None:
        payload["grammar"] = grammar
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_json_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("schema file must hold a JSON object")
    return data


# --------------------------------------------------------------------------- HTTP adapters


class OllamaWriter:
    """System B: Gemma via Ollama ``/api/chat``; ``grammar=True`` sends the JSON schema."""

    def __init__(
        self,
        url: str,
        model: str,
        *,
        schema_path: Path | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.base_url: str | None = url.rstrip("/")
        self.model = model
        self.backend_id = f"ollama:{model}"
        self.schema_path = schema_path or _root() / "schemas" / "entry.schema.json"
        self.timeout_s = timeout_s
        self.last_call: CallStats | None = None

    def generate(self, messages: list[ChatMessage], *, grammar: bool) -> str:
        import httpx  # lazy: never imported by stubbed tests

        body: dict[str, Any] = {
            "model": self.model,
            "messages": _as_dicts(messages),
            "stream": False,
            "options": {"temperature": 0, "num_predict": MAX_TOKENS},
        }
        if grammar:
            body["format"] = _read_json_file(self.schema_path)
        t0 = time.perf_counter()
        resp = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        self.last_call = CallStats(
            latency_ms=(time.perf_counter() - t0) * 1000,
            tokens_in=data.get("prompt_eval_count"),
            tokens_out=data.get("eval_count"),
        )
        return str(data.get("message", {}).get("content", ""))


class OllamaPlanner:
    """The fixed planner (Gemma, PRD §13.4) via Ollama with the mission JSON schema."""

    def __init__(
        self,
        url: str,
        model: str,
        *,
        schema_path: Path | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._writer = OllamaWriter(
            url,
            model,
            schema_path=schema_path or _root() / "schemas" / "mission.schema.json",
            timeout_s=timeout_s,
        )
        self.base_url: str | None = self._writer.base_url
        self.backend_id = f"ollama-planner:{model}"
        self.last_call: CallStats | None = None

    def plan(self, messages: list[ChatMessage]) -> str:
        out = self._writer.generate(messages, grammar=True)
        self.last_call = self._writer.last_call
        return out


class LlamaServerWriter:
    """Systems C and D: llama-server raw ``/completion`` with the training renderer.

    ``renderer`` comes from the training code (same prompt string as training, PRD §12.4).
    ``grammar=True`` sends ``grammars/entry.gbnf``.
    """

    def __init__(
        self,
        url: str,
        renderer: PromptRenderer,
        *,
        model_id: str,
        grammar_path: Path | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.base_url: str | None = url.rstrip("/")
        self.renderer = renderer
        self.backend_id = f"llama:{model_id}"
        self.grammar_path = grammar_path or _root() / "grammars" / "entry.gbnf"
        self.timeout_s = timeout_s
        self.last_call: CallStats | None = None

    def generate(self, messages: list[ChatMessage], *, grammar: bool) -> str:
        import httpx  # lazy

        body: dict[str, Any] = {
            "prompt": self.renderer.render(messages),
            "n_predict": MAX_TOKENS,
            "temperature": 0,
            "cache_prompt": True,
        }
        if grammar:
            body["grammar"] = self.grammar_path.read_text(encoding="utf-8")
        t0 = time.perf_counter()
        resp = httpx.post(f"{self.base_url}/completion", json=body, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        self.last_call = CallStats(
            latency_ms=(time.perf_counter() - t0) * 1000,
            tokens_in=data.get("tokens_evaluated"),
            tokens_out=data.get("tokens_predicted"),
        )
        return str(data.get("content", ""))


# --------------------------------------------------------------------------- in-process


_WALKERS_RE = re.compile(r"^Walkers: (.+)$", re.MULTILINE)


class TemplateWriter:
    """System A: the no-facts template ``{Group} went for their evening walk.`` for every item.

    The group is read back from the ``Walkers:`` line that ``build_writer_prompt`` writes.
    """

    backend_id = "template:no-facts"

    def __init__(self) -> None:
        self.base_url: str | None = None
        self.last_call: CallStats | None = None

    def generate(self, messages: list[ChatMessage], *, grammar: bool) -> str:
        t0 = time.perf_counter()
        user = next((m.content for m in messages if m.role == "user"), "")
        match = _WALKERS_RE.search(user)
        if match is None:
            raise ValueError("template writer: prompt has no walkers line")
        entry = build_template_entry(match.group(1).strip(), [], False)
        out = json.dumps(
            {"facts": [], "mission_result": "not_mentioned", "mission_span": None, "entry": entry},
            ensure_ascii=False,
        )
        self.last_call = CallStats(latency_ms=(time.perf_counter() - t0) * 1000)
        return out


StubFn = Callable[[list[ChatMessage], bool], str]


class StubWriter:
    """Test writer: ``responses`` is a list (served in order, last one repeated) or a
    callable ``(messages, grammar) -> str``. Records every call's messages."""

    def __init__(
        self,
        responses: Sequence[str] | StubFn,
        *,
        backend_id: str = "stub-writer",
        base_url: str | None = None,
        latency_ms: float = 1.0,
    ) -> None:
        self._responses = responses
        self.backend_id = backend_id
        self.base_url = base_url
        self.latency_ms = latency_ms
        self.calls: list[tuple[list[ChatMessage], bool]] = []
        self.last_call: CallStats | None = None

    def generate(self, messages: list[ChatMessage], *, grammar: bool) -> str:
        self.calls.append((list(messages), grammar))
        if callable(self._responses):
            out = self._responses(messages, grammar)
        else:
            seq = list(self._responses)
            if not seq:
                raise ValueError("stub writer: no responses")
            out = seq[min(len(self.calls) - 1, len(seq) - 1)]
        self.last_call = CallStats(latency_ms=self.latency_ms, tokens_in=10, tokens_out=20)
        return out


class StubPlanner:
    """Test planner: same response rules as :class:`StubWriter`, grammar flag ignored."""

    def __init__(
        self,
        responses: Sequence[str] | Callable[[list[ChatMessage]], str],
        *,
        backend_id: str = "stub-planner",
        base_url: str | None = None,
    ) -> None:
        self._responses = responses
        self.backend_id = backend_id
        self.base_url = base_url
        self.calls: list[list[ChatMessage]] = []
        self.last_call: CallStats | None = None

    def plan(self, messages: list[ChatMessage]) -> str:
        self.calls.append(list(messages))
        if callable(self._responses):
            out = self._responses(messages)
        else:
            seq = list(self._responses)
            if not seq:
                raise ValueError("stub planner: no responses")
            out = seq[min(len(self.calls) - 1, len(seq) - 1)]
        self.last_call = CallStats(latency_ms=1.0)
        return out


# --------------------------------------------------------------------------- cache


class CacheMode(StrEnum):
    LIVE = "live"  # always call, then store (overwrites)
    RESUME = "resume"  # replay if stored, else call and store
    CACHED = "cached"  # replay only; a missing entry raises CacheMiss
    OFF = "off"  # call, store nothing


class CacheMiss(LookupError):
    """A needed call is not in the cache (``--cached``). Message: backend id + sha only."""

    def __init__(self, backend_id: str, sha: str) -> None:
        super().__init__(f"cache miss: backend={backend_id} prompt_sha={sha[:16]}")
        self.backend_id = backend_id
        self.sha = sha


@dataclass
class _Store:
    cache_dir: Path
    mode: CacheMode
    hits: int = 0
    misses: int = 0
    writes: int = 0
    _ensured: bool = field(default=False, repr=False)

    def key(
        self,
        backend_id: str,
        messages: Sequence[ChatMessage],
        grammar: bool | None,
        renderer_fp: str | None = None,
    ) -> str:
        inner = prompt_sha(messages, grammar=grammar)
        material = f"{backend_id}\n{inner}"
        if renderer_fp is not None:
            material += f"\nrenderer={renderer_fp}"
        return hashlib.sha256(material.encode()).hexdigest()

    def path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self.path(key)
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None  # a corrupt entry is treated as missing
        return data if isinstance(data, dict) and "output" in data else None

    def put(self, key: str, record: dict[str, Any]) -> None:
        if not self._ensured:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._ensured = True
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False, sort_keys=True, indent=1)
            os.replace(tmp, self.path(key))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self.writes += 1


def _call_cached(
    store: _Store,
    backend_id: str,
    messages: list[ChatMessage],
    grammar: bool | None,
    call: Callable[[], tuple[str, CallStats | None]],
    renderer_fp: str | None = None,
) -> tuple[str, CallStats]:
    key = store.key(backend_id, messages, grammar, renderer_fp)
    if store.mode in (CacheMode.RESUME, CacheMode.CACHED):
        hit = store.get(key)
        if hit is not None:
            store.hits += 1
            stats = CallStats(
                latency_ms=float(hit.get("latency_ms") or 0.0),
                tokens_in=hit.get("tokens_in"),
                tokens_out=hit.get("tokens_out"),
                cached=True,
            )
            return str(hit["output"]), stats
        if store.mode is CacheMode.CACHED:
            store.misses += 1
            raise CacheMiss(backend_id, key)
    out, maybe_stats = call()
    stats = maybe_stats or CallStats(latency_ms=0.0)
    if store.mode is not CacheMode.OFF:
        store.put(
            key,
            {
                "backend_id": backend_id,
                "grammar": grammar,
                "prompt_sha": prompt_sha(messages, grammar=grammar),
                "renderer_fingerprint": renderer_fp,
                "output": out,
                "latency_ms": stats.latency_ms,
                "tokens_in": stats.tokens_in,
                "tokens_out": stats.tokens_out,
            },
        )
    return out, stats


class CachedWriter:
    """Caching wrapper around any :class:`WriterBackend` (same id and base_url). A backend
    with a ``renderer`` attribute (C/D) gets the renderer fingerprint in its cache key."""

    def __init__(self, inner: WriterBackend, cache_dir: Path, mode: CacheMode) -> None:
        self.inner = inner
        self.backend_id = inner.backend_id
        self.base_url = inner.base_url
        renderer = getattr(inner, "renderer", None)
        self.renderer_fp: str | None = None if renderer is None else renderer_fingerprint(renderer)
        self.store = _Store(cache_dir, mode)
        self.last_call: CallStats | None = None

    def generate(self, messages: list[ChatMessage], *, grammar: bool) -> str:
        def call() -> tuple[str, CallStats | None]:
            out = self.inner.generate(messages, grammar=grammar)
            return out, self.inner.last_call

        out, stats = _call_cached(
            self.store, self.backend_id, messages, grammar, call, self.renderer_fp
        )
        self.last_call = stats
        return out


class CachedPlanner:
    """Caching wrapper around any :class:`PlannerBackend`."""

    def __init__(self, inner: PlannerBackend, cache_dir: Path, mode: CacheMode) -> None:
        self.inner = inner
        self.backend_id = inner.backend_id
        self.base_url = inner.base_url
        self.store = _Store(cache_dir, mode)
        self.last_call: CallStats | None = None

    def plan(self, messages: list[ChatMessage]) -> str:
        def call() -> tuple[str, CallStats | None]:
            out = self.inner.plan(messages)
            return out, self.inner.last_call

        out, stats = _call_cached(self.store, self.backend_id, messages, None, call)
        self.last_call = stats
        return out
