"""Sentry agent tracing, dev/eval only (PRD §10, §15; NFR-P4, NFR-P7).

Prod sends no telemetry. :func:`init_tracing` returns ``False`` and does nothing (it does not
even import ``sentry_sdk``) unless ``SAIR_ENV`` is ``dev`` or ``eval`` **and** a DSN is set
**and** the DSN host is in ``ALLOWED_OUTBOUND_HOSTS``. Every other function here is a no-op
until tracing is enabled, so the graphs and the app can call them unconditionally.

What leaves the process when enabled is decided by allowlists, not denylists:

* span/transaction names come from a fixed set of graph and node names (:data:`NODE_NAMES`);
* attributes go through :func:`set_attrs`, which keeps only :data:`ALLOWED_ATTRS` keys and only
  values that are numbers, booleans, enum codes or IDs matching :data:`ID_RE` (text is dropped);
* ``before_send`` / ``before_send_transaction`` rebuild every event from an allowlist
  (:func:`scrub_event`): no messages, no exception values (type only), no frame variables, no
  request, user, breadcrumbs, extra, server name or unknown contexts/tags/span data.

Integration point (for ``hub/graphs`` and ``hub/app.py``)::

    from hub import tracing
    tracing.init_tracing(settings)                      # once, at app startup
    deps = MissionDeps(..., span=tracing.node_span)     # SpanFn: one span per node
    with tracing.graph_transaction("mission", mission_id=None):
        plan_mission(group, date, deps)
    with tracing.graph_transaction("journal", walk_id=walk_id):
        run_journal(walk_id, raw, group, mission_id, deps)
    # inside a node, after a model call (stats from hub.models):
    tracing.llm_span(model, tokens_in=stats.prompt_tokens, tokens_out=stats.completion_tokens,
                     latency_ms=stats.latency_s * 1000)
    tracing.set_attrs(regen_count=1, reason_codes=[ReasonCode.UNSUPPORTED_DETAIL])

``unsupported_details`` (words) has no key in the allowlist and can never be sent.
"""

from __future__ import annotations

import fnmatch
import math
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any, cast
from urllib.parse import urlparse

from hub.schemas import (
    EntryOrigin,
    MissionResult,
    MissionSource,
    MissionType,
    ReasonCode,
    WalkStatus,
    WriterBackend,
)
from hub.settings import Settings

__all__ = [
    "ALLOWED_ATTRS",
    "GRAPH_NAMES",
    "ID_RE",
    "NODE_NAMES",
    "SPAN_OPS",
    "clean_attrs",
    "graph_transaction",
    "init_tracing",
    "is_enabled",
    "llm_span",
    "node_span",
    "scrub_event",
    "scrub_transaction",
    "set_attrs",
    "shutdown_tracing",
]

ENABLED_ENVS = frozenset({"dev", "eval"})

GRAPH_NAMES = frozenset({"journal", "mission"})
_JOURNAL_NODES = (
    "ingest", "transcribe", "write_entry", "ground_check", "regen", "template", "store",
    "plan_next_mission", "notify",
)  # fmt: skip
_MISSION_NODES = (
    "recall", "plan", "mission_check", "regen", "fallback", "seed", "speak", "cache",
)  # fmt: skip
NODE_NAMES: frozenset[str] = frozenset(
    {*_JOURNAL_NODES, *(f"mission_{n}" for n in _MISSION_NODES), "llm"}
)
SPAN_OPS = frozenset({"graph", "graph.node", "llm"})
_FALLBACK_DESC = "node"

# IDs are uuid4 hex (hub.db.new_id) or dashed uuids. Nothing else is an "id".
ID_RE = re.compile(
    r"\A(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\Z"
)
_HEX_ID_RE = re.compile(r"\A[0-9a-f]{16}(?:[0-9a-f]{16})?\Z")  # sentry event/trace/span ids
_EXC_TYPE_RE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]{0,99}\Z")
_MODULE_RE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_.]{0,199}\Z")
_FILENAME_RE = re.compile(r"\A[A-Za-z0-9_./<>-]{1,200}\Z")
_SDK_RE = re.compile(r"\A[a-z0-9._-]{1,64}\Z")
_ORIGIN_RE = re.compile(r"\A(?:manual|auto(?:\.[a-z_]+){0,4})\Z")
_TS_RE = re.compile(r"\A[0-9T:.+Z-]{10,40}\Z")
_LEVELS = frozenset({"debug", "info", "warning", "error", "fatal"})
_STATUSES = frozenset({"ok", "internal_error", "cancelled", "unknown_error", "deadline_exceeded"})

_INT_KEYS = frozenset(
    {"tokens_in", "tokens_out", "regen_count", "facts_count", "verified_count", "chunk_count",
     "retry_count"}
)  # fmt: skip
_FLOAT_KEYS = frozenset({"latency_ms", "audio_duration_s"})
_BOOL_KEYS = frozenset({"check_passed"})
_ID_KEYS = frozenset({"walk_id", "mission_id"})
_ENUM_KEYS: dict[str, frozenset[str]] = {
    "backend": frozenset(e.value for e in WriterBackend) | {"template"},
    "origin": frozenset(e.value for e in EntryOrigin) | frozenset(e.value for e in MissionSource),
    "status": frozenset(e.value for e in WalkStatus) | _STATUSES,
    "mission_type": frozenset(e.value for e in MissionType),
    "mission_result": frozenset(e.value for e in MissionResult),
}
_REASON_VALUES = frozenset(e.value for e in ReasonCode)
_STATIC_MODELS = frozenset({"gemma", "llama", "template", "kokoro", "piper", "whisper"})

ALLOWED_ATTRS: frozenset[str] = frozenset(
    {"node", "model", "reason_codes", *_INT_KEYS, *_FLOAT_KEYS, *_BOOL_KEYS, *_ID_KEYS,
     *_ENUM_KEYS}
)  # fmt: skip


@dataclass
class _State:
    sdk: ModuleType | None = None
    models: frozenset[str] = _STATIC_MODELS


_STATE = _State()


# ------------------------------------------------------------------ value checks


def _enum_value(v: object) -> object:
    return v.value if isinstance(v, Enum) else v


def _clean_value(key: str, value: object) -> object | None:
    """The allowlisted, text-free form of ``value`` for ``key``; None means drop."""
    v = _enum_value(value)
    if key in _BOOL_KEYS:
        return v if isinstance(v, bool) else None
    if key in _INT_KEYS:
        return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None
    if key in _FLOAT_KEYS:
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            return None
        return round(float(v), 3) if v >= 0 else None
    if key in _ID_KEYS:
        return v if isinstance(v, str) and ID_RE.match(v) else None
    if key == "node":
        return v if isinstance(v, str) and v in NODE_NAMES else None
    if key == "model":
        return v if isinstance(v, str) and v in _STATE.models else None
    if key in _ENUM_KEYS:
        return v if isinstance(v, str) and v in _ENUM_KEYS[key] else None
    if key == "reason_codes":
        if isinstance(value, str | bytes) or not isinstance(value, Sequence | frozenset | set):
            return None
        codes = [str(_enum_value(c)) for c in value if _enum_value(c) in _REASON_VALUES]
        return sorted(set(codes)) if codes else None
    return None


def clean_attrs(attrs: Mapping[str, object]) -> dict[str, object]:
    """Keep only allowlisted keys with text-free values (pure; usable when disabled)."""
    out: dict[str, object] = {}
    for key, value in attrs.items():
        if not isinstance(key, str) or key not in ALLOWED_ATTRS:
            continue
        cleaned = _clean_value(key, value)
        if cleaned is not None:
            out[key] = cleaned
    return out


def _ts_ok(v: object) -> bool:
    if isinstance(v, bool):
        return False
    return isinstance(v, int | float) or (isinstance(v, str) and _TS_RE.match(v) is not None)


def _desc(name: object) -> str:
    return name if isinstance(name, str) and name in NODE_NAMES | GRAPH_NAMES else _FALLBACK_DESC


# ------------------------------------------------------------------ scrubbers


def _scrub_frame(frame: object) -> dict[str, object] | None:
    if not isinstance(frame, Mapping):
        return None
    out: dict[str, object] = {}
    fn = frame.get("function")
    if isinstance(fn, str) and _EXC_TYPE_RE.match(fn):
        out["function"] = fn
    mod = frame.get("module")
    if isinstance(mod, str) and _MODULE_RE.match(mod):
        out["module"] = mod
    filename = frame.get("filename")
    if isinstance(filename, str) and _FILENAME_RE.match(filename):
        out["filename"] = filename
    lineno = frame.get("lineno")
    if isinstance(lineno, int) and not isinstance(lineno, bool):
        out["lineno"] = lineno
    if isinstance(frame.get("in_app"), bool):
        out["in_app"] = frame["in_app"]
    return out  # never vars, abs_path, context_line, pre/post_context


def _scrub_exception(exc: object) -> dict[str, object] | None:
    if not isinstance(exc, Mapping):
        return None
    values = exc.get("values")
    if not isinstance(values, list):
        return None
    out_values: list[dict[str, object]] = []
    for item in values:
        if not isinstance(item, Mapping):
            continue
        etype = item.get("type")
        rec: dict[str, object] = {
            "type": etype if isinstance(etype, str) and _EXC_TYPE_RE.match(etype) else "Exception"
        }
        mod = item.get("module")
        if isinstance(mod, str) and _MODULE_RE.match(mod):
            rec["module"] = mod
        mech = item.get("mechanism")
        if isinstance(mech, Mapping):
            m: dict[str, object] = {}
            if isinstance(mech.get("handled"), bool):
                m["handled"] = mech["handled"]
            mtype = mech.get("type")
            m["type"] = mtype if isinstance(mtype, str) and _SDK_RE.match(mtype) else "generic"
            rec["mechanism"] = m
        st = item.get("stacktrace")
        if isinstance(st, Mapping) and isinstance(st.get("frames"), list):
            frames = [f for f in (_scrub_frame(x) for x in st["frames"]) if f]
            if frames:
                rec["stacktrace"] = {"frames": frames}
        out_values.append(rec)  # never "value" (the message)
    return {"values": out_values} if out_values else None


def _scrub_trace_like(span: object, *, top: bool) -> dict[str, object] | None:
    """A span (or the ``contexts.trace`` record) reduced to ids, timing, op, name and data."""
    if not isinstance(span, Mapping):
        return None
    out: dict[str, object] = {}
    for key in ("trace_id", "span_id", "parent_span_id"):
        v = span.get(key)
        if isinstance(v, str) and _HEX_ID_RE.match(v):
            out[key] = v
    for key in ("start_timestamp", "timestamp"):
        v = span.get(key)
        if _ts_ok(v):
            out[key] = v
    if isinstance(span.get("same_process_as_parent"), bool):
        out["same_process_as_parent"] = span["same_process_as_parent"]
    op = span.get("op")
    if op is not None:
        out["op"] = op if isinstance(op, str) and op in SPAN_OPS else "span"
    desc = span.get("description")
    if not top or desc is not None:
        out["description"] = _desc(desc)
    status = span.get("status")
    if isinstance(status, str) and status in _STATUSES:
        out["status"] = status
    origin = span.get("origin")
    if isinstance(origin, str) and _ORIGIN_RE.match(origin):
        out["origin"] = origin
    data = span.get("data")
    if isinstance(data, Mapping):
        out["data"] = clean_attrs(data)
    tags = span.get("tags")
    if isinstance(tags, Mapping):
        out["tags"] = clean_attrs(tags)
    return out


def scrub_event(event: Mapping[str, Any], hint: object = None) -> dict[str, Any]:
    """Rebuild a Sentry event from an allowlist. Anything not listed below is dropped."""
    out: dict[str, Any] = {}
    eid = event.get("event_id")
    if isinstance(eid, str) and _HEX_ID_RE.match(eid):
        out["event_id"] = eid
    for key in ("timestamp", "start_timestamp"):
        v = event.get(key)
        if _ts_ok(v):
            out[key] = v
    if event.get("type") == "transaction":
        out["type"] = "transaction"
        out["transaction"] = _desc(event.get("transaction"))
        out["transaction_info"] = {"source": "custom"}
    out["platform"] = "python"
    level = event.get("level")
    if isinstance(level, str) and level in _LEVELS:
        out["level"] = level
    env = event.get("environment")
    if isinstance(env, str) and env in ENABLED_ENVS:
        out["environment"] = env
    sdk = event.get("sdk")
    if isinstance(sdk, Mapping):
        out["sdk"] = {
            k: sdk[k] for k in ("name", "version") if isinstance(sdk.get(k), str)
            and _SDK_RE.match(sdk[k])
        }  # fmt: skip
    exc = _scrub_exception(event.get("exception"))
    if exc is not None:
        out["exception"] = exc
    contexts = event.get("contexts")
    if isinstance(contexts, Mapping):
        trace = _scrub_trace_like(contexts.get("trace"), top=True)
        if trace is not None:
            out["contexts"] = {"trace": trace}
    tags = event.get("tags")
    if isinstance(tags, Mapping):
        cleaned = clean_attrs(tags)
        if cleaned:
            out["tags"] = cleaned
    spans = event.get("spans")
    if isinstance(spans, list):
        out["spans"] = [s for s in (_scrub_trace_like(x, top=False) for x in spans) if s]
    # Dropped on purpose: message, logentry, logger, culprit, request, user, server_name,
    # extra, breadcrumbs, modules, threads, debug_meta, measurements, release, dist,
    # fingerprint, attachments.
    return out


def scrub_transaction(event: Mapping[str, Any], hint: object = None) -> dict[str, Any]:
    """``before_send_transaction``: the same allowlist as :func:`scrub_event`."""
    return scrub_event(event, hint)


def _drop_breadcrumb(crumb: object, hint: object) -> None:
    return None


# ------------------------------------------------------------------ init


def _host_allowed(dsn: str, allowed: Sequence[str]) -> bool:
    try:
        host = urlparse(dsn).hostname
    except ValueError:
        return False
    if not host:
        return False
    return any(fnmatch.fnmatchcase(host, pat.strip().lower()) for pat in allowed if pat.strip())


def init_tracing(settings: Settings, *, transport: object | None = None) -> bool:
    """Enable Sentry tracing for dev/eval runs; return whether it is on (NFR-P7).

    Refuses (returns False, imports nothing, sends nothing) unless ``SAIR_ENV`` is dev/eval, a
    DSN is set and its host is in ``ALLOWED_OUTBOUND_HOSTS``. ``transport`` is for tests (a
    fake ``sentry_sdk.transport.Transport``); production passes nothing.
    """
    dsn = settings.sentry_dsn.get_secret_value().strip()
    if settings.sair_env not in ENABLED_ENVS or not dsn:
        return False
    if not _host_allowed(dsn, settings.allowed_outbound_hosts):
        return False

    import sentry_sdk  # lazy: prod never imports the SDK

    env = settings.sentry_environment
    environment = env if env in ENABLED_ENVS else settings.sair_env
    kwargs: dict[str, Any] = {}
    if transport is not None:
        kwargs["transport"] = transport
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        default_integrations=False,
        auto_enabling_integrations=False,
        integrations=[],
        include_local_variables=False,
        max_request_body_size="never",
        send_default_pii=False,
        include_source_context=False,
        attach_stacktrace=False,
        max_breadcrumbs=0,
        before_send=cast(Any, scrub_event),
        before_send_transaction=cast(Any, scrub_transaction),
        before_breadcrumb=cast(Any, _drop_breadcrumb),
        server_name="",
        release=None,
        auto_session_tracking=False,
        send_client_reports=False,
        enable_logs=False,
        profiles_sample_rate=None,
        **kwargs,
    )
    _STATE.sdk = sentry_sdk
    _STATE.models = _STATIC_MODELS | {settings.ollama_model, settings.writer_model_path.stem}
    return True


def is_enabled() -> bool:
    return _STATE.sdk is not None


def shutdown_tracing() -> None:
    """Flush and detach the client (tests, app shutdown). No-op when disabled."""
    sdk = _STATE.sdk
    if sdk is None:
        return
    client = sdk.get_client()
    if client.is_active():
        client.close(timeout=2.0)
    sdk.get_global_scope().set_client(None)
    sdk.get_isolation_scope().set_client(None)
    sdk.get_current_scope().set_client(None)
    _STATE.sdk = None
    _STATE.models = _STATIC_MODELS


# ------------------------------------------------------------------ spans


def _apply(span: Any, attrs: Mapping[str, object]) -> None:
    for key, value in clean_attrs(attrs).items():
        span.set_data(key, value)


def set_attrs(**attrs: object) -> None:
    """Attach allowlisted attributes to the current span; text values are dropped."""
    sdk = _STATE.sdk
    if sdk is None:
        return
    span = sdk.get_current_span()
    if span is not None:
        _apply(span, attrs)


@contextmanager
def _graph_tx(name: str, ids: Mapping[str, object]) -> Iterator[None]:
    sdk = _STATE.sdk
    assert sdk is not None
    with sdk.start_transaction(op="graph", name=_desc(name), source="custom") as tx:
        _apply(tx, ids)
        try:
            yield
        except BaseException as exc:
            tx.set_status("internal_error")
            sdk.capture_exception(exc)  # scrubbed to the exception type
            raise


def graph_transaction(
    name: str, *, walk_id: str | None = None, mission_id: str | None = None
) -> AbstractContextManager[None]:
    """One transaction per graph run (``journal`` or ``mission``)."""
    if _STATE.sdk is None:
        return nullcontext()
    return _graph_tx(name, {"walk_id": walk_id, "mission_id": mission_id})


@contextmanager
def _node(name: str) -> Iterator[None]:
    sdk = _STATE.sdk
    assert sdk is not None
    with sdk.start_span(op="graph.node", name=_desc(name)) as span:
        _apply(span, {"node": name})
        try:
            yield
        except BaseException:
            span.set_status("internal_error")
            raise


def node_span(name: str) -> AbstractContextManager[None]:
    """One span per graph node. Matches ``hub.graphs.common.SpanFn``; also a decorator."""
    if _STATE.sdk is None:
        return nullcontext()
    return _node(name)


def llm_span(
    model: str,
    *,
    tokens_in: int | None,
    tokens_out: int | None,
    latency_ms: float,
) -> None:
    """Record a finished model call as a child of the current span (model, tokens, latency)."""
    sdk = _STATE.sdk
    if sdk is None:
        return
    parent = sdk.get_current_span()
    if parent is None:
        return
    ms = latency_ms if isinstance(latency_ms, int | float) and math.isfinite(latency_ms) else 0.0
    end = time.time()
    child = parent.start_child(op="llm", name="llm", start_timestamp=end - max(0.0, ms) / 1000)
    _apply(child, {"model": model, "tokens_in": tokens_in, "tokens_out": tokens_out,
                   "latency_ms": ms})  # fmt: skip
    child.finish(end_timestamp=end)
