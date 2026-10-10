"""Shared HTTP plumbing for the model clients: CallStats and error mapping (NFR-P4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from hub.models.errors import Permanent, Transient
from hub.schemas import ReasonCode

__all__ = ["CallStats", "post_json"]

_TRANSIENT_4XX = frozenset({408, 425, 429})


@dataclass(frozen=True)
class CallStats:
    """Metrics for one model call. Counts and timings only, never text."""

    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    load_s: float | None = None
    hit_limit: bool = False


def post_json(
    client: httpx.Client,
    url: str,
    body: dict[str, Any],
    *,
    timeout: float,
    transient: ReasonCode,
    permanent: ReasonCode,
    component: str,
) -> dict[str, Any]:
    """POST and decode JSON. Maps failures to Transient/Permanent with fixed messages."""
    try:
        r = client.post(url, json=body, timeout=timeout)
    except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
        raise Transient(transient, component) from None
    except httpx.HTTPError:
        raise Permanent(permanent, component) from None
    if r.status_code >= 500 or r.status_code in _TRANSIENT_4XX:
        raise Transient(transient, component)
    if r.status_code >= 400:
        raise Permanent(permanent, component)
    try:
        data = r.json()
    except ValueError:
        raise Permanent(permanent, component) from None
    if not isinstance(data, dict):
        raise Permanent(permanent, component)
    return data
