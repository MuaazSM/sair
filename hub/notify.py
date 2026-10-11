"""Self-hosted ntfy client (FR-O1, FR-O4; NFR-P3, NFR-P4).

Metadata only. A notification body is built from exactly three kinds of input: a group
name that must be one of the configured groups, an ``int`` count and a ``MissionResult``
enum. No transcript, entry, mission or fact text can reach this module's sinks.

ntfy upstream: with ``NTFY_IOS_UPSTREAM=on`` the self-hosted server is expected to be run
with ``upstream-base-url`` set, so iOS can be woken by Apple's push relay. Per the ntfy
docs the server then forwards only a poll request carrying the message ID (and a topic
hash), never the message body or title; the phone then polls this server over the
tailnet. This client sends nothing extra in that mode and never sets any ``X-Firebase``
header. The default is off and the plists do not configure an upstream.

Failures never raise into the caller; the only log line is ``ntfy_failed`` with a code.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Protocol

import httpx
import structlog

from hub.schemas import MissionResult
from hub.settings import Settings

__all__ = [
    "NotifierProtocol",
    "NtfyNotifier",
    "NullNotifier",
    "build_health_body",
    "build_walk_body",
]

log = structlog.get_logger("hub.notify")

TITLE = "Sair"
TAGS = "walking"
_TIMEOUT_S = 5.0

_RESULT_WORDS: dict[MissionResult, str] = {
    MissionResult.DONE: "done",
    MissionResult.PARTLY: "partly",
    MissionResult.NOT_DONE: "not done",
    MissionResult.NOT_MENTIONED: "not mentioned",
}
# FR-O4 wording for the three groups; anything else is refused.
_HEALTH_NAMES: dict[str, str] = {"Mom": "Mom", "Dad": "Dad", "Mom and Dad": "Mom and Dad"}


class NotifierProtocol(Protocol):
    def walk_done(self, group: str, facts: int, mission_result: MissionResult) -> None: ...

    def health_ping(self, group: str) -> None: ...


def _check_group(group: object, allowed: Collection[str]) -> str:
    if not isinstance(group, str):
        raise TypeError("group must be str")
    if group not in allowed:
        raise ValueError("group is not a configured group")
    return group


def build_walk_body(
    group: str, facts: int, mission_result: MissionResult, allowed_groups: Collection[str]
) -> str:
    """``Sair · Mom + Dad walked · 2 things noticed · mission done`` (PRD J3.5)."""
    g = _check_group(group, allowed_groups)
    if isinstance(facts, bool) or not isinstance(facts, int):
        raise TypeError("facts must be int")
    if not isinstance(mission_result, MissionResult):
        raise TypeError("mission_result must be MissionResult")
    if facts < 0:
        raise ValueError("facts must be >= 0")
    noun = "thing" if facts == 1 else "things"
    who = g.replace(" and ", " + ")
    return f"Sair · {who} walked · {facts} {noun} noticed · mission {_RESULT_WORDS[mission_result]}"


def build_health_body(group: str, allowed_groups: Collection[str]) -> str:
    """``Check in with Mom`` / ``Dad`` / ``Mom and Dad`` (FR-O4). Never any walk content."""
    g = _check_group(group, allowed_groups)
    name = _HEALTH_NAMES.get(g)
    if name is None:
        raise ValueError("group has no health wording")
    return f"Check in with {name}"


class NullNotifier:
    """Does nothing; for tests and dev."""

    def walk_done(self, group: str, facts: int, mission_result: MissionResult) -> None:
        return None

    def health_ping(self, group: str) -> None:
        return None


class NtfyNotifier:
    def __init__(self, settings: Settings, *, client: httpx.Client | None = None) -> None:
        self._url = f"{settings.ntfy_url.rstrip('/')}/{settings.ntfy_topic or ''}"
        self._enabled = bool(settings.ntfy_topic)
        self._groups: frozenset[str] = frozenset(settings.device_groups.values())
        self._upstream = settings.ntfy_ios_upstream
        self._client = client or httpx.Client(timeout=_TIMEOUT_S, trust_env=False)  # NFR-P1/P3

    @property
    def upstream_enabled(self) -> bool:
        return self._upstream == "on"

    def walk_done(self, group: str, facts: int, mission_result: MissionResult) -> None:
        body = build_walk_body(group, facts, mission_result, self._groups)
        self._send(body, priority="default")

    def health_ping(self, group: str) -> None:
        body = build_health_body(group, self._groups)
        self._send(body, priority="high")

    def _send(self, body: str, *, priority: str) -> None:
        if not self._enabled:
            log.warning("ntfy_failed", code="no_topic")
            return
        headers = {"Title": TITLE, "Tags": TAGS, "Priority": priority}
        try:
            resp = self._client.post(self._url, content=body.encode("utf-8"), headers=headers)
            if resp.status_code >= 400:
                log.warning("ntfy_failed", code=f"http_{resp.status_code}")
        except httpx.TimeoutException:
            log.warning("ntfy_failed", code="timeout")
        except httpx.HTTPError:
            log.warning("ntfy_failed", code="network")
        except Exception:  # never raise into the caller
            log.warning("ntfy_failed", code="unknown")
