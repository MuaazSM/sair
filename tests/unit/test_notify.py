"""FR-O1 / FR-O4: ntfy notifications are metadata only."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
import yaml
from pydantic import ValidationError

from hub.notify import NtfyNotifier, NullNotifier, build_health_body, build_walk_body
from hub.schemas import MissionResult
from hub.settings import Settings

URL = "http://127.0.0.1:8090/topic123"
P = "Sair · Mom + Dad walked · "
ROOT = Path(__file__).resolve().parents[2]


def _settings(**kw: object) -> Settings:
    base: dict[str, object] = {
        "sair_env": "test",
        "ntfy_url": "http://127.0.0.1:8090",
        "ntfy_topic": "topic123",
        "sair_device_groups": "mom:Mom and Dad,dad:Mom and Dad,x:Mom,y:Dad",
    }
    base.update(kw)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("facts", "result", "expected"),
    [
        (0, MissionResult.NOT_MENTIONED, P + "0 things noticed · mission not mentioned"),
        (1, MissionResult.PARTLY, P + "1 thing noticed · mission partly"),
        (2, MissionResult.DONE, P + "2 things noticed · mission done"),
        (3, MissionResult.NOT_DONE, P + "3 things noticed · mission not done"),
    ],
)
@respx.mock
def test_walk_done_body_exact(facts: int, result: MissionResult, expected: str) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200))
    NtfyNotifier(_settings()).walk_done("Mom and Dad", facts, result)
    req = route.calls.last.request
    assert req.content.decode("utf-8") == expected
    assert req.headers["Title"] == "Sair"
    assert req.headers["Tags"] == "walking"
    assert not any(h.lower().startswith("x-firebase") for h in req.headers)


@respx.mock
def test_walk_done_single_person() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200))
    NtfyNotifier(_settings()).walk_done("Mom", 1, MissionResult.DONE)
    assert route.calls.last.request.content.decode() == (
        "Sair · Mom walked · 1 thing noticed · mission done"
    )


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("Mom", "Check in with Mom"),
        ("Dad", "Check in with Dad"),
        ("Mom and Dad", "Check in with Mom and Dad"),
    ],
)
@respx.mock
def test_health_ping_fixed_body(group: str, expected: str) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200))
    NtfyNotifier(_settings()).health_ping(group)
    assert route.calls.last.request.content.decode() == expected


def test_content_cannot_be_passed() -> None:
    marker = "CANARY-7f3a"
    groups = {"Mom and Dad"}
    with pytest.raises(TypeError):
        build_walk_body("Mom and Dad", marker, MissionResult.DONE, groups)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_walk_body("Mom and Dad", 1, marker, groups)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_walk_body("Mom and Dad", True, MissionResult.DONE, groups)
    with pytest.raises(TypeError):
        build_walk_body(marker.encode(), 1, MissionResult.DONE, groups)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_walk_body(marker, 1, MissionResult.DONE, groups)
    with pytest.raises(ValueError):
        build_health_body(marker, groups)
    with pytest.raises(TypeError):
        build_health_body(1, groups)  # type: ignore[arg-type]


@respx.mock
def test_unconfigured_group_is_refused_before_any_request() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200))
    with pytest.raises(ValueError):
        NtfyNotifier(_settings()).health_ping("CANARY-7f3a")
    assert not route.called


@respx.mock
@pytest.mark.parametrize(
    "effect",
    [httpx.ConnectError("boom CANARY"), httpx.ReadTimeout("slow"), httpx.Response(500)],
)
def test_failures_do_not_raise(effect: object) -> None:
    if isinstance(effect, httpx.Response):
        respx.post(URL).mock(return_value=effect)
    else:
        respx.post(URL).mock(side_effect=effect)
    n = NtfyNotifier(_settings())
    n.walk_done("Mom and Dad", 2, MissionResult.DONE)
    n.health_ping("Mom")


@respx.mock
def test_no_topic_sends_nothing() -> None:
    route = respx.post("http://127.0.0.1:8090/").mock(return_value=httpx.Response(200))
    NtfyNotifier(_settings(ntfy_topic=None)).walk_done("Mom", 1, MissionResult.DONE)
    assert not route.called


def test_prod_refuses_non_loopback_ntfy_url() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            sair_env="prod",
            ntfy_url="https://ntfy.sh",
            ntfy_topic="t",
        )  # type: ignore[call-arg]


def test_null_notifier_is_inert() -> None:
    n = NullNotifier()
    n.walk_done("Mom", 1, MissionResult.DONE)
    n.health_ping("Mom")


def test_health_keywords_cover_three_scripts() -> None:
    data = yaml.safe_load((ROOT / "data" / "health_keywords.yaml").read_text(encoding="utf-8"))
    for script in ("en", "hinglish", "devanagari"):
        assert data["keywords"][script], script


def test_ntfy_client_ignores_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """NFR-P1/P3: HTTP(S)_PROXY / ALL_PROXY can never route a notification off the loopback."""
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(var, "http://proxy.invalid:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    client = NtfyNotifier(_settings())._client
    assert client.trust_env is False
    assert client._mounts == {}
    assert client._transport_for_url(httpx.URL(URL)) is client._transport
