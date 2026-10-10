"""NFR-P7: no Sentry in prod. Tracing turns on only for dev/eval with a DSN (fake transport)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from hub import tracing
from hub.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
# Loopback discard port: even if the fake transport were bypassed nothing leaves the machine.
FAKE_DSN = "http://publickey@127.0.0.1:9/0"


class CapturingTransport:
    """Built lazily so this module does not import sentry_sdk at collection time."""

    @staticmethod
    def make() -> Any:
        from sentry_sdk.transport import Transport

        class _Cap(Transport):
            def __init__(self) -> None:
                super().__init__()
                self.envelopes: list[Any] = []

            def capture_envelope(self, envelope: Any) -> None:
                self.envelopes.append(envelope)

        return _Cap()


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    tracing.shutdown_tracing()
    yield
    tracing.shutdown_tracing()


def _client_active() -> bool:
    import sentry_sdk

    return bool(sentry_sdk.get_client().is_active())


def test_prod_with_dsn_is_refused_by_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAIR_ENV", "prod")
    monkeypatch.setenv("SENTRY_DSN", FAKE_DSN)
    with pytest.raises(ValidationError) as err:
        Settings(_env_file=None)  # type: ignore[call-arg]
    assert "SENTRY_DSN must be empty in prod" in str(err.value)
    assert FAKE_DSN not in str(err.value)


def test_prod_never_initialises_even_if_settings_were_bypassed() -> None:
    # model_construct skips the validator: init_tracing must refuse on its own.
    s = Settings.model_construct(sair_env="prod", sentry_dsn=SecretStr(FAKE_DSN))
    assert tracing.init_tracing(s, transport=CapturingTransport.make()) is False
    assert tracing.is_enabled() is False
    assert _client_active() is False


@pytest.mark.parametrize("env", ["prod", "test"])
def test_non_dev_envs_return_false(env: str) -> None:
    s = Settings.model_construct(sair_env=env, sentry_dsn=SecretStr(FAKE_DSN))
    assert tracing.init_tracing(s) is False
    assert _client_active() is False


@pytest.mark.parametrize("env", ["dev", "eval"])
def test_dev_eval_without_dsn_return_false(env: str) -> None:
    s = Settings(_env_file=None, sair_env=env)  # type: ignore[call-arg]
    assert tracing.init_tracing(s) is False
    assert _client_active() is False


def test_eval_dsn_host_outside_allowlist_is_refused() -> None:
    s = Settings(  # type: ignore[call-arg]
        _env_file=None,
        sair_env="eval",
        sentry_dsn=SecretStr("https://k@o1.ingest.example.org/1"),
    )
    assert tracing.init_tracing(s, transport=CapturingTransport.make()) is False
    assert _client_active() is False


def test_eval_with_dsn_enables_with_fake_transport() -> None:
    cap = CapturingTransport.make()
    s = Settings(_env_file=None, sair_env="eval", sentry_dsn=SecretStr(FAKE_DSN))  # type: ignore[call-arg]
    assert tracing.init_tracing(s, transport=cap) is True
    assert tracing.is_enabled() and _client_active()
    with tracing.graph_transaction("mission"), tracing.node_span("mission_recall"):
        pass
    tracing.shutdown_tracing()
    assert _client_active() is False
    kinds = [item.headers.get("type") for env in cap.envelopes for item in env.items]
    assert kinds == ["transaction"]


def test_wildcard_allowlist_matches_regional_ingest_host() -> None:
    s = Settings(  # type: ignore[call-arg]
        _env_file=None,
        sair_env="dev",
        sentry_dsn=SecretStr("https://k@o123.ingest.us.sentry.io/1"),
        allowed_outbound_hosts="127.0.0.1,localhost,*.sentry.io",
    )
    assert tracing.init_tracing(s, transport=CapturingTransport.make()) is True


def test_disabled_api_is_noop_and_never_imports_sentry_sdk() -> None:
    """In a fresh interpreter: prod settings, every API call, and still no sentry_sdk module."""
    code = textwrap.dedent(
        f"""
        import sys
        from pydantic import SecretStr
        from hub import tracing
        from hub.settings import Settings
        for env in ("prod", "test"):
            s = Settings.model_construct(sair_env=env, sentry_dsn=SecretStr({FAKE_DSN!r}))
            assert tracing.init_tracing(s) is False
        with tracing.graph_transaction("journal", walk_id="0" * 32):
            with tracing.node_span("transcribe"):
                tracing.llm_span("gemma", tokens_in=1, tokens_out=2, latency_ms=3.0)
                tracing.set_attrs(regen_count=1)
        tracing.shutdown_tracing()
        assert "sentry_sdk" not in sys.modules, "sentry_sdk imported while disabled"
        print("ok")
        """
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("SENTRY_")}
    env.update({"SAIR_ENV": "prod", "SENTRY_DSN": FAKE_DSN})
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "ok"
