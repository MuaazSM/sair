"""NFR-P2 / NFR-P7 / NFR-P1: settings refuse unsafe configuration at startup."""

from __future__ import annotations

import datetime as dt
import os

import pytest
from pydantic import ValidationError

from hub import settings as settings_mod
from hub.settings import Settings, apply_offline_env, get_settings, reset_settings


def make(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_host_accepted(host: str) -> None:
    assert make(sair_host=host).sair_host == host


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com", "::", ""])
def test_non_loopback_host_refused(host: str) -> None:
    with pytest.raises(ValidationError) as ei:
        make(sair_host=host)
    assert "loopback" in str(ei.value)
    assert "input_value" not in str(ei.value)


def test_host_refused_in_every_env() -> None:
    for env in ("dev", "eval", "test", "prod"):
        with pytest.raises(ValidationError):
            make(sair_env=env, sair_host="0.0.0.0")


@pytest.mark.parametrize(
    "field,url",
    [
        ("ollama_host", "http://192.168.1.9:11434"),
        ("writer_url", "http://example.com:8091"),
        ("ntfy_url", "https://ntfy.sh"),
        ("ollama_host", "not a url"),
    ],
)
def test_non_loopback_model_url_refused_in_prod(field: str, url: str) -> None:
    with pytest.raises(ValidationError):
        make(sair_env="prod", **{field: url})


def test_loopback_urls_accepted_in_prod() -> None:
    s = make(sair_env="prod", writer_url="http://localhost:8091", ntfy_url="http://[::1]:8090")
    assert s.sair_env == "prod"


def test_prod_with_sentry_dsn_refused() -> None:
    with pytest.raises(ValidationError) as ei:
        make(sair_env="prod", sentry_dsn="https://key@o1.ingest.sentry.io/1")
    assert "key@" not in str(ei.value)


def test_sentry_enabled_only_dev_eval_with_dsn() -> None:
    dsn = "https://key@o1.ingest.sentry.io/1"
    synthetic = "~/.sair-eval"  # NFR-P7: Sentry never runs against the prod data dir
    assert make(sair_env="eval", sentry_dsn=dsn, sair_data_dir=synthetic).sentry_enabled
    assert make(sair_env="dev", sentry_dsn=dsn, sair_data_dir=synthetic).sentry_enabled
    assert not make(sair_env="eval").sentry_enabled
    assert not make(sair_env="test", sentry_dsn=dsn).sentry_enabled
    assert not make(sair_env="prod").sentry_enabled


def test_allowed_outbound_hosts_parsed_and_prod_restricted() -> None:
    s = make(sair_env="eval", allowed_outbound_hosts="127.0.0.1, localhost,*.sentry.io")
    assert s.allowed_outbound_hosts == ["127.0.0.1", "localhost", "*.sentry.io"]
    assert make(
        sair_env="prod", allowed_outbound_hosts="127.0.0.1,localhost"
    ).allowed_outbound_hosts
    with pytest.raises(ValidationError):
        make(sair_env="prod", allowed_outbound_hosts="127.0.0.1,*.sentry.io")


def test_device_groups_parsing() -> None:
    s = make(sair_device_groups="mom:Mom and Dad,dad:Mom and Dad, aunt : Auntie ")
    assert s.device_groups == {"mom": "Mom and Dad", "dad": "Mom and Dad", "aunt": "Auntie"}
    with pytest.raises(ValidationError):
        make(sair_device_groups="mom")


def test_tokens_never_in_repr_or_str() -> None:
    s = make(sair_token_mom="tok-mom-SECRET", sair_token_owner="tok-owner-SECRET")
    for text in (repr(s), str(s), s.model_dump_json()):
        assert "SECRET" not in text
    assert s.token_for("mom") == "tok-mom-SECRET"
    assert s.token_for("dad") == ""
    assert s.token_for("nobody") == ""


def test_offline_env_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    s = make()
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setenv(k, "0")
    apply_offline_env(s)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_prod_refuses_online_hf() -> None:
    with pytest.raises(ValidationError):
        make(sair_env="prod", hf_hub_offline=False)


def test_time_and_tz_parsing() -> None:
    s = make(
        mission_pregen_time="16:30", mission_pregen_block="11:30-16:00", sair_tz="Asia/Kolkata"
    )
    assert s.mission_pregen_time == dt.time(16, 30)
    assert s.mission_pregen_block == (dt.time(11, 30), dt.time(16, 0))
    assert str(s.tz) == "Asia/Kolkata"
    with pytest.raises(ValidationError):
        make(sair_tz="Mars/Olympus")
    with pytest.raises(ValidationError):
        make(mission_pregen_block="1130")


def test_defaults_and_path_expansion() -> None:
    s = make()
    assert s.sair_port == 8787 and s.audio_max_s == 90 and s.transient_retry_window_s == 600
    assert not str(s.data_dir).startswith("~")
    assert s.db_path == s.data_dir / "sair.db"
    assert s.tmp_dir == s.data_dir / "tmp"
    assert s.audio_dir == s.data_dir / "audio"
    assert s.ollama_model == "gemma4:e2b" and s.writer_max_tokens == 400


def test_test_env_loads_no_env_file(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    d = Path(str(tmp_path))
    (d / ".env").write_text("SAIR_PORT=9999\n")
    monkeypatch.chdir(d)
    monkeypatch.setenv("SAIR_ENV", "test")
    reset_settings()
    try:
        assert get_settings().sair_port == 8787
        assert get_settings() is get_settings()
        # Same dir, prod: the file is used.
        monkeypatch.setenv("SAIR_ENV", "prod")
        reset_settings()
        assert get_settings().sair_port == 9999
    finally:
        reset_settings()


def test_eval_layers_env_eval(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    d = Path(str(tmp_path))
    (d / ".env").write_text("SAIR_PORT=9001\n")
    (d / ".env.eval").write_text(
        f"SAIR_PORT=9002\nSENTRY_DSN=https://k@x.ingest.sentry.io/1\nSAIR_DATA_DIR={d / 'eval'}\n"
    )
    monkeypatch.chdir(d)
    monkeypatch.setenv("SAIR_ENV", "eval")
    assert settings_mod.load_settings().sair_port == 9002
    assert settings_mod.load_settings().sentry_enabled
    monkeypatch.setenv("SAIR_ENV", "prod")
    assert settings_mod.load_settings().sair_port == 9001


# ---------------------------------------- NFR-P7: Sentry only on synthetic or fixture data

DSN = "https://key@o1.ingest.sentry.io/1"


@pytest.fixture
def home(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.mark.usefixtures("home")
@pytest.mark.parametrize("env", ["dev", "eval"])
def test_sentry_refused_on_prod_default_data_dir(env: str) -> None:
    with pytest.raises(ValidationError) as ei:
        make(sair_env=env, sentry_dsn=DSN)  # data dir defaults to ~/.sair
    msg = str(ei.value)
    assert settings_mod.MSG_SENTRY_DATA in msg
    assert "key@" not in msg and ".sair" not in msg.replace(settings_mod.MSG_SENTRY_DATA, "")


@pytest.mark.usefixtures("home")
@pytest.mark.parametrize(
    "field,value",
    [
        ("sair_db_path", "~/.sair/eval.db"),
        ("sair_tmp_dir", "~/.sair/tmp"),
        ("sair_audio_dir", "~/.sair/audio"),
        ("sair_data_dir", "~/.sair/sub"),
    ],
)
def test_sentry_refused_when_any_store_is_under_prod_data_dir(field: str, value: str) -> None:
    kw: dict[str, object] = {"sair_data_dir": "~/.sair-eval", field: value}
    with pytest.raises(ValidationError) as ei:
        make(sair_env="eval", sentry_dsn=DSN, **kw)
    assert settings_mod.MSG_SENTRY_DATA in str(ei.value)


@pytest.mark.usefixtures("home")
def test_sentry_allowed_on_sair_eval_dir() -> None:
    s = make(
        sair_env="eval",
        sentry_dsn=DSN,
        sair_data_dir="~/.sair-eval",
        sair_db_path="~/.sair-eval/eval.db",
        sair_tmp_dir="~/.sair-eval/tmp",
    )
    assert s.sentry_enabled


@pytest.mark.usefixtures("home")
def test_prod_data_dir_fine_when_sentry_off() -> None:
    assert not make(sair_env="eval").sentry_enabled  # no DSN: no rule
    assert make(sair_env="prod").data_dir.name == ".sair"
    assert not make(sair_env="test", sentry_dsn=DSN).sentry_enabled


def test_eval_example_points_away_from_prod_data_dir() -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / ".env.eval.example").read_text()
    lines = dict(
        ln.split("#", 1)[0].strip().split("=", 1)
        for ln in text.splitlines()
        if "=" in ln.split("#", 1)[0]
    )
    for key in ("SAIR_DATA_DIR", "SAIR_DB_PATH", "SAIR_TMP_DIR"):
        value = lines[key].strip()
        assert value == "~/.sair-eval" or value.startswith("~/.sair-eval/")
    for key in ("SAIR_TOKEN_MOM", "SAIR_TOKEN_DAD"):
        assert lines.get(key, "").strip() == ""
