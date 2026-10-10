"""eval: per-backend renderers for C and D, renderer-aware cache keys, --cache-dir, --set dev."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from eval import config, run
from eval.backends import (
    CachedWriter,
    CacheMiss,
    CacheMode,
    LlamaServerWriter,
    StubWriter,
    _Store,
    prompt_sha,
    renderer_fingerprint,
)
from hub.prompts.builders import ChatMessage
from tests.unit.test_eval_run import GOOD, _items

MSGS = [ChatMessage("system", "S"), ChatMessage("user", "U")]


class RendererC:
    def render(self, messages: Sequence[ChatMessage]) -> str:
        return "C:" + messages[-1].content

    def fingerprint(self) -> str:
        return "fp-c"


class RendererD:
    def render(self, messages: Sequence[ChatMessage]) -> str:
        return "D:" + messages[-1].content

    def fingerprint(self) -> str:
        return "fp-d"


class NoFingerprint:
    def __init__(self, shots: int = 2) -> None:
        self.shots = shots

    def render(self, messages: Sequence[ChatMessage]) -> str:
        return "N"


HERE = "tests.unit.test_eval_renderers"


def _inner(w: Any) -> Any:
    assert isinstance(w, CachedWriter)
    return w.inner


# --------------------------------------------------------------------------- E1


def test_separate_renderers_and_urls_for_c_and_d(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EVAL_QWEN_BASE_URL", raising=False)
    ws = config.build_writers(
        ["qwen_base", "sair_writer"],
        cache_dir=tmp_path,
        mode=CacheMode.RESUME,
        renderer_c=f"{HERE}:RendererC",
        renderer_d=f"{HERE}:RendererD",
    )
    c, d = _inner(ws["qwen_base"]), _inner(ws["sair_writer"])
    assert isinstance(c.renderer, RendererC) and isinstance(d.renderer, RendererD)
    assert c.base_url == "http://127.0.0.1:8092"
    assert d.base_url == "http://127.0.0.1:8091"


def test_legacy_renderer_flag_is_fallback_for_both(tmp_path: Path) -> None:
    ws = config.build_writers(
        ["qwen_base", "sair_writer"],
        cache_dir=tmp_path,
        mode=CacheMode.RESUME,
        renderer_spec=f"{HERE}:RendererC",
        renderer_d=f"{HERE}:RendererD",
    )
    assert isinstance(_inner(ws["qwen_base"]).renderer, RendererC)
    assert isinstance(_inner(ws["sair_writer"]).renderer, RendererD)


def test_defaults_are_training_renderers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import training.renderer_runtime as rr

    ex = [
        rr.FewShotExample("t1", "u1", "a1"),
        rr.FewShotExample("t2", "u2", "a2"),
    ]
    fs = tmp_path / "few_shot.json"
    fs.write_text(
        json.dumps({"examples": [e.__dict__ for e in ex], "sha256": rr._examples_sha(ex)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(rr, "FEW_SHOT_JSON", fs)
    ws = config.build_writers(
        ["qwen_base", "sair_writer"], cache_dir=tmp_path, mode=CacheMode.RESUME
    )
    c, d = _inner(ws["qwen_base"]).renderer, _inner(ws["sair_writer"]).renderer
    assert type(c) is rr.FewShotRenderer
    assert type(d) is rr.RuntimeRenderer


def test_default_c_unavailable_refuses_live_c_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import training.renderer_runtime as rr

    monkeypatch.setattr(rr, "FEW_SHOT_JSON", tmp_path / "missing.json")
    # D alone still works: its default renderer loads
    ws = config.build_writers(["sair_writer"], cache_dir=tmp_path, mode=CacheMode.RESUME)
    assert isinstance(_inner(ws["sair_writer"]).renderer, rr.RuntimeRenderer)
    with pytest.raises(config.ConfigError, match="renderer-c"):
        config.build_writers(["qwen_base"], cache_dir=tmp_path, mode=CacheMode.RESUME)


def test_run_cli_passes_per_backend_specs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_build(keys: Sequence[str], **kw: Any) -> dict[str, Any]:
        seen.update(kw)
        return {k: StubWriter([GOOD]) for k in keys}

    monkeypatch.setattr(config, "build_writers", fake_build)
    items = _items(tmp_path / "t.jsonl")
    code = run.main(
        [
            "--items", str(items), "--systems", "C+,D+", "--resamples", "50",
            "--out", str(tmp_path / "o"), "--renderer-c", "a:b", "--renderer-d", "c:d",
            "--renderer", "e:f",
        ]
    )  # fmt: skip
    assert code == 0
    assert seen["renderer_c"] == "a:b" and seen["renderer_d"] == "c:d"
    assert seen["renderer_spec"] == "e:f"


def test_chains_cli_passes_per_backend_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    from eval import chains

    seen: dict[str, Any] = {}

    class Stop(Exception):
        pass

    def fake_build(keys: Sequence[str], **kw: Any) -> dict[str, Any]:
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(config, "build_writers", fake_build)
    with pytest.raises(Stop):
        chains.main(["--systems", "C,D", "--renderer-c", "a:b", "--renderer-d", "c:d"])
    assert seen["renderer_c"] == "a:b" and seen["renderer_d"] == "c:d"


# --------------------------------------------------------------------------- E2


def test_cache_key_includes_renderer_fingerprint() -> None:
    st = _Store(Path("/nonexistent"), CacheMode.RESUME)
    k_none = st.key("llama:x", MSGS, True)
    k_c = st.key("llama:x", MSGS, True, renderer_fp="fp-c")
    k_d = st.key("llama:x", MSGS, True, renderer_fp="fp-d")
    assert len({k_none, k_c, k_d}) == 3
    # backends without a renderer keep the original key formula (old caches load)
    old = hashlib.sha256(f"llama:x\n{prompt_sha(MSGS, grammar=True)}".encode()).hexdigest()
    assert k_none == old


def test_fingerprint_fallback_is_stable_and_config_sensitive() -> None:
    assert renderer_fingerprint(RendererC()) == "fp-c"
    a, b = renderer_fingerprint(NoFingerprint(2)), renderer_fingerprint(NoFingerprint(2))
    assert a == b and len(a) == 64
    assert renderer_fingerprint(NoFingerprint(3)) != a


def test_changed_renderer_never_replays_stale_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def writer(r: Any, out: str) -> LlamaServerWriter:
        w = LlamaServerWriter("http://127.0.0.1:8092", r, model_id="m")
        monkeypatch.setattr(w, "generate", lambda messages, *, grammar: out)
        return w

    live = CachedWriter(writer(RendererC(), "first"), tmp_path, CacheMode.RESUME)
    assert live.generate(list(MSGS), grammar=True) == "first"
    same = CachedWriter(writer(RendererC(), "second"), tmp_path, CacheMode.CACHED)
    assert same.generate(list(MSGS), grammar=True) == "first"
    other = CachedWriter(writer(RendererD(), "second"), tmp_path, CacheMode.CACHED)
    with pytest.raises(CacheMiss):
        other.generate(list(MSGS), grammar=True)
    rec = json.loads(next(tmp_path.glob("*.json")).read_text("utf-8"))
    assert rec["renderer_fingerprint"] == "fp-c"


def test_old_cache_entry_still_loads_without_renderer(tmp_path: Path) -> None:
    key = hashlib.sha256(f"stub-writer\n{prompt_sha(MSGS, grammar=True)}".encode()).hexdigest()
    (tmp_path / f"{key}.json").write_text(json.dumps({"output": "old"}), encoding="utf-8")
    w = CachedWriter(StubWriter([]), tmp_path, CacheMode.CACHED)
    assert w.generate(list(MSGS), grammar=True) == "old"


# --------------------------------------------------------------------------- E3


def _capture_build(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_build(keys: Sequence[str], **kw: Any) -> dict[str, Any]:
        seen.update(kw)
        return {k: StubWriter([GOOD]) for k in keys}

    monkeypatch.setattr(config, "build_writers", fake_build)
    return seen


def test_cache_dir_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_build(monkeypatch)
    items = _items(tmp_path / "t.jsonl")
    base = ["--items", str(items), "--systems", "D+", "--resamples", "50"]
    assert run.main(base + ["--out", str(tmp_path / "o1")]) == 0
    assert seen["cache_dir"] == run.SYNTHETIC_CACHE
    assert run.main(base + ["--out", str(tmp_path / "o2"), "--cache-dir", str(tmp_path / "c")]) == 0
    assert seen["cache_dir"] == tmp_path / "c"


def test_dev_set_reads_dev_items_and_stays_out_of_eval_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _capture_build(monkeypatch)
    monkeypatch.setattr(run, "DEV_ITEMS", _items(tmp_path / "dev.jsonl"))
    monkeypatch.setattr(run, "DEV_ROOT", tmp_path / "gates")
    assert run.main(["--set", "dev", "--systems", "D+", "--resamples", "50"]) == 0
    assert seen["cache_dir"] == tmp_path / "gates"
    out = tmp_path / "gates" / "eval" / "latest"
    m = json.loads((out / "metrics.json").read_text("utf-8"))
    assert m["set"] == "dev" and m["n_items"] == 6
    assert "set=dev" in capsys.readouterr().out

    code = run.main(["--set", "dev", "--systems", "D+", "--out", str(run.RESULTS / "dev-x")])
    assert code == 2 and "eval/results" in capsys.readouterr().err
    assert not (run.RESULTS / "dev-x").exists()


def test_real_set_refuses_cache_dir_outside_real_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run.main(
        ["--set", "real", "--systems", "A", "--cache-dir", str(tmp_path / "c")],
        writers={"template": StubWriter([GOOD])},
    )
    assert code == 3 and "cache" in capsys.readouterr().err
