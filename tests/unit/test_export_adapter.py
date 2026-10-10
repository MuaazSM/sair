"""training/export_adapter.py with fakes: preflight, step order, idempotency, key conversion.

No model is loaded, no server is contacted, nothing is downloaded.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from training import export_adapter as ea

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    65536.
Pages active:                                 153834.
Pages inactive:                               131072.
Pages speculative:                              2048.
Pages purgeable:                                  0.
"""


@pytest.fixture
def listener() -> Iterator[int]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        yield s.getsockname()[1]
    finally:
        s.close()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --------------------------------------------------------------------------- preflight


def test_preflight_refuses_when_a_port_is_open(listener: int, capsys: Any) -> None:
    with pytest.raises(ea.Refused, match=f"port {listener}"):
        ea.preflight(ports={listener: "llama-server (writer)"}, ps_lines=[], vm_stat_text=VM_STAT)
    assert "free_mb=1024" in capsys.readouterr().out


def test_preflight_ok_when_quiet(capsys: Any) -> None:
    pf = ea.preflight(ports={_free_port(): "x"}, ps_lines=[], vm_stat_text=VM_STAT)
    assert pf.ok and pf.memory == {"free_mb": 1024, "available_est_mb": 3104}
    assert "open_ports=[]" in capsys.readouterr().out


def test_preflight_refuses_on_model_processes() -> None:
    lines = [
        "  101 /opt/homebrew/bin/ollama /opt/homebrew/bin/ollama serve",
        "  102 /opt/homebrew/Cellar/ollama/x/ollama-runner runner --model blob",
        "  103 /opt/homebrew/bin/llama-server llama-server -m m.gguf --port 8091",
        "  104 /usr/bin/python3.12 python -m hub.models._tts_child",
        "  105 /bin/zsh zsh -c grep ollama llama-server kokoro",  # not a server
        "  106 /usr/bin/python3 python -m pytest tests/unit",
    ]
    found = ea.model_processes(lines)
    assert [p for p, _ in found] == [101, 102, 103, 104]
    with pytest.raises(ea.Refused, match="pid 104"):
        ea.preflight(ports={}, ps_lines=lines, vm_stat_text=VM_STAT)


def test_preflight_min_free_memory() -> None:
    with pytest.raises(ea.Refused, match="available_est_mb=3104"):
        ea.preflight(ports={}, ps_lines=[], vm_stat_text=VM_STAT, min_free_gb=8)


def test_default_ports_cover_the_model_servers() -> None:
    assert set(ea.DEFAULT_PORTS) == {11434, 8091, 8092}


# --------------------------------------------------------------------------- fake pipeline


class FakeRunner:
    """Pretends to be convert_hf_to_gguf.py / llama-quantize / tinker: writes the outputs."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str]] = []

    def __call__(self, cmd: Sequence[str], env: Mapping[str, str]) -> int:
        cmd = list(cmd)
        self.calls.append(cmd)
        self.envs.append(dict(env))
        if "--outfile" in cmd:
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(
                b"GGUF-f16" + (Path(cmd[2]) / "model.safetensors").read_bytes()
            )
        elif cmd[0].endswith("llama-quantize"):
            Path(cmd[2]).write_bytes(b"GGUF-q4" + Path(cmd[1]).read_bytes())
        elif "checkpoint" in cmd:
            target = ea.adapter_dir_for(cmd[3], Path(cmd[5]))
            target.mkdir(parents=True, exist_ok=True)
            (target / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 32}))
            _fake_safetensors(target / "adapter_model.safetensors")
            (target / "checkpoint_complete").write_text("")
        return 0

    def kinds(self) -> list[str]:
        out = []
        for c in self.calls:
            out.append(
                "convert"
                if "--outfile" in c
                else "quantize"
                if c[0].endswith("llama-quantize")
                else "tinker"
            )
        return out


def _fake_safetensors(path: Path) -> None:
    hdr = json.dumps(
        {
            "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [2, 4],
                "data_offsets": [0, 32],
            }
        }
    ).encode()
    path.write_bytes(len(hdr).to_bytes(8, "little") + hdr + b"\0" * 32)


def _adapter(tmp: Path) -> Path:
    d = tmp / "adapters" / "run_sampler_weights_e1"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text(json.dumps({"r": 32}))
    _fake_safetensors(d / "adapter_model.safetensors")
    (d / "checkpoint_complete").write_text("")
    return d


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    conv_py = tmp_path / "convert_hf_to_gguf.py"
    conv_py.write_text("")
    py = tmp_path / "python"
    py.write_text("")
    monkeypatch.setattr(ea, "CONVERT_PY", conv_py)
    monkeypatch.setattr(ea, "CONVERT_PYTHON", py)
    monkeypatch.setattr(ea.shutil, "which", lambda name: f"/fake/{name}")
    snap = tmp_path / "hf-snapshot"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"base")
    merges: list[Path] = []
    checks: list[int] = []

    def impl(adapter: Path, snapshot: Path, out: Path, base_id: str) -> None:
        merges.append(out)
        out.mkdir(parents=True)
        (out / "config.json").write_text("{}")
        (out / "model.safetensors").write_bytes(
            b"merged" + (adapter / "adapter_config.json").read_bytes()
        )

    return {
        "tmp": tmp_path,
        "runner": FakeRunner(),
        "impl": impl,
        "merges": merges,
        "snapshot": lambda base_id: snap,
        "check": lambda: checks.append(1),
        "checks": checks,
        "paths": {
            "merged": tmp_path / "out" / "sair-writer-hf",
            "f16": tmp_path / "out" / "w-f16.gguf",
            "q4": tmp_path / "out" / "w-q4.gguf",
        },
    }


def _all(env: dict[str, Any], adapter: Path, **kw: Any) -> Path:
    (env["tmp"] / "out").mkdir(exist_ok=True)
    return ea.run_all(
        adapter,
        runner=env["runner"],
        impl=env["impl"],
        snapshot=env["snapshot"],
        check=env["check"],
        **env["paths"],
        **kw,
    )


def test_step_order_and_outputs(env: dict[str, Any]) -> None:
    adapter = _adapter(env["tmp"])
    q4 = _all(env, adapter)
    assert q4.read_bytes().startswith(b"GGUF-q4")
    assert len(env["merges"]) == 1
    assert env["runner"].kinds() == ["convert", "quantize"]
    conv = env["runner"].calls[0]
    assert conv[conv.index("--outtype") + 1] == "f16" and "--no-mtp" in conv
    assert env["runner"].envs[0]["HF_HUB_OFFLINE"] == "1"
    assert env["runner"].calls[1][-1] == "Q4_K_M"
    assert len(env["checks"]) == 3  # preflight before merge, convert, quantize
    for p in env["paths"].values():
        st = ea.read_stamp(p)
        assert st is not None and len(st["output"]["sha256"]) == 64
    assert not any(p.name.endswith(".partial") for p in (env["tmp"] / "out").iterdir())


def test_idempotent_rerun_skips_everything(env: dict[str, Any], capsys: Any) -> None:
    adapter = _adapter(env["tmp"])
    _all(env, adapter)
    capsys.readouterr()
    _all(env, adapter)
    out = capsys.readouterr().out
    assert len(env["merges"]) == 1 and len(env["runner"].calls) == 2
    assert out.count("skipped (up to date)") == 3
    assert "sha256=" in out and "bytes=" in out


def test_changed_input_reruns_downstream(env: dict[str, Any]) -> None:
    adapter = _adapter(env["tmp"])
    _all(env, adapter)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 64}))
    _all(env, adapter)
    assert len(env["merges"]) == 2
    assert env["runner"].kinds() == ["convert", "quantize", "convert", "quantize"]


def test_output_tampered_or_force_reruns(env: dict[str, Any]) -> None:
    adapter = _adapter(env["tmp"])
    _all(env, adapter)
    env["paths"]["q4"].write_bytes(b"truncated")
    _all(env, adapter)
    assert env["runner"].kinds() == ["convert", "quantize", "quantize"]
    _all(env, adapter, force=True)
    assert len(env["merges"]) == 2


def test_steps_refuse_out_of_order(env: dict[str, Any]) -> None:
    t = env["tmp"]
    with pytest.raises(ea.Refused, match="run merge first"):
        ea.convert(t / "nope", t / "x.gguf", runner=env["runner"], check=env["check"])
    with pytest.raises(ea.Refused, match="run convert first"):
        ea.quantize(t / "nope.gguf", t / "q.gguf", runner=env["runner"], check=env["check"])
    empty = t / "no-weights"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    with pytest.raises(ea.Refused, match="safetensors"):
        ea.convert(empty, t / "x.gguf", runner=env["runner"], check=env["check"])
    bad = t / "incomplete-adapter"
    bad.mkdir()
    with pytest.raises(ea.Refused, match="PEFT adapter"):
        ea.merge(bad, impl=env["impl"], snapshot=env["snapshot"], check=env["check"])
    assert env["runner"].calls == [] and env["merges"] == []


def test_merge_refused_by_preflight_writes_nothing(env: dict[str, Any], listener: int) -> None:
    adapter = _adapter(env["tmp"])

    def check() -> None:
        ea.preflight(ports={listener: "ollama"}, ps_lines=[], vm_stat_text=VM_STAT, quiet=True)

    out = env["tmp"] / "merged"
    with pytest.raises(ea.Refused, match="model servers are running"):
        ea.merge(adapter, out=out, impl=env["impl"], snapshot=env["snapshot"], check=check)
    assert env["merges"] == [] and not out.exists()


def test_base_uses_the_same_convert_and_quantize(env: dict[str, Any]) -> None:
    t = env["tmp"]
    ea.base(
        f16=t / "b-f16.gguf",
        out=t / "b-q4.gguf",
        runner=env["runner"],
        snapshot=env["snapshot"],
        check=env["check"],
    )
    assert env["runner"].kinds() == ["convert", "quantize"]
    c = env["runner"].calls[0]
    assert "--no-mtp" in c and c[c.index("--outtype") + 1] == "f16"
    ea.base(
        f16=t / "b-f16.gguf",
        out=t / "b-q4.gguf",
        runner=env["runner"],
        snapshot=env["snapshot"],
        check=env["check"],
    )
    assert len(env["runner"].calls) == 2  # idempotent


def test_require_weights_message(tmp_path: Path) -> None:
    with pytest.raises(ea.Refused, match="hf download Qwen/Qwen3.5-4B"):
        ea.require_weights(tmp_path)


# --------------------------------------------------------------------------- download


def test_download_path_validation_and_idempotency(env: dict[str, Any]) -> None:
    root = env["tmp"] / "adapters"
    with pytest.raises(ea.Refused, match="expected tinker://"):
        ea.download("https://example.org/x", out_root=root, runner=env["runner"])
    path = "tinker://abc-123/sampler_weights/sft-r1-e2"
    d = ea.download(path, out_root=root, runner=env["runner"])
    assert d.name == "abc-123_sampler_weights_sft-r1-e2" and ea.adapter_complete(d)
    cmd = env["runner"].calls[0]
    assert cmd[1:4] == ["checkpoint", "download", path] and cmd[4:6] == ["--output", str(root)]
    ea.download(path, out_root=root, runner=env["runner"])
    assert len(env["runner"].calls) == 1


def test_tinker_path_from_run(tmp_path: Path) -> None:
    (tmp_path / "r.json").write_text(
        json.dumps(
            {
                "epochs": [
                    {"epoch": 1, "sampler_path": "tinker://a/sampler_weights/r-e1"},
                    {"epoch": 2, "sampler_path": "tinker://a/sampler_weights/r-e2"},
                ]
            }
        )
    )
    assert ea.tinker_path_from_run("r", 2, tmp_path) == "tinker://a/sampler_weights/r-e2"
    with pytest.raises(ea.Refused):
        ea.tinker_path_from_run("r", 3, tmp_path)


# --------------------------------------------------------------------------- adapter keys

BASE_SHAPES = {
    "model.language_model.layers.0.self_attn.q_proj": (8, 4),
    "model.language_model.layers.0.mlp.up_proj": (16, 4),
    "model.language_model.layers.1.linear_attn.in_proj_qkv": (12, 4),
    "model.visual.blocks.0.attn.qkv": (12, 4),
    "lm_head": (100, 4),
}


def _keys(mod: str) -> list[str]:
    return [f"base_model.model.{mod}.lora_A.weight", f"base_model.model.{mod}.lora_B.weight"]


def test_convert_keys_tinker_text_names_to_vlm_names() -> None:
    keys = (
        _keys("model.layers.0.self_attn.q_proj")
        + _keys("model.layers.0.mlp.up_proj")
        + _keys("model.layers.1.linear_attn.in_proj_qkv")
        + _keys("lm_head")
    )
    shapes = {
        k: (
            [2, 4]
            if "lora_A" in k
            else [
                BASE_SHAPES[
                    ea._resolve_module(
                        k.split(".lora_")[0].removeprefix("base_model.model."), sorted(BASE_SHAPES)
                    )
                    or ""
                ][0],
                2,
            ]
        )
        for k in keys
    }
    km = ea.convert_adapter_keys(keys, BASE_SHAPES, shapes)
    assert km.modules == sorted(
        [
            "lm_head",
            "model.language_model.layers.0.mlp.up_proj",
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.1.linear_attn.in_proj_qkv",
        ]
    )
    assert km.mapping["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"] == (
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight"
    )
    cfg = ea.converted_config({"r": 32, "lora_alpha": 32}, km.modules, "Qwen/Qwen3.5-4B")
    assert cfg["target_modules"] == km.modules and cfg["lora_dropout"] == 0.0


def test_convert_keys_refusals() -> None:
    with pytest.raises(ea.AdapterFormatError, match="unmapped_modules"):
        ea.convert_adapter_keys(_keys("model.layers.0.linear_attn.in_proj_qkvz"), BASE_SHAPES)
    with pytest.raises(ea.AdapterFormatError, match="incomplete_lora_pair"):
        ea.convert_adapter_keys(_keys("lm_head")[:1], BASE_SHAPES)
    with pytest.raises(ea.AdapterFormatError, match="unrecognised"):
        ea.convert_adapter_keys(["base_model.model.lm_head.lora_magnitude_vector"], BASE_SHAPES)
    with pytest.raises(ea.AdapterFormatError, match="shape_mismatch"):
        ks = _keys("lm_head")
        ea.convert_adapter_keys(ks, BASE_SHAPES, {ks[0]: [2, 4], ks[1]: [99, 2]})
    with pytest.raises(ea.AdapterFormatError, match="unsupported_use_dora"):
        ea.converted_config({"r": 8, "use_dora": True}, [], "b")


def test_describe_adapter_prints_no_content(env: dict[str, Any], capsys: Any) -> None:
    info = ea.describe_adapter(_adapter(env["tmp"]))
    assert info["r"] == 32 and info["tensors"] == 1 and len(info["sha256"]) == 64
    assert "lora_A" not in capsys.readouterr().out
