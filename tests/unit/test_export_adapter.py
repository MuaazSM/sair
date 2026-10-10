"""training/export_adapter.py: preflight, step order, idempotency, key mapping, merge math.

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


# ------------------------------------------------------------------------ adapter keys + merge math

# Tiny Gated DeltaNet geometry: key_dim = 2*2 = 4, value_dim = 2*4 = 8 -> fused qkv rows 16.
TEXT_CFG = {
    "linear_key_head_dim": 2,
    "linear_num_key_heads": 2,
    "linear_value_head_dim": 2,
    "linear_num_value_heads": 4,
}
H, R = 3, 2
ACFG = {"r": R, "lora_alpha": 4, "use_rslora": False}  # scale 2
BASE_SHAPES = {
    "model.language_model.embed_tokens.weight": [10, H],
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": [16, H],
    "model.language_model.layers.0.linear_attn.in_proj_z.weight": [8, H],
    "model.language_model.layers.0.linear_attn.out_proj.weight": [H, 8],
    "model.language_model.layers.1.self_attn.q_proj.weight": [8, H],
    "model.language_model.layers.1.mlp.up_proj.weight": [5, H],
    "mtp.layers.0.mlp.up_proj.weight": [5, H],
    "model.visual.blocks.0.attn.qkv.weight": [12, H],
}
MODULES = {  # adapter module -> (lora B rows, lora A in)
    "model.layers.0.linear_attn.in_proj_q": (4, H),
    "model.layers.0.linear_attn.in_proj_k": (4, H),
    "model.layers.0.linear_attn.in_proj_v": (8, H),
    "model.layers.0.linear_attn.in_proj_z": (8, H),
    "model.layers.0.linear_attn.out_proj": (H, 8),
    "model.layers.1.self_attn.q_proj": (8, H),
    "model.layers.1.mlp.up_proj": (5, H),
    "model.unembed_tokens": (10, H),
}


def _keys(mod: str) -> list[str]:
    return [f"base_model.model.{mod}.lora_A.weight", f"base_model.model.{mod}.lora_B.weight"]


def _shapes(mods: Mapping[str, tuple[int, int]] = MODULES) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for mod, (rows, in_f) in mods.items():
        a, b = _keys(mod)
        out[a], out[b] = [R, in_f], [rows, R]
    return out


def test_fused_slices_match_the_qwen35_4b_config() -> None:
    cfg = {
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_value_head_dim": 128,
        "linear_num_value_heads": 32,
    }
    assert ea.fused_slices(cfg) == {
        "in_proj_q": (0, 2048),
        "in_proj_k": (2048, 2048),
        "in_proj_v": (4096, 4096),
    }
    assert ea.fused_slices({}) == {}


def test_plan_maps_split_fused_and_unembed_names() -> None:
    plan = ea.plan_merge(_shapes(), BASE_SHAPES, ACFG, TEXT_CFG)
    t = {x.module: x for x in plan.targets}
    qkv = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    assert [
        (
            t[f"model.layers.0.linear_attn.in_proj_{p}"].row0,
            t[f"model.layers.0.linear_attn.in_proj_{p}"].rows,
        )
        for p in "qkv"
    ] == [(0, 4), (4, 4), (8, 8)]
    assert {t[f"model.layers.0.linear_attn.in_proj_{p}"].tensor for p in "qkv"} == {qkv}
    assert t["model.layers.0.linear_attn.in_proj_z"].tensor.endswith("linear_attn.in_proj_z.weight")
    # text-LM names go to the language model, never to the mtp head or the vision tower
    assert (
        t["model.layers.1.mlp.up_proj"].tensor == "model.language_model.layers.1.mlp.up_proj.weight"
    )
    head = t["model.unembed_tokens"]
    assert (
        head.tensor == "lm_head.weight"
        and head.source == "model.language_model.embed_tokens.weight"
    )
    assert plan.untie_lm_head and plan.scale == 2.0 and len(plan.targets) == len(MODULES)


def test_plan_keeps_an_untied_lm_head() -> None:
    base = {**BASE_SHAPES, "lm_head.weight": [10, H]}
    plan = ea.plan_merge(_shapes({"lm_head": (10, H)}), base, ACFG, TEXT_CFG)
    assert plan.targets[0].tensor == "lm_head.weight" and plan.targets[0].source is None
    assert not plan.untie_lm_head


def test_plan_refusals() -> None:
    with pytest.raises(ea.AdapterFormatError, match="unmapped_modules"):
        ea.plan_merge(
            _shapes({"model.layers.0.linear_attn.in_proj_w": (4, H)}), BASE_SHAPES, ACFG, TEXT_CFG
        )
    with pytest.raises(ea.AdapterFormatError, match="incomplete_lora_pair"):
        a = _keys("model.layers.1.mlp.up_proj")[0]
        ea.plan_merge({a: [R, H]}, BASE_SHAPES, ACFG, TEXT_CFG)
    with pytest.raises(ea.AdapterFormatError, match="unrecognised"):
        ea.plan_merge(
            {"base_model.model.lm_head.lora_magnitude_vector": [1]}, BASE_SHAPES, ACFG, TEXT_CFG
        )
    with pytest.raises(ea.AdapterFormatError, match="shape_mismatch"):  # v slice is 8 rows
        ea.plan_merge(
            _shapes({"model.layers.0.linear_attn.in_proj_v": (4, H)}), BASE_SHAPES, ACFG, TEXT_CFG
        )
    with pytest.raises(ea.AdapterFormatError, match="shape_mismatch"):  # rank != r
        a, b = _keys("model.layers.1.mlp.up_proj")
        ea.plan_merge({a: [3, H], b: [5, 3]}, BASE_SHAPES, ACFG, TEXT_CFG)
    with pytest.raises(ea.AdapterFormatError, match="duplicate_target"):  # fused + split overlap
        mods = {
            "model.layers.0.linear_attn.in_proj_qkv": (16, H),
            "model.layers.0.linear_attn.in_proj_k": (4, H),
        }
        ea.plan_merge(_shapes(mods), BASE_SHAPES, ACFG, TEXT_CFG)
    with pytest.raises(ea.AdapterFormatError, match="fused_layout_mismatch"):
        cfg = {**TEXT_CFG, "linear_num_value_heads": 2}  # q|k|v would be 12 rows, base has 16
        ea.plan_merge(
            _shapes({"model.layers.0.linear_attn.in_proj_q": (4, H)}), BASE_SHAPES, ACFG, cfg
        )
    for bad in (
        {"use_dora": True},
        {"bias": "all"},
        {"modules_to_save": ["x"]},
        {"rank_pattern": {"x": 4}},
    ):
        with pytest.raises(ea.AdapterFormatError, match="unsupported"):
            ea.plan_merge(_shapes(), BASE_SHAPES, {**ACFG, **bad}, TEXT_CFG)


def test_lora_scale() -> None:
    assert ea.lora_scale({"r": 32, "lora_alpha": 32}) == 1.0
    assert ea.lora_scale({"r": 4, "lora_alpha": 16}) == 4.0
    assert ea.lora_scale({"r": 16, "lora_alpha": 16, "use_rslora": True}) == 4.0


def test_fused_slice_merge_equals_split_lora_reference() -> None:
    """base_slice(x) + scale·B(A(x)) == merged_slice(x) for q, k and v; other rows untouched."""
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    w = torch.randn(16, H, generator=g)
    slices = ea.fused_slices(TEXT_CFG)
    lora = {
        p: (torch.randn(R, H, generator=g), torch.randn(n, R, generator=g))
        for p, (_, n) in slices.items()
    }
    scale = 2.0
    merged = ea.apply_deltas(
        w, [(slices[p][0], ea.lora_delta(a, b, scale)) for p, (a, b) in lora.items()]
    )
    x = torch.randn(7, H, generator=g)
    q, k, v = torch.split(x @ merged.T, [4, 4, 8], dim=-1)  # as Qwen3_5GatedDeltaNet.forward
    for part, got in zip(("in_proj_q", "in_proj_k", "in_proj_v"), (q, k, v), strict=True):
        r0, n = slices[part]
        a, b = lora[part]
        ref = x @ w[r0 : r0 + n].T + scale * ((x @ a.T) @ b.T)
        assert torch.allclose(got, ref, atol=1e-5)
    # only k trained: q and v rows bit-identical, k rows changed
    only_k = ea.apply_deltas(w, [(4, ea.lora_delta(*lora["in_proj_k"], scale))])
    assert torch.equal(only_k[:4], w[:4]) and torch.equal(only_k[8:], w[8:])
    assert not torch.equal(only_k[4:8], w[4:8])
    # bf16 in -> bf16 out, accumulated in float32
    wb = w.to(torch.bfloat16)
    mb = ea.apply_deltas(wb, [(0, ea.lora_delta(*lora["in_proj_q"], scale))])
    assert mb.dtype == torch.bfloat16
    exp = (wb.float()[:4] + ea.lora_delta(*lora["in_proj_q"], scale)).to(torch.bfloat16)
    assert torch.equal(mb[:4], exp)


@pytest.mark.parametrize(("r", "alpha", "rslora"), [(4, 4, False), (4, 16, False), (4, 8, True)])
def test_manual_merge_matches_peft_for_plain_linear(r: int, alpha: int, rslora: bool) -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    torch.manual_seed(1)
    model = torch.nn.Sequential()
    model.add_module("proj", torch.nn.Linear(6, 5, bias=False))
    base_w = model.proj.weight.detach().clone()
    cfg = peft.LoraConfig(
        r=r, lora_alpha=alpha, use_rslora=rslora, target_modules=["proj"], init_lora_weights=False
    )
    pm = peft.get_peft_model(model, cfg)
    a = pm.base_model.model.proj.lora_A["default"].weight.detach().clone()
    b = pm.base_model.model.proj.lora_B["default"].weight.detach().clone()
    ref = pm.merge_and_unload().proj.weight.detach()
    scale = ea.lora_scale({"r": r, "lora_alpha": alpha, "use_rslora": rslora})
    got = ea.apply_deltas(base_w, [(0, ea.lora_delta(a, b, scale))])
    assert torch.allclose(got, ref, atol=1e-6)


def _write_snapshot(
    torch: Any, snap: Path, tensors: Mapping[str, Any], cfg: Mapping[str, Any]
) -> None:
    from safetensors.torch import save_file

    snap.mkdir()
    names = sorted(tensors)
    half = len(names) // 2
    wm = {}
    for i, part in enumerate((names[:half], names[half:])):
        fn = f"model-{i + 1:05d}-of-00002.safetensors"
        save_file({n: tensors[n] for n in part}, str(snap / fn))
        wm.update({n: fn for n in part})
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    (snap / "config.json").write_text(json.dumps(cfg))
    (snap / "tokenizer.json").write_text("{}")
    (snap / "chat_template.jinja").write_text("t")


@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
def test_merge_checkpoint_streams_shards_and_unties(tmp_path: Path, dtype: str) -> None:
    torch = pytest.importorskip("torch")
    out_dt = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[dtype]
    from safetensors.torch import load_file, save_file

    g = torch.Generator().manual_seed(2)
    base = {n: torch.randn(*s, generator=g).to(torch.bfloat16) for n, s in BASE_SHAPES.items()}
    base["model.language_model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    cfg = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings": True,
        "text_config": {**TEXT_CFG, "tie_word_embeddings": True},
    }
    snap = tmp_path / "snap"
    _write_snapshot(torch, snap, base, cfg)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps(ACFG))
    lora = {k: torch.randn(*s, generator=g) for k, s in _shapes().items()}
    save_file(lora, str(adapter / "adapter_model.safetensors"))
    out = tmp_path / "merged"
    plan = ea.merge_checkpoint(adapter, snap, out, shard_bytes=64, dtype=dtype)

    idx = json.loads((out / "model.safetensors.index.json").read_text())
    shards = sorted(p.name for p in out.glob("*.safetensors"))
    assert len(shards) > 2 and set(idx["weight_map"].values()) == set(shards)
    got: dict[str, Any] = {}
    for s in shards:
        got.update(load_file(str(out / s)))
    assert set(got) == set(base) | {"lm_head.weight"} == set(idx["weight_map"])

    def delta(mod: str) -> Any:
        a, b = _keys(mod)
        return ea.lora_delta(lora[a], lora[b], plan.scale)

    expect = {n: t.float().clone() for n, t in base.items()}
    qkv = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    expect[qkv][0:4] += delta("model.layers.0.linear_attn.in_proj_q")
    expect[qkv][4:8] += delta("model.layers.0.linear_attn.in_proj_k")
    expect[qkv][8:16] += delta("model.layers.0.linear_attn.in_proj_v")
    for mod in ("in_proj_z", "out_proj"):
        expect[f"model.language_model.layers.0.linear_attn.{mod}.weight"] += delta(
            f"model.layers.0.linear_attn.{mod}"
        )
    expect["model.language_model.layers.1.self_attn.q_proj.weight"] += delta(
        "model.layers.1.self_attn.q_proj"
    )
    expect["model.language_model.layers.1.mlp.up_proj.weight"] += delta(
        "model.layers.1.mlp.up_proj"
    )
    expect["lm_head.weight"] = base["model.language_model.embed_tokens.weight"].float() + delta(
        "model.unembed_tokens"
    )
    for n, t in got.items():
        if n in plan.tensors:  # merged in float32, written once in the requested dtype
            assert t.dtype == out_dt, n
            assert torch.equal(t, expect[n].to(out_dt)), n
        else:
            assert t.dtype == torch.bfloat16 and torch.equal(t, base[n]), n
    # untouched tensors are bit-identical, incl. the tied embeddings, mtp and vision
    for n in (
        "model.language_model.embed_tokens.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.language_model.norm.weight",
    ):
        assert torch.equal(got[n], base[n])
    new_cfg = json.loads((out / "config.json").read_text())
    assert (
        new_cfg["tie_word_embeddings"] is False
        and new_cfg["text_config"]["tie_word_embeddings"] is False
    )
    assert (out / "tokenizer.json").is_file() and (out / "chat_template.jinja").is_file()
    assert not list(out.glob("*.tmp"))


def test_merge_dtype_is_part_of_the_stamp_and_validated(env: dict[str, Any]) -> None:
    adapter = _adapter(env["tmp"])
    out = env["tmp"] / "m"
    kw = {"out": out, "impl": env["impl"], "snapshot": env["snapshot"], "check": env["check"]}
    ea.merge(adapter, **kw)
    assert ea.read_stamp(out)["inputs"]["dtype"] == ea.DEFAULT_MERGE_DTYPE == "f32"  # type: ignore[index]
    ea.merge(adapter, **kw)
    assert len(env["merges"]) == 1  # same dtype: skipped
    ea.merge(adapter, dtype="f16", **kw)
    assert len(env["merges"]) == 2  # dtype change: re-merged
    with pytest.raises(ea.Refused, match="merge dtype"):
        ea.merge(adapter, dtype="int8", **kw)
    with pytest.raises(SystemExit):
        ea.main(["merge", "--adapter", str(adapter), "--dtype", "fp8"])


def test_merged_tiny_qwen35_matches_on_the_fly_lora(tmp_path: Path) -> None:
    """A real (tiny, random) Qwen3.5 text model: merged logits == base + split LoRA hooks."""
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    from safetensors.torch import save_file

    tcfg = tf.Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=4,
        linear_num_key_heads=2,
        linear_value_head_dim=4,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=2,
        tie_word_embeddings=True,
    )
    torch.manual_seed(3)
    model = tf.Qwen3_5ForCausalLM(tcfg).eval()
    snap = tmp_path / "snap"
    model.save_pretrained(snap, safe_serialization=True)
    shapes = {k: list(v[1]["shape"]) for k, v in ea._base_headers(snap).items()}
    assert "lm_head.weight" not in shapes  # tied: not stored
    cfg = json.loads((snap / "config.json").read_text())
    acfg = {"r": 2, "lora_alpha": 4}
    kd, vd = 8, 16
    mods: dict[str, tuple[int, int]] = {
        "model.layers.0.linear_attn.in_proj_q": (kd, 16),
        "model.layers.0.linear_attn.in_proj_k": (kd, 16),
        "model.layers.0.linear_attn.in_proj_v": (vd, 16),
        "model.layers.0.linear_attn.in_proj_z": (vd, 16),
        "model.layers.0.linear_attn.out_proj": (16, vd),
        "model.layers.1.self_attn.q_proj": tuple(shapes["model.layers.1.self_attn.q_proj.weight"]),  # type: ignore[dict-item]
        "model.layers.1.mlp.down_proj": (16, 24),
        "model.unembed_tokens": (64, 16),
    }
    g = torch.Generator().manual_seed(4)
    lora = {}
    for mod, (rows, in_f) in mods.items():
        a, b = _keys(mod)
        lora[a] = torch.randn(2, in_f, generator=g) * 0.3
        lora[b] = torch.randn(rows, 2, generator=g) * 0.3
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps(acfg))
    save_file(lora, str(adapter / "adapter_model.safetensors"))
    plan = ea.plan_merge({k: list(v.shape) for k, v in lora.items()}, shapes, acfg, cfg)
    assert plan.untie_lm_head

    def lo(mod: str, x: Any) -> Any:
        a, b = _keys(mod)
        return plan.scale * ((x @ lora[a].T) @ lora[b].T)

    def add(mod: str) -> Any:
        return lambda m, inp, out: out + lo(mod, inp[0])

    la = model.model.layers[0].linear_attn
    hooks = [
        la.in_proj_qkv.register_forward_hook(
            lambda m, inp, out: (
                out
                + torch.cat(
                    [lo(f"model.layers.0.linear_attn.in_proj_{p}", inp[0]) for p in "qkv"], dim=-1
                )
            )
        ),
        la.in_proj_z.register_forward_hook(add("model.layers.0.linear_attn.in_proj_z")),
        la.out_proj.register_forward_hook(add("model.layers.0.linear_attn.out_proj")),
        model.model.layers[1].self_attn.q_proj.register_forward_hook(
            add("model.layers.1.self_attn.q_proj")
        ),
        model.model.layers[1].mlp.down_proj.register_forward_hook(
            add("model.layers.1.mlp.down_proj")
        ),
        model.lm_head.register_forward_hook(add("model.unembed_tokens")),
    ]
    ids = torch.tensor([[1, 5, 9, 2, 33, 7]])
    with torch.no_grad():
        ref = model(ids).logits
        for h in hooks:
            h.remove()
        base_logits = model(ids).logits
    out = tmp_path / "merged"
    ea.merge_checkpoint(adapter, snap, out)
    merged = tf.Qwen3_5ForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    with torch.no_grad():
        got = merged(ids).logits
    assert (ref - base_logits).abs().max() > 1e-2  # the adapter matters
    assert torch.allclose(got, ref, atol=1e-4), float((got - ref).abs().max())
    assert not torch.equal(merged.lm_head.weight, merged.model.embed_tokens.weight)


def test_describe_adapter_prints_no_content(env: dict[str, Any], capsys: Any) -> None:
    info = ea.describe_adapter(_adapter(env["tmp"]))
    assert info["r"] == 32 and info["tensors"] == 1 and len(info["sha256"]) == 64
    assert "lora_A" not in capsys.readouterr().out
