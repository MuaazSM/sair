"""Export Sair-Writer: Tinker LoRA adapter -> merged HF bf16 -> GGUF f16 -> Q4_K_M (PRD §12.4).

    # the Tinker key reaches the CLI only through the caller's subshell; never read here
    (set -a; . ./.env.training; set +a; uv run python training/export_adapter.py download \\
        --from-run <run_id> --epoch <k>)
    uv run python training/export_adapter.py preflight          # servers stopped? free memory
    uv run python training/export_adapter.py merge --adapter models/adapters/<dir>
    uv run python training/export_adapter.py convert
    uv run python training/export_adapter.py quantize
    uv run python training/export_adapter.py base               # baseline C, same path
    uv run python training/export_adapter.py all --adapter ...  # merge -> convert -> quantize

Steps
-----
* ``download``: ``tinker checkpoint download <tinker://...> --output models/adapters/`` (the
  tinker CLI from the uv env). The archive is PEFT format: ``adapter_config.json`` +
  ``adapter_model.safetensors`` (or ``.bin``) + ``checkpoint_complete`` (tinker 0.33.1,
  ``tinker/cli/commands/checkpoint.py``). ``--from-run`` takes the epoch's ``sampler_path``
  from ``training/runs/<run>.json``: the exact weights whose dev metrics G-export compares.
* ``merge``: a manual LoRA merge on the CPU, streaming the base safetensors shard by shard
  (no model object, no peft), ``HF_HUB_OFFLINE=1``. Tinker names modules as in the text-only
  causal LM (``model.layers.N...``); the HF snapshot is ``Qwen3_5ForConditionalGeneration``
  (``model.language_model.layers.N...``). :func:`plan_merge` maps every LoRA pair onto one
  base tensor and a row slice: Tinker's split Gated DeltaNet ``in_proj_q``/``_k``/``_v`` go
  into the fused HF ``in_proj_qkv`` rows ``[0,kd)``/``[kd,2kd)``/``[2kd,2kd+vd)`` (layout
  from transformers ``Qwen3_5GatedDeltaNet.forward``; see ``FUSED_PARTS``), and
  ``unembed_tokens`` goes into ``lm_head``. Any unmapped module, shape or rank mismatch,
  overlap, or unsupported option refuses the merge. ΔW = (alpha / r) · B @ A is added in
  float32, then cast to bf16. If the adapter trains the unembedding while the base ties it to
  the embeddings, ``lm_head.weight`` is written as embeddings + ΔW and
  ``tie_word_embeddings`` is set to false (the input embeddings stay untouched). All other
  tensors (incl. ``mtp.*`` and the vision tower) are copied unchanged, like the ``base`` path.
  Output: ``models/sair-writer-hf/`` + the tokenizer + chat template.
* ``convert``: ``vendor/llama.cpp/.venv/bin/python vendor/llama.cpp/convert_hf_to_gguf.py
  --outtype f16 --no-mtp`` -> ``models/sair-writer-f16.gguf``. ``--no-mtp`` on both C and D:
  the writer serves without the ``mtp.*`` draft head, and both conversions must match.
* ``quantize``: ``llama-quantize`` (brew) -> ``models/sair-writer-q4_k_m.gguf``.
* ``base``: the same convert + quantize on the untouched HF snapshot ->
  ``models/qwen3.5-4b-base-q4_k_m.gguf`` (baseline C goes through the identical path).

Every heavy step runs :func:`preflight` first: it refuses while anything listens on the model
ports (11434 Ollama, 8091 writer llama-server, 8092 base llama-server) or while an
ollama / llama-server / Kokoro process is running, and prints free memory.

Idempotency: each output gets a stamp (``<file>.stamp.json``, or ``export_stamp.json`` inside
a directory) holding the input fingerprints and the output sha256 + size. A step whose stamp
matches its inputs and whose output still has the recorded size is skipped (``--force``
redoes it). Printed: paths, sizes, sha256, counts, codes. Never model content, never keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
ADAPTERS_DIR = MODELS / "adapters"
MERGED_DIR = MODELS / "sair-writer-hf"
WRITER_F16 = MODELS / "sair-writer-f16.gguf"
WRITER_Q4 = MODELS / "sair-writer-q4_k_m.gguf"
BASE_F16 = MODELS / "qwen3.5-4b-base-f16.gguf"
BASE_Q4 = MODELS / "qwen3.5-4b-base-q4_k_m.gguf"
RUNS_DIR = ROOT / "training" / "runs"
LLAMA_CPP = ROOT / "vendor" / "llama.cpp"
CONVERT_PY = LLAMA_CPP / "convert_hf_to_gguf.py"
CONVERT_PYTHON = LLAMA_CPP / ".venv" / "bin" / "python"
DEFAULT_BASE_ID = "Qwen/Qwen3.5-4B"
QUANT_TYPE = "Q4_K_M"
OUTTYPE = "f16"

DEFAULT_PORTS: dict[int, str] = {
    11434: "ollama",
    8091: "llama-server (writer)",
    8092: "llama-server (base)",
}
EXE_NAMES = ("ollama", "llama-server")  # executable basename starts with one of these
PY_ARG_MARKERS = ("_tts_child", "kokoro")  # python processes whose args carry these

OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}

Runner = Callable[[Sequence[str], Mapping[str, str]], int]


class Refused(SystemExit):
    """A step refused to run; the message is a fixed code plus paths/ports, never content."""


# --------------------------------------------------------------------------- hashing


def file_digest(path: Path, chunk: int = 8 << 20) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


STAMP_NAME = "export_stamp.json"


def dir_digest(path: Path) -> tuple[str, int]:
    """sha256 over sorted ``relpath\\tsha256`` lines of every file (stamp excluded)."""
    lines: list[str] = []
    total = 0
    for p in sorted(x for x in path.rglob("*") if x.is_file() and x.name != STAMP_NAME):
        sha, n = file_digest(p)
        total += n
        lines.append(f"{p.relative_to(path).as_posix()}\t{sha}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest(), total


def digest(path: Path) -> tuple[str, int]:
    return dir_digest(path) if path.is_dir() else file_digest(path)


def _size(path: Path) -> int:
    if path.is_dir():
        return sum(
            p.stat().st_size for p in path.rglob("*") if p.is_file() and p.name != STAMP_NAME
        )
    return path.stat().st_size


def stamp_path(out: Path) -> Path:
    return (
        out / STAMP_NAME
        if out.is_dir() or out.suffix == ""
        else out.with_name(out.name + ".stamp.json")
    )


def read_stamp(out: Path) -> dict[str, Any] | None:
    sp = stamp_path(out)
    if not sp.is_file():
        return None
    try:
        data = json.loads(sp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_stamp(out: Path, step: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    sha, n = digest(out)
    stamp = {"step": step, "inputs": dict(inputs), "output": {"sha256": sha, "bytes": n}}
    sp = stamp_path(out)
    sp.parent.mkdir(parents=True, exist_ok=True)
    tmp = sp.with_name(sp.name + ".tmp")
    tmp.write_text(json.dumps(stamp, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(sp)
    return stamp


def up_to_date(out: Path, inputs: Mapping[str, Any]) -> bool:
    """Stamp matches the inputs and the output still has the recorded size."""
    if not out.exists():
        return False
    st = read_stamp(out)
    if st is None or st.get("inputs") != dict(inputs):
        return False
    try:
        return int(st["output"]["bytes"]) == _size(out)
    except (KeyError, TypeError, ValueError):
        return False


def fingerprint(path: Path) -> str:
    """Input fingerprint: the stamp's output sha when the input has a stamp, else hashed."""
    st = read_stamp(path)
    if st is not None and up_to_date(path, st.get("inputs", {})):
        return str(st["output"]["sha256"])
    return digest(path)[0]


def report(label: str, out: Path) -> None:
    st = read_stamp(out)
    if st is not None:
        sha, n = st["output"]["sha256"], st["output"]["bytes"]
    else:
        sha, n = digest(out)
    print(
        f"{label}: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} "
        f"bytes={n} ({n / 2**30:.2f} GiB) sha256={sha}"
    )


# --------------------------------------------------------------------------- preflight


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def _ps_lines() -> list[str]:
    out = subprocess.run(
        ["ps", "-axo", "pid=,comm=,args="], capture_output=True, text=True, check=False
    )
    return out.stdout.splitlines()


def model_processes(lines: Iterable[str] | None = None) -> list[tuple[int, str]]:
    """(pid, label) of running model servers. Labels only; full args are never printed."""
    own = {os.getpid(), os.getppid()}
    found: list[tuple[int, str]] = []
    for line in _ps_lines() if lines is None else lines:
        parts = line.strip().split(None, 2)
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        pid, comm = int(parts[0]), parts[1]
        args = parts[2] if len(parts) > 2 else ""
        if pid in own:
            continue
        base = os.path.basename(comm).lower()
        exe = next((n for n in EXE_NAMES if base.startswith(n)), None)
        if exe:
            found.append((pid, exe))
            continue
        if base.startswith("python"):
            marker = next((m for m in PY_ARG_MARKERS if m in args.lower()), None)
            if marker:
                found.append((pid, f"python[{marker}]"))
    return found


def free_memory_mb(vm_stat_text: str | None = None) -> dict[str, int] | None:
    """macOS vm_stat: free and (free + inactive + speculative + purgeable) in MiB."""
    if vm_stat_text is None:
        try:
            vm_stat_text = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, check=False
            ).stdout
        except OSError:
            return None
    m = re.search(r"page size of (\d+) bytes", vm_stat_text)
    if not m:
        return None
    page = int(m.group(1))

    def pages(name: str) -> int:
        mm = re.search(rf"Pages {name}:\s+(\d+)\.", vm_stat_text or "")
        return int(mm.group(1)) if mm else 0

    free = pages("free")
    avail = free + pages("inactive") + pages("speculative") + pages("purgeable")
    return {"free_mb": free * page >> 20, "available_est_mb": avail * page >> 20}


@dataclass
class Preflight:
    open_ports: list[tuple[int, str]] = field(default_factory=list)
    processes: list[tuple[int, str]] = field(default_factory=list)
    memory: dict[str, int] | None = None

    @property
    def ok(self) -> bool:
        return not self.open_ports and not self.processes


def preflight(
    *,
    ports: Mapping[int, str] | None = None,
    ps_lines: Iterable[str] | None = None,
    vm_stat_text: str | None = None,
    min_free_gb: float | None = None,
    quiet: bool = False,
) -> Preflight:
    """Refuse (``Refused``) while a model server is up; print free memory."""
    pf = Preflight()
    for port, label in (DEFAULT_PORTS if ports is None else ports).items():
        if port_open(port):
            pf.open_ports.append((port, label))
    pf.processes = model_processes(ps_lines)
    pf.memory = free_memory_mb(vm_stat_text)
    if not quiet:
        mem = pf.memory or {}
        print(
            f"preflight: open_ports={[p for p, _ in pf.open_ports]} "
            f"model_processes={[f'{pid}:{lab}' for pid, lab in pf.processes]} "
            f"free_mb={mem.get('free_mb')} available_est_mb={mem.get('available_est_mb')}"
        )
    if not pf.ok:
        what = [f"port {p} ({lab})" for p, lab in pf.open_ports] + [
            f"pid {pid} ({lab})" for pid, lab in pf.processes
        ]
        raise Refused(
            "refusing: model servers are running (" + ", ".join(what) + "); "
            "stop them first (make down; ollama: launchctl/pkill), see training/README.md"
        )
    if (
        min_free_gb is not None
        and pf.memory is not None
        and pf.memory["available_est_mb"] < min_free_gb * 1024
    ):
        raise Refused(
            f"refusing: available_est_mb={pf.memory['available_est_mb']} "
            f"< {min_free_gb:g} GiB (close desktop apps)"
        )
    return pf


# --------------------------------------------------------------------------- download

TINKER_PATH_RE = re.compile(
    r"^tinker://[A-Za-z0-9_.:\-]+/(sampler_weights|weights)/[A-Za-z0-9_.\-]+$"
)


def tinker_path_from_run(run_id: str, epoch: int, runs_dir: Path = RUNS_DIR) -> str:
    meta = json.loads((runs_dir / f"{run_id}.json").read_text(encoding="utf-8"))
    for ep in meta.get("epochs", []):
        if int(ep.get("epoch", -1)) == epoch:
            path = ep.get("sampler_path")
            if isinstance(path, str):
                return path
    raise Refused(f"refusing: run {run_id} has no sampler_path for epoch {epoch}")


def adapter_dir_for(tinker_path: str, out_root: Path = ADAPTERS_DIR) -> Path:
    """Where ``tinker checkpoint download --output out_root`` extracts the archive."""
    return out_root / tinker_path.replace("tinker://", "").replace("/", "_")


def adapter_complete(d: Path) -> bool:
    return (
        (d / "adapter_config.json").is_file()
        and ((d / "adapter_model.safetensors").is_file() or (d / "adapter_model.bin").is_file())
        and (d / "checkpoint_complete").exists()
    )


def _tinker_cli() -> list[str]:
    exe = Path(sys.executable).with_name("tinker")
    return [str(exe)] if exe.is_file() else ["tinker"]


def _default_runner(cmd: Sequence[str], env: Mapping[str, str]) -> int:
    return subprocess.run(list(cmd), env={**os.environ, **env}, check=False).returncode


def download(
    tinker_path: str,
    *,
    out_root: Path = ADAPTERS_DIR,
    runner: Runner = _default_runner,
    force: bool = False,
) -> Path:
    if not TINKER_PATH_RE.match(tinker_path):
        raise Refused("refusing: expected tinker://<run>/(sampler_weights|weights)/<name>")
    target = adapter_dir_for(tinker_path, out_root)
    if adapter_complete(target) and not force:
        print(f"download: skipped (present) {target.name}")
        return target
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [*_tinker_cli(), "checkpoint", "download", tinker_path, "--output", str(out_root)]
    if force or target.exists():
        cmd.append("--force")
    rc = runner(cmd, {})
    if rc != 0 or not adapter_complete(target):
        raise Refused(f"refusing: download failed (rc={rc}) or archive is not a PEFT adapter")
    describe_adapter(target)
    return target


def describe_adapter(d: Path) -> dict[str, Any]:
    cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
    st = d / "adapter_model.safetensors"
    n_tensors = None
    if st.is_file():
        n_tensors = len(_safetensors_header(st))
    sha, n = file_digest(st if st.is_file() else d / "adapter_model.bin")
    info = {
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "use_rslora": cfg.get("use_rslora", False),
        "base_model": cfg.get("base_model_name_or_path"),
        "target_modules": len(cfg.get("target_modules") or []),
        "tensors": n_tensors,
        "bytes": n,
        "sha256": sha,
    }
    print("adapter: " + " ".join(f"{k}={v}" for k, v in info.items()))
    return info


def _safetensors_header(path: Path) -> dict[str, Any]:
    """Tensor name -> {dtype, shape} from the safetensors header (no tensor data read)."""
    with path.open("rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    return {k: v for k, v in hdr.items()}


# --------------------------------------------------------------------------- adapter keys

_LORA_KEY_RE = re.compile(
    r"^(?:base_model\.model\.)?(?P<module>.+?)\.(?P<ab>lora_A|lora_B)(?:\.default)?\.weight$"
)

# Tinker trains LoRA on the text-only causal LM's modules. Three of its names do not exist in
# the HF checkpoint: the Gated DeltaNet q/k/v projections are SPLIT in Tinker but FUSED in HF
# (``linear_attn.in_proj_qkv``), and the unembedding is called ``unembed_tokens`` (HF:
# ``lm_head``, tied to the embeddings in Qwen3.5-4B).
#
# Fused layout, from transformers ``Qwen3_5GatedDeltaNet`` (``__init__`` + ``forward``):
#   in_proj_qkv = Linear(hidden, 2 * key_dim + value_dim)
#   query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
#   key_dim = linear_key_head_dim * linear_num_key_heads
#   value_dim = linear_value_head_dim * linear_num_value_heads
# i.e. plain contiguous blocks [q | k | v] along the output rows, no per-head interleaving in
# the HF tensor (each block is head-major: reshape(-1, head_dim)). The V-head regrouping that
# llama.cpp needs is done later by convert_hf_to_gguf on the merged HF tensor, so the merge
# must write HF order. ``in_proj_z``/``in_proj_a``/``in_proj_b`` are separate HF tensors.
FUSED_PARTS = ("in_proj_q", "in_proj_k", "in_proj_v")
FUSED_NAME = "in_proj_qkv"
UNEMBED_NAMES = ("unembed_tokens", "lm_head")
LM_HEAD = "lm_head.weight"


class AdapterFormatError(ValueError):
    """Fixed-code adapter format problem (module names are structural, not content)."""


@dataclass(frozen=True)
class Target:
    """Where one adapter module's ΔW = scale · B @ A goes in the base checkpoint."""

    module: str  # adapter module name (``model.layers.N....``)
    a_key: str
    b_key: str
    tensor: str  # base tensor name it is added into (``....weight``)
    row0: int  # first output row of the slice (0 unless fused)
    rows: int  # rows of the slice (== out_features unless fused)
    source: str | None = None  # base tensor to copy first (``lm_head`` from tied embeddings)


@dataclass(frozen=True)
class MergePlan:
    targets: list[Target]
    scale: float
    untie_lm_head: bool

    @property
    def tensors(self) -> list[str]:
        return sorted({t.tensor for t in self.targets})


def fused_slices(text_cfg: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """Split part -> (row0, rows) inside the fused ``in_proj_qkv`` weight (HF order q|k|v)."""
    try:
        key_dim = int(text_cfg["linear_key_head_dim"]) * int(text_cfg["linear_num_key_heads"])
        value_dim = int(text_cfg["linear_value_head_dim"]) * int(text_cfg["linear_num_value_heads"])
    except (KeyError, TypeError, ValueError):
        return {}
    return {
        "in_proj_q": (0, key_dim),
        "in_proj_k": (key_dim, key_dim),
        "in_proj_v": (2 * key_dim, value_dim),
    }


def lora_scale(cfg: Mapping[str, Any]) -> float:
    r = int(cfg["r"])
    alpha = float(cfg.get("lora_alpha", r))
    return alpha / (r**0.5) if cfg.get("use_rslora", False) else alpha / r


def check_adapter_config(cfg: Mapping[str, Any]) -> None:
    if cfg.get("peft_type", "LORA") != "LORA":
        raise AdapterFormatError("unsupported_peft_type")
    if cfg.get("bias", "none") != "none":
        raise AdapterFormatError("unsupported_bias")
    for k in ("use_dora", "modules_to_save", "alpha_pattern", "rank_pattern", "fan_in_fan_out"):
        if cfg.get(k):
            raise AdapterFormatError(f"unsupported_{k}")


def _candidates(module: str) -> list[str]:
    """Base names a Tinker text-LM module may have (text LM, or the VLM's language_model)."""
    out = [module]
    if module.startswith("model.") and not module.startswith("model.language_model."):
        out.append("model.language_model." + module.removeprefix("model."))
    return out


def plan_merge(
    adapter_shapes: Mapping[str, Sequence[int]],
    base_shapes: Mapping[str, Sequence[int]],
    adapter_cfg: Mapping[str, Any],
    text_cfg: Mapping[str, Any],
) -> MergePlan:
    """Map every adapter LoRA pair onto a base tensor (row slice) and check every shape.

    ``adapter_shapes``: adapter tensor name -> shape. ``base_shapes``: base checkpoint tensor
    name -> shape. Refuses (``AdapterFormatError``, fixed codes) on unknown keys, incomplete
    pairs, unmapped modules, duplicate or overlapping slices, rank or shape mismatches.
    """
    check_adapter_config(adapter_cfg)
    r = int(adapter_cfg["r"])
    by_module: dict[str, dict[str, str]] = {}
    bad: list[str] = []
    for key in adapter_shapes:
        m = _LORA_KEY_RE.match(key)
        if not m:
            bad.append(key)
            continue
        by_module.setdefault(m.group("module"), {})[m.group("ab")] = key
    if bad:
        raise AdapterFormatError(f"unrecognised_adapter_keys n={len(bad)} e.g. {bad[:3]}")
    if not by_module:
        raise AdapterFormatError("empty_adapter")
    slices = fused_slices(text_cfg)
    embed = next(
        (
            n
            for n in ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")
            if n in base_shapes
        ),
        None,
    )
    targets: list[Target] = []
    unmapped: list[str] = []
    untie = False
    for mod, ab in sorted(by_module.items()):
        if set(ab) != {"lora_A", "lora_B"}:
            raise AdapterFormatError(f"incomplete_lora_pair module={mod}")
        parent, _, leaf = mod.rpartition(".")
        tensor: str | None = None
        row0, rows, source = 0, -1, None
        if leaf in UNEMBED_NAMES and (parent in ("", "model")):
            tensor = LM_HEAD
            if LM_HEAD not in base_shapes:
                if embed is None:
                    raise AdapterFormatError("lm_head_without_embeddings")
                source, untie = embed, True
        else:
            for cand in _candidates(mod):
                if cand + ".weight" in base_shapes:
                    tensor = cand + ".weight"
                    break
            if tensor is None and leaf in FUSED_PARTS and leaf in slices:
                for cand in _candidates(parent):
                    if f"{cand}.{FUSED_NAME}.weight" in base_shapes:
                        tensor = f"{cand}.{FUSED_NAME}.weight"
                        row0, rows = slices[leaf]
                        break
        if tensor is None:
            unmapped.append(mod)
            continue
        base_shape = list(base_shapes[source or tensor])
        if len(base_shape) != 2:
            raise AdapterFormatError(f"not_a_matrix tensor={tensor}")
        out_f, in_f = base_shape
        if rows < 0:
            rows = out_f
        elif row0 + rows > out_f or sum(s[1] for s in slices.values()) != out_f:
            raise AdapterFormatError(f"fused_layout_mismatch tensor={tensor} base={base_shape}")
        a, b = list(adapter_shapes[ab["lora_A"]]), list(adapter_shapes[ab["lora_B"]])
        if len(a) != 2 or len(b) != 2 or a != [r, in_f] or b != [rows, r]:
            raise AdapterFormatError(
                f"shape_mismatch module={mod} A={a} B={b} slice=({rows},{in_f}) r={r}"
            )
        targets.append(Target(mod, ab["lora_A"], ab["lora_B"], tensor, row0, rows, source))
    if unmapped:
        raise AdapterFormatError(f"unmapped_modules n={len(unmapped)} e.g. {unmapped[:3]}")
    by_tensor: dict[str, list[Target]] = {}
    for t in targets:
        by_tensor.setdefault(t.tensor, []).append(t)
    for name, ts in by_tensor.items():
        spans = sorted((t.row0, t.row0 + t.rows) for t in ts)
        if any(s1 < e0 for (_, e0), (s1, _) in zip(spans, spans[1:], strict=False)):
            raise AdapterFormatError(f"duplicate_target tensor={name}")
    return MergePlan(targets=targets, scale=lora_scale(adapter_cfg), untie_lm_head=untie)


def lora_delta(a: Any, b: Any, scale: float) -> Any:
    """ΔW = scale · B @ A in float32 (shape rows × in)."""
    return (b.float() @ a.float()) * scale


def apply_deltas(weight: Any, deltas: Iterable[tuple[int, Any]]) -> Any:
    """Add each (row0, ΔW) into rows [row0, row0 + ΔW.rows) in float32, cast back."""
    w = weight.float().clone()
    for row0, d in deltas:
        w[row0 : row0 + d.shape[0]] += d
    return w.to(weight.dtype)


# --------------------------------------------------------------------------- base snapshot


def base_snapshot(base_id: str = DEFAULT_BASE_ID) -> Path:
    """Local HF snapshot dir of the base (offline). Refuses if the weights are not cached."""
    from huggingface_hub import snapshot_download

    old = {k: os.environ.get(k) for k in OFFLINE_ENV}
    os.environ.update(OFFLINE_ENV)
    try:
        path = Path(snapshot_download(base_id, local_files_only=True))
    except Exception as exc:  # noqa: BLE001 - fixed message
        raise Refused(f"refusing: {base_id} is not in the local HF cache") from exc
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    require_weights(path, base_id)
    return path


def require_weights(path: Path, base_id: str = DEFAULT_BASE_ID) -> None:
    if not any(path.glob("*.safetensors")):
        raise Refused(
            f"refusing: the cached {base_id} snapshot has no *.safetensors weights "
            f"(only config/tokenizer); fetch them first: hf download {base_id}"
        )


# --------------------------------------------------------------------------- merge


def merge(
    adapter: Path,
    *,
    base_id: str = DEFAULT_BASE_ID,
    out: Path = MERGED_DIR,
    force: bool = False,
    impl: Callable[[Path, Path, Path, str], None] | None = None,
    snapshot: Callable[[str], Path] = base_snapshot,
    check: Callable[[], Any] = preflight,
) -> Path:
    if not adapter_complete(adapter):
        raise Refused("refusing: adapter dir is not a complete PEFT adapter (run download)")
    inputs = {"adapter": fingerprint(adapter), "base": base_id}
    if not force and up_to_date(out, inputs):
        print("merge: skipped (up to date)")
        report("merged", out)
        return out
    check()
    snap = snapshot(base_id)
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    (impl or _merge_impl)(adapter, snap, tmp, base_id)
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    write_stamp(out, "merge", inputs)
    report("merged", out)
    return out


SHARD_BYTES = 2 << 30  # output shard size


def _base_headers(snap: Path) -> dict[str, tuple[str, dict[str, Any]]]:
    """Base tensor name -> (shard file name, {dtype, shape}) from the safetensors headers."""
    files = sorted(p.name for p in snap.glob("*.safetensors"))
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    for fn in files:
        for k, v in _safetensors_header(snap / fn).items():
            out[k] = (fn, v)
    return out


def merge_checkpoint(
    adapter: Path, snap: Path, out: Path, *, shard_bytes: int = SHARD_BYTES
) -> MergePlan:
    """Stream the base checkpoint shard by shard, add every ΔW in float32, write bf16.

    The base's tensors and names are kept as they are (same as the ``base`` path, so C and D
    convert identically); only the planned tensors change, plus ``lm_head.weight`` when the
    adapter trains the unembedding of a tied base (then ``tie_word_embeddings`` -> false).
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    acfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    cfg = json.loads((snap / "config.json").read_text(encoding="utf-8"))
    text_cfg = cfg.get("text_config", cfg)
    st = adapter / "adapter_model.safetensors"
    lora = (
        load_file(str(st))
        if st.is_file()
        else torch.load(adapter / "adapter_model.bin", map_location="cpu", weights_only=True)
    )
    headers = _base_headers(snap)
    plan = plan_merge(
        {k: tuple(v.shape) for k, v in lora.items()},
        {k: v[1]["shape"] for k, v in headers.items()},
        acfg,
        text_cfg,
    )
    fused = sum(1 for t in plan.targets if t.tensor.endswith(f".{FUSED_NAME}.weight"))
    print(
        f"merge: lora_modules={len(plan.targets)} base_tensors={len(plan.tensors)} "
        f"fused_slices={fused} scale={plan.scale:g} untie_lm_head={plan.untie_lm_head}"
    )
    by_tensor: dict[str, list[Target]] = {}
    for t in plan.targets:
        by_tensor.setdefault(t.tensor, []).append(t)
    by_source = {t.source: t.tensor for t in plan.targets if t.source}

    out.mkdir(parents=True, exist_ok=True)
    parts: list[tuple[Path, list[str]]] = []
    buf: dict[str, Any] = {}
    nbytes = 0
    applied = 0

    def flush() -> None:
        nonlocal buf, nbytes
        if not buf:
            return
        p = out / f"part-{len(parts):05d}.tmp"
        save_file(buf, str(p), metadata={"format": "pt"})
        parts.append((p, sorted(buf)))
        buf, nbytes = {}, 0

    def emit(name: str, t: Any) -> None:
        nonlocal nbytes, applied
        if name in by_tensor:
            ts = by_tensor[name]
            t = apply_deltas(
                t, [(x.row0, lora_delta(lora[x.a_key], lora[x.b_key], plan.scale)) for x in ts]
            )
            applied += len(ts)
        buf[name] = t.contiguous()
        nbytes += t.numel() * t.element_size()
        if nbytes >= shard_bytes:
            flush()

    for fn in sorted({fn for fn, _ in headers.values()}):
        with safe_open(str(snap / fn), framework="pt") as f:
            for name in f.keys():  # noqa: SIM118 - safe_open is not a dict
                t = f.get_tensor(name)
                emit(name, t)
                if name in by_source:  # untied lm_head starts as a copy of the embeddings
                    emit(by_source[name], t.clone())
                del t
    flush()
    if applied != len(plan.targets):
        raise Refused(f"refusing: applied {applied} of {len(plan.targets)} LoRA modules")

    weight_map: dict[str, str] = {}
    total = 0
    for i, (p, names) in enumerate(parts):
        final = out / f"model-{i + 1:05d}-of-{len(parts):05d}.safetensors"
        p.rename(final)
        total += final.stat().st_size
        weight_map.update({n: final.name for n in names})
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1) + "\n",
        encoding="utf-8",
    )
    if plan.untie_lm_head:
        for c in (cfg, cfg.get("text_config")):
            if isinstance(c, dict) and "tie_word_embeddings" in c:
                c["tie_word_embeddings"] = False
        cfg["tie_word_embeddings"] = False
        print("merge: lm_head untied from the embeddings (adapter trains the unembedding)")
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    for p in snap.iterdir():
        if (
            p.is_file()
            and p.suffix != ".safetensors"
            and p.name
            not in (
                "config.json",
                "model.safetensors.index.json",
            )
            and not p.name.startswith(".")
        ):
            shutil.copyfile(p, out / p.name)
    return plan


def _merge_impl(adapter: Path, snap: Path, out: Path, base_id: str) -> None:  # pragma: no cover
    """The heavy part (streams shard by shard; all model servers stopped)."""
    os.environ.update(OFFLINE_ENV)
    merge_checkpoint(adapter, snap, out)


# --------------------------------------------------------------------------- convert / quantize


def convert(
    hf_dir: Path,
    out: Path,
    *,
    outtype: str = OUTTYPE,
    runner: Runner = _default_runner,
    force: bool = False,
    check: Callable[[], Any] = preflight,
    label: str = "convert",
) -> Path:
    if not hf_dir.is_dir() or not (hf_dir / "config.json").is_file():
        raise Refused(f"refusing: {label} needs an HF model dir ({hf_dir.name}); run merge first")
    require_weights(hf_dir)
    inputs = {"hf": fingerprint(hf_dir), "outtype": outtype}
    if not force and up_to_date(out, inputs):
        print(f"{label}: skipped (up to date)")
        report(label, out)
        return out
    if not CONVERT_PY.is_file() or not CONVERT_PYTHON.exists():
        raise Refused("refusing: vendor/llama.cpp or its .venv missing (run make models)")
    check()
    tmp = out.with_name(out.name + ".partial")
    tmp.unlink(missing_ok=True)
    cmd = [
        str(CONVERT_PYTHON),
        str(CONVERT_PY),
        str(hf_dir),
        "--outfile",
        str(tmp),
        "--outtype",
        outtype,
        "--no-mtp",
    ]
    rc = runner(cmd, OFFLINE_ENV)
    if rc != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise Refused(f"refusing: convert_hf_to_gguf failed (rc={rc})")
    tmp.replace(out)
    write_stamp(out, label, inputs)
    report(label, out)
    return out


def quantize(
    f16: Path,
    out: Path,
    *,
    qtype: str = QUANT_TYPE,
    runner: Runner = _default_runner,
    force: bool = False,
    check: Callable[[], Any] = preflight,
    label: str = "quantize",
) -> Path:
    if not f16.is_file():
        raise Refused(f"refusing: {label} needs {f16.name}; run convert first")
    inputs = {"gguf": fingerprint(f16), "type": qtype}
    if not force and up_to_date(out, inputs):
        print(f"{label}: skipped (up to date)")
        report(label, out)
        return out
    exe = shutil.which("llama-quantize")
    if exe is None:
        raise Refused("refusing: llama-quantize not on PATH (brew install llama.cpp)")
    check()
    tmp = out.with_name(out.name + ".partial")
    tmp.unlink(missing_ok=True)
    rc = runner([exe, str(f16), str(tmp), qtype], {})
    if rc != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise Refused(f"refusing: llama-quantize failed (rc={rc})")
    tmp.replace(out)
    write_stamp(out, label, inputs)
    report(label, out)
    return out


def base(
    *,
    base_id: str = DEFAULT_BASE_ID,
    f16: Path = BASE_F16,
    out: Path = BASE_Q4,
    runner: Runner = _default_runner,
    force: bool = False,
    snapshot: Callable[[str], Path] = base_snapshot,
    check: Callable[[], Any] = preflight,
) -> Path:
    snap = snapshot(base_id)
    convert(snap, f16, runner=runner, force=force, check=check, label="base-convert")
    return quantize(f16, out, runner=runner, force=force, check=check, label="base-quantize")


def run_all(
    adapter: Path,
    *,
    base_id: str = DEFAULT_BASE_ID,
    runner: Runner = _default_runner,
    force: bool = False,
    impl: Callable[[Path, Path, Path, str], None] | None = None,
    snapshot: Callable[[str], Path] = base_snapshot,
    check: Callable[[], Any] = preflight,
    merged: Path = MERGED_DIR,
    f16: Path = WRITER_F16,
    q4: Path = WRITER_Q4,
) -> Path:
    """merge -> convert -> quantize (each idempotent)."""
    merge(
        adapter, base_id=base_id, out=merged, force=force, impl=impl, snapshot=snapshot, check=check
    )
    convert(merged, f16, runner=runner, force=force, check=check)
    return quantize(f16, q4, runner=runner, force=force, check=check)


# --------------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sair-Writer export (PRD §12.4)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--min-free-gb", type=float, default=None)
    d = sub.add_parser("download")
    d.add_argument("tinker_path", nargs="?")
    d.add_argument("--from-run", default=None)
    d.add_argument("--epoch", type=int, default=None)
    d.add_argument("--force", action="store_true")
    for name in ("merge", "all"):
        m = sub.add_parser(name)
        m.add_argument("--adapter", type=Path, required=True)
        m.add_argument("--base-id", default=DEFAULT_BASE_ID)
        m.add_argument("--force", action="store_true")
        m.add_argument("--min-free-gb", type=float, default=None)
    for name in ("convert", "quantize", "base"):
        c = sub.add_parser(name)
        c.add_argument("--force", action="store_true")
        c.add_argument("--min-free-gb", type=float, default=None)
        if name == "base":
            c.add_argument("--base-id", default=DEFAULT_BASE_ID)
    a = ap.parse_args(argv)

    def check() -> Preflight:
        return preflight(min_free_gb=getattr(a, "min_free_gb", None))

    if a.cmd == "preflight":
        try:
            check()
        except Refused as exc:
            print(exc, file=sys.stderr)
            return 3
        print("preflight: ok")
        return 0
    try:
        if a.cmd == "download":
            if a.from_run:
                if a.epoch is None:
                    raise Refused("refusing: --from-run needs --epoch")
                path = tinker_path_from_run(a.from_run, a.epoch)
            elif a.tinker_path:
                path = a.tinker_path
            else:
                raise Refused("refusing: give a tinker:// path or --from-run/--epoch")
            print(f"download: adapter_dir={download(path, force=a.force).name}")
        elif a.cmd == "merge":
            merge(a.adapter, base_id=a.base_id, force=a.force, check=check)
        elif a.cmd == "all":
            run_all(a.adapter, base_id=a.base_id, force=a.force, check=check)
        elif a.cmd == "convert":
            convert(MERGED_DIR, WRITER_F16, force=a.force, check=check)
        elif a.cmd == "quantize":
            quantize(WRITER_F16, WRITER_Q4, force=a.force, check=check)
        elif a.cmd == "base":
            base(base_id=a.base_id, force=a.force, check=check)
    except Refused as exc:
        print(exc, file=sys.stderr)
        return 3
    except AdapterFormatError as exc:
        print(f"refusing: adapter format: {exc}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
