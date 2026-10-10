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
* ``merge``: transformers + peft on the CPU, bf16, ``low_cpu_mem_usage=True`` (a no-op in
  transformers 5, which always loads lazily), ``HF_HUB_OFFLINE=1``. The adapter's keys are
  first CONVERTED to the base model's module names (Tinker names modules as in the text-only
  causal LM, ``model.layers.N...``; the HF snapshot is ``Qwen3_5ForConditionalGeneration``,
  ``model.language_model.layers.N...``). Every adapter tensor must map to exactly one base
  module with matching shapes, and every tensor must be present in the PEFT model after
  loading, or the merge is refused. If the adapter trains ``lm_head`` while the base ties it
  to the embeddings, ``lm_head`` is untied first (otherwise the merge would also change the
  input embeddings). Output: ``models/sair-writer-hf/`` + the tokenizer + chat template.
* ``convert``: ``vendor/llama.cpp/.venv/bin/python vendor/llama.cpp/convert_hf_to_gguf.py
  --outtype f16 --no-mtp`` -> ``models/sair-writer-f16.gguf``. ``--no-mtp`` on both C and D:
  transformers drops the ``mtp.*`` head on load, so the merged dir has none, and the base
  conversion must match it.
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


class AdapterFormatError(ValueError):
    """Fixed-code adapter format problem (module names are structural, not content)."""


@dataclass(frozen=True)
class KeyMap:
    mapping: dict[str, str]  # adapter key -> PEFT key on the base model
    modules: list[str]  # resolved base module names (sorted)


def _resolve_module(name: str, base_modules: Sequence[str]) -> str | None:
    """Exact match, else a unique suffix match, dropping leading components of ``name``."""
    if name in base_modules:
        return name
    parts = name.split(".")
    for k in range(len(parts)):
        tail = ".".join(parts[k:])
        hits = [m for m in base_modules if m == tail or m.endswith("." + tail)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            text = [h for h in hits if ".visual." not in f".{h}." and "vision" not in h]
            if len(text) == 1:
                return text[0]
            return None
    return None


def convert_adapter_keys(
    keys: Iterable[str],
    base_shapes: Mapping[str, tuple[int, int]],
    tensor_shapes: Mapping[str, Sequence[int]] | None = None,
) -> KeyMap:
    """Map adapter tensor names onto the base model's Linear modules.

    ``base_shapes``: module name -> (out_features, in_features) of every Linear in the base.
    Checks: every key is a ``lora_A``/``lora_B`` weight; each module has both; each module
    resolves to exactly one base Linear; and (when ``tensor_shapes`` is given)
    ``A`` is (r, in) and ``B`` is (out, r) for that module.
    """
    base_modules = sorted(base_shapes)
    by_module: dict[str, dict[str, str]] = {}
    bad: list[str] = []
    for key in keys:
        m = _LORA_KEY_RE.match(key)
        if not m:
            bad.append(key)
            continue
        by_module.setdefault(m.group("module"), {})[m.group("ab")] = key
    if bad:
        raise AdapterFormatError(f"unrecognised_adapter_keys n={len(bad)} e.g. {bad[:3]}")
    if not by_module:
        raise AdapterFormatError("empty_adapter")
    mapping: dict[str, str] = {}
    resolved: dict[str, str] = {}
    unmapped: list[str] = []
    for mod, ab in sorted(by_module.items()):
        if set(ab) != {"lora_A", "lora_B"}:
            raise AdapterFormatError(f"incomplete_lora_pair module={mod}")
        target = _resolve_module(mod, base_modules)
        if target is None:
            unmapped.append(mod)
            continue
        if target in resolved.values():
            raise AdapterFormatError(f"duplicate_target module={target}")
        resolved[mod] = target
        if tensor_shapes is not None:
            out_f, in_f = base_shapes[target]
            a = list(tensor_shapes[ab["lora_A"]])
            b = list(tensor_shapes[ab["lora_B"]])
            if len(a) != 2 or len(b) != 2 or a[1] != in_f or b[0] != out_f or a[0] != b[1]:
                raise AdapterFormatError(
                    f"shape_mismatch module={target} A={a} B={b} base=({out_f},{in_f})"
                )
        for which, key in ab.items():
            mapping[key] = f"base_model.model.{target}.{which}.weight"
    if unmapped:
        raise AdapterFormatError(f"unmapped_modules n={len(unmapped)} e.g. {unmapped[:3]}")
    return KeyMap(mapping=mapping, modules=sorted(resolved.values()))


def converted_config(
    cfg: Mapping[str, Any], modules: Sequence[str], base_id: str
) -> dict[str, Any]:
    """PEFT LoRA config targeting exactly the resolved modules (merge only: no dropout)."""
    out = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_id,
        "r": int(cfg["r"]),
        "lora_alpha": cfg.get("lora_alpha", cfg["r"]),
        "use_rslora": bool(cfg.get("use_rslora", False)),
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "target_modules": list(modules),
        "task_type": None,
        "inference_mode": True,
    }
    if cfg.get("bias", "none") != "none":
        raise AdapterFormatError("unsupported_bias")
    for k in ("use_dora", "modules_to_save"):
        if cfg.get(k):
            raise AdapterFormatError(f"unsupported_{k}")
    return out


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


def _merge_impl(adapter: Path, snap: Path, out: Path, base_id: str) -> None:  # pragma: no cover
    """The heavy part (needs ~9-10 GB RAM; all model servers stopped)."""
    os.environ.update(OFFLINE_ENV)
    import tempfile

    import torch
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    cfg = AutoConfig.from_pretrained(snap, local_files_only=True)
    arch = (getattr(cfg, "architectures", None) or [""])[0]
    cls = AutoModelForImageTextToText if "ConditionalGeneration" in arch else AutoModelForCausalLM
    model = cls.from_pretrained(
        snap,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
        local_files_only=True,
    )
    model.eval()
    linears = {
        n: (m.out_features, m.in_features)
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
    }
    acfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    st = adapter / "adapter_model.safetensors"
    tensors = (
        load_file(str(st))
        if st.is_file()
        else torch.load(adapter / "adapter_model.bin", map_location="cpu", weights_only=True)
    )
    km = convert_adapter_keys(
        tensors.keys(), linears, {k: tuple(v.shape) for k, v in tensors.items()}
    )
    print(f"merge: arch={arch} lora_modules={len(km.modules)} tensors={len(km.mapping)}")

    # untie lm_head if the adapter trains it while the base ties it to the embeddings
    if any(m.split(".")[-1] == "lm_head" for m in km.modules):
        emb = model.get_input_embeddings().weight
        head = model.get_output_embeddings()
        if head is not None and head.weight.data_ptr() == emb.data_ptr():
            head.weight = torch.nn.Parameter(emb.detach().clone())
            for c in (model.config, getattr(model.config, "text_config", None)):
                if c is not None and hasattr(c, "tie_word_embeddings"):
                    c.tie_word_embeddings = False
            print("merge: lm_head untied from the embeddings (adapter trains lm_head)")

    with tempfile.TemporaryDirectory() as td:
        conv = Path(td)
        (conv / "adapter_config.json").write_text(
            json.dumps(converted_config(acfg, km.modules, base_id), indent=1), encoding="utf-8"
        )
        save_file(
            {km.mapping[k]: v.contiguous() for k, v in tensors.items()},
            str(conv / "adapter_model.safetensors"),
        )
        peft_model = PeftModel.from_pretrained(model, str(conv), is_trainable=False)
        sd = peft_model.state_dict()
        missing = 0
        for k, v in tensors.items():
            pk = km.mapping[k].removesuffix(".weight") + ".default.weight"
            if pk not in sd or not torch.equal(sd[pk].to(v.dtype), v):
                missing += 1
        if missing:
            raise Refused(f"refusing: {missing} adapter tensors did not load into the model")
        merged = peft_model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    tok.save_pretrained(out)
    for extra in (
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        if (snap / extra).is_file() and not (out / extra).is_file():
            shutil.copyfile(snap / extra, out / extra)


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
