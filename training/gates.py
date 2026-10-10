"""Sair-Writer gates (PRD §12.5). They print a recommendation; they never edit .env files.

    uv run python training/gates.py c-dev                         # base few-shot dev reference
    uv run python training/gates.py select --run <run_id>         # §12.3 epoch choice
    uv run python training/gates.py g-export --run <run_id> [--epoch k]
    uv run python training/gates.py g-ship --records eval/results/gship-c/items.jsonl \\
        eval/results/gship-d/items.jsonl

G-export (rule as implemented)
------------------------------
Same dev items (``config.dev_items_path`` / ``dev_limit`` recorded in the run), same production
writer prompt, first attempt, grammar OFF, no regeneration, temperature 0, scored by the same
function (``training.train.score_dev``) on both sides:

* (a) the Tinker sampler checkpoint: the per-epoch dev metrics ``train.py`` stored in
  ``training/runs/<run>.json`` for the SELECTED epoch (``--epoch``; else ``selected_epoch`` in
  the run file; else the §12.3 rule against ``training/gates/c_dev.json``);
* (b) the Q4_K_M GGUF served by llama-server (``WRITER_URL``) through eval's
  ``LlamaServerWriter`` with :class:`training.renderer_runtime.RuntimeRenderer`.

For each of unsupported_rate, fact_recall, fallback_rate, valid_json_no_grammar the delta is
``gguf - tinker`` in proportion units; PASS iff EVERY ``|delta| <= 0.02`` (2 percentage
points, inclusive, compared after both sides are rounded to 4 dp as train.py stores them). A
metric that is None on exactly one side FAILS; None on both sides is "n/a" (not compared).
The run's lexicon hash and writer prompt fingerprint must equal the current ones, else the
gate refuses (the scores would not be comparable).

G-ship (rule as implemented)
----------------------------
Inputs: eval records (``eval/run.py ... --items training/data/dev.jsonl``) for ``D+`` and ``C+``
on the same dev items, metrics from ``eval.metrics`` (final entries, grammar on, 1 regen +
template). A metric that is None for both systems (e.g. recall with no gold facts) is n/a,
except unsupported_rate, which must exist. ``llama`` iff ALL of:

1. G-export passed (``training/gates/g_export.json``, ``pass: true``);
2. unsupported_rate(D+) < unsupported_rate(C+)  (strictly lower);
3. fact_recall(D+) >= fact_recall(C+);
4. template_fallback_rate(D+) <= template_fallback_rate(C+);

otherwise ``gemma``, with every failed condition listed as a reason code. Also reported: 95 %
cluster-bootstrap-by-seed CIs per system and for each paired difference D+ - C+ (10k
resamples, ``eval/bootstrap.py``); ``borderline: true`` when the CI of the unsupported-rate
difference includes 0 (or is degenerate). Borderline does not change the point-estimate
decision; it is a flag for the owner, who makes the final call (dev set, never reported).

Printed and written: rates, counts, deltas, ids, codes. Never item text.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "training" / "runs"
GATES_DIR = ROOT / "training" / "gates"
DEV_ITEMS = ROOT / "training" / "data" / "dev.jsonl"
RAW_CACHE = ROOT / "training" / "cache" / "gates"  # gitignored; synthetic dev outputs
G_EXPORT_JSON = GATES_DIR / "g_export.json"
G_SHIP_JSON = GATES_DIR / "g_ship.json"
C_DEV_JSON = GATES_DIR / "c_dev.json"

EXPORT_METRICS = ("unsupported_rate", "fact_recall", "fallback_rate", "valid_json_no_grammar")
EXPORT_TOL = 0.02  # 2 percentage points
_EPS = 1e-9  # float slack so an exact 0.0200 delta passes

SHIP_METRICS = ("unsupported_rate", "fact_recall", "template_fallback_rate")

G_EXPORT_RULE = (
    "PASS iff |gguf - tinker| <= 0.02 (2 pp, inclusive) for every one of "
    + ", ".join(EXPORT_METRICS)
    + "; same dev items, first attempt, grammar off, temperature 0, train.score_dev; "
    "None on one side fails, None on both is n/a"
)
G_SHIP_RULE = (
    "llama iff g_export passed AND unsupported_rate(D+) < unsupported_rate(C+) AND "
    "fact_recall(D+) >= fact_recall(C+) AND template_fallback_rate(D+) <= "
    "template_fallback_rate(C+) on dev; else gemma. borderline iff the 95% cluster-bootstrap "
    "CI of unsupported_rate(D+) - unsupported_rate(C+) includes 0"
)


class Refused(SystemExit):
    pass


def _write(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- G-export


def compare_export(
    tinker: Mapping[str, float | None], gguf: Mapping[str, float | None], tol: float = EXPORT_TOL
) -> dict[str, Any]:
    """Pure G-export decision over two rate dicts (proportions)."""
    rows: dict[str, Any] = {}
    ok = True
    for m in EXPORT_METRICS:
        t, g = tinker.get(m), gguf.get(m)
        if t is None and g is None:
            rows[m] = {"tinker": None, "gguf": None, "delta_pp": None, "pass": None}
            continue
        if t is None or g is None:
            rows[m] = {"tinker": t, "gguf": g, "delta_pp": None, "pass": False}
            ok = False
            continue
        delta = float(g) - float(t)
        passed = abs(delta) <= tol + _EPS
        ok &= passed
        rows[m] = {"tinker": t, "gguf": g, "delta_pp": round(delta * 100, 2), "pass": passed}
    return {"pass": ok, "metrics": rows, "tolerance_pp": tol * 100, "rule": G_EXPORT_RULE}


def select_epoch(
    epochs: Sequence[Mapping[str, Any]], c_ref: Mapping[str, float | None]
) -> int | None:
    """§12.3: min dev unsupported_rate s.t. recall >= C's and fallback <= C's (ties: earlier)."""
    best: tuple[float, int] | None = None
    for ep in epochs:
        dev = ep.get("dev") or {}
        u, r, f = dev.get("unsupported_rate"), dev.get("fact_recall"), dev.get("fallback_rate")
        cr, cf = c_ref.get("fact_recall"), c_ref.get("fallback_rate")
        if u is None or r is None or f is None or cr is None or cf is None:
            continue
        if r + _EPS < cr or f - _EPS > cf:
            continue
        key = (float(u), int(ep["epoch"]))
        if best is None or key < best:
            best = key
    return None if best is None else best[1]


def load_run(run_id: str, runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    p = runs_dir / f"{run_id}.json"
    if not p.is_file():
        raise Refused(f"refusing: no run file training/runs/{run_id}.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("epochs"):
        raise Refused("refusing: run file has no per-epoch dev metrics")
    return data


def resolve_epoch(run: Mapping[str, Any], epoch: int | None, c_ref_path: Path = C_DEV_JSON) -> int:
    if epoch is not None:
        return epoch
    if isinstance(run.get("selected_epoch"), int):
        return int(run["selected_epoch"])
    if c_ref_path.is_file():
        c_ref = json.loads(c_ref_path.read_text(encoding="utf-8"))["dev"]
        sel = select_epoch(run["epochs"], c_ref)
        if sel is None:
            raise Refused(
                "refusing: no epoch meets the §12.3 constraints vs base few-shot "
                "(G-ship would fail: WRITER_BACKEND=gemma)"
            )
        return sel
    raise Refused("refusing: pass --epoch, or run `gates.py c-dev` for the §12.3 selection")


def check_comparable(run: Mapping[str, Any]) -> None:
    from hub.grounding import lexicon_hash
    from hub.prompts.builders import prompt_fingerprint

    if run.get("lexicon_hash") != lexicon_hash():
        raise Refused("refusing: lexicon changed since training (lexicon_hash mismatch)")
    if run.get("prompt_fingerprint") != prompt_fingerprint("writer"):
        raise Refused("refusing: writer prompt changed since training (fingerprint mismatch)")


def dev_items_for(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    from training.tinker_common import read_jsonl

    cfg = run.get("config") or {}
    path = Path(cfg.get("dev_items_path") or DEV_ITEMS)
    items = read_jsonl(path)
    if cfg.get("dev_limit"):
        items = items[: int(cfg["dev_limit"])]
    return items


Generate = Callable[[list[Any]], str]


def run_dev(
    items: Sequence[dict[str, Any]], generate: Generate, raw_out: Path | None = None
) -> tuple[dict[str, float | None], int, dict[str, int]]:
    """First-attempt, no-regen dev pass scored exactly like train.py (``score_dev``)."""
    from training.tinker_common import writer_messages
    from training.train import score_dev

    raws = [generate(writer_messages(it)) for it in items]
    if raw_out is not None:  # synthetic dev outputs, local + gitignored, for debugging
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        with raw_out.open("w", encoding="utf-8") as fh:
            for it, raw in zip(items, raws, strict=True):
                fh.write(
                    json.dumps({"item_id": it.get("item_id"), "raw": raw}, ensure_ascii=False)
                    + "\n"
                )
    m = score_dev(items, raws)
    return m.rates(), m.n, m.reasons


def llama_generate(url: str, renderer: Any, model_id: str) -> Generate:  # pragma: no cover
    from eval.backends import LlamaServerWriter

    w = LlamaServerWriter(url, renderer, model_id=model_id)

    def gen(messages: list[Any]) -> str:
        return w.generate(messages, grammar=False)

    return gen


def _require_loopback(url: str) -> None:
    from eval.config import is_loopback_url

    if not is_loopback_url(url):
        raise Refused("refusing: model URL must be loopback")


def g_export(
    run_id: str,
    *,
    epoch: int | None = None,
    generate: Generate | None = None,
    url: str | None = None,
    runs_dir: Path = RUNS_DIR,
    out: Path = G_EXPORT_JSON,
    c_ref_path: Path = C_DEV_JSON,
    items: Sequence[dict[str, Any]] | None = None,
    comparable: Callable[[Mapping[str, Any]], None] = check_comparable,
    gguf_sha: str | None = None,
) -> dict[str, Any]:
    run = load_run(run_id, runs_dir)
    comparable(run)
    k = resolve_epoch(run, epoch, c_ref_path)
    ep = next((e for e in run["epochs"] if int(e.get("epoch", -1)) == k), None)
    if ep is None or not ep.get("dev"):
        raise Refused(f"refusing: epoch {k} has no dev metrics in the run file")
    dev_items = list(items) if items is not None else dev_items_for(run)
    if int(ep.get("dev_n", len(dev_items))) != len(dev_items):
        raise Refused("refusing: dev item count differs from the Tinker dev eval")
    if generate is None:  # pragma: no cover - live
        from hub.settings import get_settings
        from training.renderer_runtime import RuntimeRenderer

        url = url or get_settings().writer_url
        _require_loopback(url)
        generate = llama_generate(url, RuntimeRenderer(), "sair-writer-q4_k_m")
    rates, n, reasons = run_dev(
        dev_items, generate, RAW_CACHE / f"g_export-{run_id}-e{k}.jsonl" if items is None else None
    )
    res = compare_export(ep["dev"], rates)
    res.update(
        {
            "gate": "G-export",
            "run_id": run_id,
            "epoch": k,
            "sampler_path": ep.get("sampler_path"),
            "dev_n": n,
            "gguf": {"url": url, "sha256": gguf_sha, "reasons": reasons},
        }
    )
    _write(out, res)
    return res


def c_dev(
    *,
    generate: Generate | None = None,
    url: str | None = None,
    items_path: Path = DEV_ITEMS,
    out: Path = C_DEV_JSON,
) -> dict[str, Any]:
    """Base few-shot (C) dev reference with the same first-attempt/no-grammar scoring."""
    from training.tinker_common import read_jsonl

    items = read_jsonl(items_path)
    fs_sha = None
    if generate is None:  # pragma: no cover - live
        from training.renderer_runtime import FewShotRenderer

        url = url or os.environ.get("EVAL_QWEN_BASE_URL", "http://127.0.0.1:8092")
        _require_loopback(url)
        r = FewShotRenderer()
        fs_sha = r.few_shot_sha
        generate = llama_generate(url, r, f"qwen3.5-4b-base-q4_k_m-fs{fs_sha[:8]}")
    rates, n, reasons = run_dev(items, generate)
    res = {
        "system": "C (base few-shot, first attempt, grammar off)",
        "dev": rates,
        "dev_n": n,
        "reasons": reasons,
        "few_shot_sha256": fs_sha,
        "url": url,
    }
    _write(out, res)
    return res


# --------------------------------------------------------------------------- G-ship


def ship_decision(
    d: Mapping[str, float | None],
    c: Mapping[str, float | None],
    *,
    g_export_passed: bool | None,
    unsupported_diff_ci: tuple[float | None, float | None] | None = None,
) -> dict[str, Any]:
    """Pure G-ship decision table. ``d``/``c``: point estimates for SHIP_METRICS."""
    reasons: list[str] = []
    if g_export_passed is None:
        reasons.append("g_export_missing")
    elif not g_export_passed:
        reasons.append("g_export_failed")
    vals = {m: (d.get(m), c.get(m)) for m in SHIP_METRICS}
    for m, (dv, cv) in vals.items():
        if (dv is None) != (cv is None) or (m == "unsupported_rate" and dv is None):
            reasons.append(f"metric_unavailable:{m}")  # both None (e.g. no gold facts) is n/a
    du, cu = vals["unsupported_rate"]
    if du is not None and cu is not None and not du < cu:
        reasons.append("unsupported_not_lower")
    dr, cr = vals["fact_recall"]
    if dr is not None and cr is not None and dr + _EPS < cr:
        reasons.append("recall_below_c")
    df, cf = vals["template_fallback_rate"]
    if df is not None and cf is not None and df - _EPS > cf:
        reasons.append("fallback_above_c")
    lo, hi = unsupported_diff_ci or (None, None)
    borderline = lo is None or hi is None or (lo <= 0.0 <= hi)
    return {
        "recommendation": "llama" if not reasons else "gemma",
        "reasons": reasons,
        "borderline": borderline,
        "rule": G_SHIP_RULE,
    }


def _load_records(paths: Sequence[Path]) -> dict[str, list[Any]]:
    from eval.systems import ItemRecord

    out: dict[str, list[Any]] = {}
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = ItemRecord.from_dict(json.loads(line))
                    out.setdefault(r.system, []).append(r)
    return out


def ship_metrics(
    records: Mapping[str, Sequence[Any]],
    items: Sequence[Any],
    *,
    d_label: str = "D+",
    c_label: str = "C+",
    resamples: int | None = None,
    rng_seed: int | None = None,
) -> dict[str, Any]:
    from eval.bootstrap import (
        DEFAULT_RESAMPLES,
        DEFAULT_RNG_SEED,
        cluster_bootstrap,
        paired_bootstrap,
    )
    from eval.metrics import METRICS

    res = resamples or DEFAULT_RESAMPLES
    seed = rng_seed if rng_seed is not None else DEFAULT_RNG_SEED
    if d_label not in records or c_label not in records:
        raise Refused(f"refusing: records need both {d_label} and {c_label}")
    by_id = {it.id: it for it in items}
    d_by = {r.item_id: r for r in records[d_label]}
    c_by = {r.item_id: r for r in records[c_label]}
    if set(d_by) != set(c_by):
        raise Refused("refusing: D+ and C+ records cover different items")
    missing = set(d_by) - set(by_id)
    if missing:
        raise Refused(f"refusing: {len(missing)} record ids are not in the dev items")
    ids = sorted(d_by)
    clusters = [by_id[i].seed for i in ids]
    out: dict[str, Any] = {
        "n_items": len(ids),
        "n_clusters": len(set(clusters)),
        "systems": {d_label: {}, c_label: {}},
        "paired": {},
    }
    for m in SHIP_METRICS:
        fn = METRICS[m]
        dv = [fn(d_by[i], by_id[i]) for i in ids]
        cv = [fn(c_by[i], by_id[i]) for i in ids]
        out["systems"][d_label][m] = cluster_bootstrap(
            dv, clusters, n_resamples=res, rng_seed=seed
        ).to_dict()
        out["systems"][c_label][m] = cluster_bootstrap(
            cv, clusters, n_resamples=res, rng_seed=seed
        ).to_dict()
        out["paired"][m] = paired_bootstrap(
            dv, cv, clusters, n_resamples=res, rng_seed=seed
        ).to_dict()
    return out


def g_ship(
    record_paths: Sequence[Path],
    *,
    items_path: Path = DEV_ITEMS,
    g_export_path: Path = G_EXPORT_JSON,
    out: Path = G_SHIP_JSON,
    d_label: str = "D+",
    c_label: str = "C+",
    resamples: int | None = None,
) -> dict[str, Any]:
    from eval.items import load_items

    items = load_items(items_path)
    recs = _load_records(record_paths)
    met = ship_metrics(recs, items, d_label=d_label, c_label=c_label, resamples=resamples)
    gx: bool | None = None
    if g_export_path.is_file():
        gx = bool(json.loads(g_export_path.read_text(encoding="utf-8")).get("pass"))
    d = {m: met["systems"][d_label][m]["point"] for m in SHIP_METRICS}
    c = {m: met["systems"][c_label][m]["point"] for m in SHIP_METRICS}
    pu = met["paired"]["unsupported_rate"]
    ci = None if pu.get("degenerate") else (pu.get("lo"), pu.get("hi"))
    dec = ship_decision(d, c, g_export_passed=gx, unsupported_diff_ci=ci)
    dec.update(
        {
            "gate": "G-ship",
            "set": "dev",
            "d_label": d_label,
            "c_label": c_label,
            "g_export_passed": gx,
            "metrics": met,
            "note": "recommendation only; the owner sets WRITER_BACKEND",
        }
    )
    _write(out, dec)
    return dec


# --------------------------------------------------------------------------- cli


def _fmt(v: float | None) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sair-Writer gates (PRD §12.5)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ge = sub.add_parser("g-export")
    ge.add_argument("--run", required=True)
    ge.add_argument("--epoch", type=int, default=None)
    ge.add_argument("--url", default=None, help="llama-server serving the Sair-Writer Q4_K_M")
    se = sub.add_parser("select", help="§12.3 epoch choice vs training/gates/c_dev.json")
    se.add_argument("--run", required=True)
    cd = sub.add_parser("c-dev")
    cd.add_argument("--url", default=None, help="llama-server serving the base Q4_K_M")
    gs = sub.add_parser("g-ship")
    gs.add_argument("--records", type=Path, nargs="+", required=True)
    gs.add_argument("--items", type=Path, default=DEV_ITEMS)
    gs.add_argument("--d-label", default="D+")
    gs.add_argument("--c-label", default="C+")
    gs.add_argument("--resamples", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "g-export":
            gguf_sha = None
            from training.export_adapter import WRITER_Q4, read_stamp

            st = read_stamp(WRITER_Q4)
            if st:
                gguf_sha = st["output"]["sha256"]
            r = g_export(a.run, epoch=a.epoch, url=a.url, gguf_sha=gguf_sha)
            print(f"G-export run={r['run_id']} epoch={r['epoch']} dev_n={r['dev_n']}")
            for m, row in r["metrics"].items():
                d = row["delta_pp"]
                print(
                    f"  {m:<22} tinker={_fmt(row['tinker'])} gguf={_fmt(row['gguf'])} "
                    f"delta_pp={'n/a' if d is None else f'{d:+.2f}'} pass={row['pass']}"
                )
            print(
                f"G-export: {'PASS' if r['pass'] else 'FAIL'} -> {G_EXPORT_JSON.relative_to(ROOT)}"
            )
            return 0 if r["pass"] else 1
        if a.cmd == "select":
            k = resolve_epoch(load_run(a.run), None)
            print(f"select: run={a.run} epoch={k}")
            return 0
        if a.cmd == "c-dev":
            r = c_dev(url=a.url)
            print(
                "C dev: n="
                + str(r["dev_n"])
                + " "
                + " ".join(f"{k}={_fmt(v)}" for k, v in r["dev"].items())
            )
            return 0
        r = g_ship(
            a.records,
            items_path=a.items,
            d_label=a.d_label,
            c_label=a.c_label,
            resamples=a.resamples,
        )
        met = r["metrics"]
        print(
            f"G-ship dev n_items={met['n_items']} n_clusters={met['n_clusters']} "
            f"g_export_passed={r['g_export_passed']}"
        )
        for m in SHIP_METRICS:
            dd, cc, pp = (
                met["systems"][a.d_label][m],
                met["systems"][a.c_label][m],
                met["paired"][m],
            )
            print(
                f"  {m:<22} D+={_fmt(dd['point'])} [{_fmt(dd['lo'])},{_fmt(dd['hi'])}] "
                f"C+={_fmt(cc['point'])} [{_fmt(cc['lo'])},{_fmt(cc['hi'])}] "
                f"diff={_fmt(pp['point'])} [{_fmt(pp['lo'])},{_fmt(pp['hi'])}]"
            )
        print(
            f"G-ship: recommend WRITER_BACKEND={r['recommendation']} reasons={r['reasons']} "
            f"borderline={r['borderline']} -> {G_SHIP_JSON.relative_to(ROOT)}"
        )
        return 0
    except Refused as exc:
        print(exc, file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
