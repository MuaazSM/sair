"""Eval CLI (PRD §13): systems A-D and B+/C+/D+ on the synthetic or the real set.

Usage::

    uv run python eval/run.py --cached                          # replay eval/cache (CI, make eval)
    uv run python eval/run.py --systems B+,C+,D+ --out eval/results/dev1/
    uv run python eval/run.py --cached --assert-baseline eval/baseline.json
    uv run python eval/run.py --no-grammar-pass                 # adds B/C/D without grammar
    uv run python eval/run.py --set real                        # OWNER ONLY, aggregates only

Exit codes: 0 ok · 1 run error (type name only) · 2 nothing to evaluate (no items or no
cached results yet) · 3 real-set guard refused · 4 cache miss in ``--cached`` mode ·
5 baseline assertion failed.

Real-set guard (NFR-P6, PRD §13.2): ``--set real`` refuses unless every backend that will
be used has a loopback ``base_url`` (127.0.0.1, ::1, localhost; in-process backends have
none), refuses when a Sentry DSN is configured or a Sentry client is active (real data must
never reach telemetry, NFR-P7), and refuses an ``--out`` outside ``eval/results/real/``.
The checks run before the real items are opened. The real set prints aggregates only and
writes ``metrics.json`` + CSVs under ``eval/results/real/`` (gitignored); no per-item file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.backends import CachedWriter, CacheMiss, CacheMode, WriterBackend
from eval.bootstrap import (
    CI,
    DEFAULT_RESAMPLES,
    DEFAULT_RNG_SEED,
    cluster_bootstrap,
    paired_bootstrap,
    plain_bootstrap,
)
from eval.config import ConfigError, is_loopback_url
from eval.items import EvalItem, ItemsError, load_items
from eval.metrics import (
    METRICS,
    Ratio,
    cer_ratio,
    checker_recall_details,
    checker_recall_items,
    human_item_label,
    latency_percentiles,
    wer_ratio,
)
from eval.systems import ALL_SYSTEMS, BASE_OF, ItemRecord, run_item

__all__ = ["MSG_REAL_NOT_LOOPBACK", "MSG_REAL_OUT", "MSG_REAL_SENTRY", "main"]

ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_ITEMS = ROOT / "eval" / "data" / "synthetic" / "test.jsonl"
SYNTHETIC_CACHE = ROOT / "eval" / "cache" / "synthetic"
RESULTS = ROOT / "eval" / "results"
REAL_ROOT = ROOT / "eval" / "real"  # owner-only; never opened except by --set real
REAL_RESULTS = ROOT / "eval" / "results" / "real"

MSG_REAL_NOT_LOOPBACK = (
    "refusing --set real: every backend URL must be loopback (127.0.0.1, ::1, localhost)"
)
MSG_REAL_SENTRY = "refusing --set real: a Sentry DSN is configured or Sentry is active"
MSG_REAL_OUT = "refusing --set real: output must stay under eval/results/real/"
MSG_NO_CACHE = (
    "no cached results yet: run `make eval-live` (synthetic) to fill eval/cache/synthetic/"
)
MSG_NO_ITEMS = "no synthetic test set yet: expected eval/data/synthetic/test.jsonl"

DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (("D+", "C+"), ("D+", "B+"), ("C+", "B+"), ("D", "C"))
NO_GRAMMAR_SYSTEMS = ("B", "C", "D")

DEFINITIONS: dict[str, str] = {
    "unsupported_rate": "entries with >=1 checker unsupported detail / items with a final entry",
    "details_per_entry": "distinct checker unsupported details per final entry",
    "hedge_preservation": (
        "mentioned gold-unsure facts kept unsure AND hedged in every mentioning sentence"
    ),
    "fact_precision": (
        "matched verified facts / verified facts (span token Jaccard>=0.5 + alias subject)"
    ),
    "fact_recall": "matched verified facts / gold facts",
    "template_fallback_rate": "items whose final entry is the code-built template",
    "mission_accuracy": "final (check-7 coerced) mission_result == gold",
    "invalid_rate": "items with no final entry (unparseable plain output or backend error)",
    "valid_json_no_grammar": "first-attempt outputs that parse to the 9.1 schema, grammar off",
    "wer": "word edit distance / reference words (NFC, casefold, no punctuation)",
    "cer": "code-point edit distance / reference code points (same normalisation)",
    "checker_recall_items": "items human-flagged unsupported that the checker also flagged",
    "checker_recall_details": "human-listed unsupported details the checker also found",
}


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _sentry_configured() -> bool:
    if os.environ.get("SENTRY_DSN", "").strip():
        return True
    try:
        from hub.settings import get_settings

        if get_settings().sentry_dsn.get_secret_value().strip():
            return True
    except Exception:  # noqa: BLE001 - unreadable settings: be safe and refuse
        return True
    try:
        import sentry_sdk

        return bool(sentry_sdk.get_client().is_active())
    except Exception:  # noqa: BLE001
        return False


def _has_cache(model_writers: Sequence[WriterBackend]) -> bool:
    """--cached needs every model writer to be a cache replayer over a non-empty cache dir."""
    for w in model_writers:
        if not isinstance(w, CachedWriter) or w.store.mode is not CacheMode.CACHED:
            return False
        d = w.store.cache_dir
        if not d.is_dir() or not any(d.glob("*.json")):
            return False
    return True


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _ci(
    values: Sequence[Ratio], clusters: Sequence[str], *, real: bool, resamples: int, seed: int
) -> CI:
    if real:
        return plain_bootstrap(values, n_resamples=resamples, rng_seed=seed)
    return cluster_bootstrap(values, clusters, n_resamples=resamples, rng_seed=seed)


def _evaluate(
    records: dict[str, list[ItemRecord]],
    items: Sequence[EvalItem],
    *,
    real: bool,
    resamples: int,
    seed: int,
    pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    by_id = {it.id: it for it in items}
    out: dict[str, Any] = {
        "set": "real" if real else "synthetic",
        "bootstrap": "plain" if real else "cluster-by-seed",
        "resamples": resamples,
        "rng_seed": seed,
        "n_items": len(items),
        "n_clusters": len({it.seed for it in items}) if not real else len(items),
        "definitions": DEFINITIONS,
        "systems": {},
        "paired": {},
        "latency": {},
    }
    per_metric: dict[str, dict[str, list[Ratio]]] = {}
    for system, recs in records.items():
        clusters = [r.seed for r in recs]
        sysout: dict[str, Any] = {
            "n_items": len(recs),
            "n_clusters": len(set(clusters)),
            "metrics": {},
        }
        for name, fn in METRICS.items():
            vals = [fn(r, by_id[r.item_id]) for r in recs]
            per_metric.setdefault(system, {})[name] = vals
            if all(v.den == 0 for v in vals):
                continue
            sysout["metrics"][name] = _ci(
                vals, clusters, real=real, resamples=resamples, seed=seed
            ).to_dict()
        human_items: list[tuple[bool, bool]] = []
        human_details: list[Ratio] = []
        for r in recs:
            labels = (by_id[r.item_id].human or {}).get(system)
            if labels is None or r.final_check is None:
                continue
            flag = human_item_label(labels)
            if flag is not None:
                human_items.append((flag, bool(r.final_check.unsupported_details)))
            if labels.unsupported is not None:
                human_details.append(
                    checker_recall_details(labels.unsupported, r.final_check.unsupported_details)
                )
        if human_items:
            v = checker_recall_items(human_items)
            sysout["metrics"]["checker_recall_items"] = {
                "point": None if v.den == 0 else v.num / v.den,
                "num": v.num,
                "den": v.den,
                "n_labelled_items": len(human_items),
            }
        if human_details:
            sysout["metrics"]["checker_recall_details"] = plain_bootstrap(
                human_details, n_resamples=resamples, rng_seed=seed
            ).to_dict()
        out["systems"][system] = sysout
        out["latency"][system] = latency_percentiles(recs)

    for a, b in pairs:
        if a not in records or b not in records:
            continue
        ida = [r.item_id for r in records[a]]
        idb = {r.item_id: k for k, r in enumerate(records[b])}
        common = [k for k, i in enumerate(ida) if i in idb]
        if not common:
            continue
        clusters = [records[a][k].seed for k in common]
        pair_out: dict[str, Any] = {}
        for name in METRICS:
            va = [per_metric[a][name][k] for k in common]
            vb = [per_metric[b][name][idb[ida[k]]] for k in common]
            if all(v.den == 0 for v in va) or all(v.den == 0 for v in vb):
                continue
            pair_out[name] = paired_bootstrap(
                va, vb, clusters, n_resamples=resamples, rng_seed=seed, plain=real
            ).to_dict()
        out["paired"][f"{a} - {b}"] = pair_out

    refs = [it for it in items if it.reference_transcript]
    if refs:
        wv = [wer_ratio(it.reference_transcript or "", it.transcript) for it in refs]
        cv = [cer_ratio(it.reference_transcript or "", it.transcript) for it in refs]
        cl = [it.seed for it in refs]
        out["transcription"] = {
            "wer": _ci(wv, cl, real=real, resamples=resamples, seed=seed).to_dict(),
            "cer": _ci(cv, cl, real=real, resamples=resamples, seed=seed).to_dict(),
        }
    return out


def _write_outputs(
    out_dir: Path, metrics: dict[str, Any], records: dict[str, list[ItemRecord]], *, real: bool
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["system", "metric", "point", "lo", "hi", "n_items", "n_clusters", "den", "degenerate"]
        )
        for system, s in metrics["systems"].items():
            for name, m in s["metrics"].items():
                w.writerow(
                    [
                        system,
                        name,
                        m.get("point"),
                        m.get("lo"),
                        m.get("hi"),
                        m.get("n_items", m.get("n_labelled_items")),
                        m.get("n_clusters"),
                        m.get("den"),
                        m.get("degenerate"),
                    ]
                )
    with (out_dir / "paired.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "metric", "diff", "lo", "hi", "n_items", "n_clusters", "degenerate"])
        for pair, ms in metrics["paired"].items():
            for name, m in ms.items():
                w.writerow(
                    [
                        pair,
                        name,
                        m["point"],
                        m["lo"],
                        m["hi"],
                        m["n_items"],
                        m["n_clusters"],
                        m["degenerate"],
                    ]
                )
    with (out_dir / "latency.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "node", "n", "p50_ms", "p95_ms"])
        for system, nodes in metrics["latency"].items():
            for node, v in nodes.items():
                w.writerow([system, node, v["n"], v["p50_ms"], v["p95_ms"]])
    if not real:
        with (out_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
            for recs in records.values():
                for r in recs:
                    fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def _print_summary(metrics: dict[str, Any]) -> None:
    """Aggregates only: names, numbers and counts. Never item ids or text."""
    print(
        f"set={metrics['set']} n_items={metrics['n_items']} n_clusters={metrics['n_clusters']} "
        f"bootstrap={metrics['bootstrap']} resamples={metrics['resamples']}"
    )
    for system, s in metrics["systems"].items():
        for name, m in s["metrics"].items():
            p = m.get("point")
            lo, hi = m.get("lo"), m.get("hi")
            ci = f" [{lo:.3f}, {hi:.3f}]" if lo is not None and hi is not None else ""
            val = "n/a" if p is None else f"{p:.3f}"
            print(
                f"{system:<14} {name:<24} {val}{ci} n={m.get('n_items', m.get('n_labelled_items'))}"
            )
    for pair, ms in metrics["paired"].items():
        for name, m in ms.items():
            if m["point"] is None:
                continue
            print(f"{pair:<14} {name:<24} diff={m['point']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}]")
    for name, m in metrics.get("transcription", {}).items():
        ci = f"[{m['lo']:.3f}, {m['hi']:.3f}]"
        print(f"transcription  {name:<24} {m['point']:.3f} {ci} n={m['n_items']}")


def _assert_baseline(metrics: dict[str, Any], baseline_path: Path, tol: float) -> list[str]:
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for system, s in base.get("systems", {}).items():
        cur = metrics["systems"].get(system)
        for name, m in s.get("metrics", {}).items():
            bp = m.get("point")
            cp = None if cur is None else cur["metrics"].get(name, {}).get("point")
            if bp is None and cp is None:
                continue
            if bp is None or cp is None or abs(float(cp) - float(bp)) > tol:
                failures.append(f"{system} {name}: baseline={bp} current={cp}")
    return failures


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="eval/run.py", description="Sair eval harness (PRD §13)")
    ap.add_argument("--set", dest="set_", choices=("synthetic", "real"), default="synthetic")
    ap.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--cached", action="store_true", help="replay eval/cache only; fail on a miss"
    )
    mode.add_argument(
        "--live", action="store_true", help="call every model again (overwrite cache)"
    )
    ap.add_argument(
        "--no-grammar-pass", action="store_true", help="also run B/C/D with grammar off"
    )
    ap.add_argument("--items", type=Path, default=None, help="synthetic item file")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--assert-baseline", type=Path, default=None)
    ap.add_argument("--tolerance", type=float, default=1e-9)
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    ap.add_argument("--renderer", default=None, help="module:attr prompt renderer for C/D")
    return ap.parse_args(argv)


def main(
    argv: Sequence[str] | None = None, *, writers: dict[str, WriterBackend] | None = None
) -> int:
    args = _parse(argv)
    real = args.set_ == "real"
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if not systems or any(s not in BASE_OF for s in systems):
        _err("unknown system (choose from " + ",".join(ALL_SYSTEMS) + ")")
        return 2
    runs: list[tuple[str, str, bool]] = [(s, s, True) for s in systems]  # (label, system, grammar)
    if args.no_grammar_pass:
        runs += [(f"{s}-nogrammar", s, False) for s in NO_GRAMMAR_SYSTEMS if s in systems]
    keys = sorted({BASE_OF[s] for _, s, _ in runs})
    mode = CacheMode.CACHED if args.cached else CacheMode.LIVE if args.live else CacheMode.RESUME
    cache_dir = (REAL_RESULTS / "cache") if real else SYNTHETIC_CACHE
    out_dir: Path = args.out or ((REAL_RESULTS / "latest") if real else (RESULTS / "latest"))

    if writers is None:
        from eval.config import build_writers

        try:
            writers = build_writers(
                keys, cache_dir=cache_dir, mode=mode, renderer_spec=args.renderer
            )
        except ConfigError as exc:
            _err(str(exc))
            return 2
        except Exception as exc:  # noqa: BLE001 - settings/config problems, type name only
            _err(f"could not configure backends: {type(exc).__name__}")
            return 2
    missing = [k for k in keys if k not in writers]
    if missing:
        _err("no backend configured for: " + ",".join(missing))
        return 2

    if real:  # guard first: nothing in the real root is opened before these checks pass
        if not all(is_loopback_url(writers[k].base_url) for k in keys):
            _err(MSG_REAL_NOT_LOOPBACK)
            return 3
        if _sentry_configured():
            _err(MSG_REAL_SENTRY)
            return 3
        if not _within(out_dir, REAL_RESULTS):
            _err(MSG_REAL_OUT)
            return 3
        items_path = REAL_ROOT / "items.jsonl"
    else:
        items_path = args.items or SYNTHETIC_ITEMS

    if not items_path.is_file():
        if real:
            _err("real set not found (owner: eval/real/items.jsonl)")
            return 2
        _err(MSG_NO_CACHE if args.cached else MSG_NO_ITEMS)
        return 2
    if args.cached and not _has_cache([writers[k] for k in keys if k != "template"]):
        _err(MSG_NO_CACHE)
        return 2

    try:
        items = load_items(items_path)
    except ItemsError as exc:
        _err(str(exc))
        return 2
    if not items:
        _err("item file is empty")
        return 2

    records: dict[str, list[ItemRecord]] = {}
    try:
        for label, system, grammar in runs:
            w = writers[BASE_OF[system]]
            recs = []
            for it in items:
                rec = run_item(system, it, w, grammar=grammar)
                rec.system = label
                recs.append(rec)
            records[label] = recs
        metrics = _evaluate(
            records,
            items,
            real=real,
            resamples=args.resamples,
            seed=args.rng_seed,
            pairs=DEFAULT_PAIRS,
        )
        _write_outputs(out_dir, metrics, records, real=real)
    except CacheMiss as miss:
        _err(f"{miss}; the cache does not cover this run (run without --cached to fill it)")
        return 4
    except Exception as exc:  # noqa: BLE001 - type name only (NFR-P4)
        _err(f"eval failed: {type(exc).__name__}")
        return 1

    _print_summary(metrics)
    if args.assert_baseline is not None:
        failures = _assert_baseline(metrics, args.assert_baseline, args.tolerance)
        if failures:
            _err("baseline assertion failed:\n" + "\n".join(failures))
            return 5
        print("baseline: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
