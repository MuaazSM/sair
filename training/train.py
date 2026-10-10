"""Sair-Writer LoRA SFT on Tinker (PRD §12.3). Synthetic data only (NFR-P6).

    uv run python training/train.py --dry-run                       # plan + cost, no API
    (set -a; . ./.env.training; set +a; uv run python training/train.py --limit 100 --epochs 1)
    (set -a; . ./.env.training; set +a; uv run python training/train.py --lr-mult 0.5 ...)

* base ``TINKER_BASE_MODEL`` (default ``Qwen/Qwen3.5-4B``), LoRA rank 32, batch 32,
  ``--epochs`` <= 3, max length 1,024 (longer examples are DROPPED, never truncated, so the
  loss never sees a cut-off JSON), linear LR decay (tinker_cookbook default);
* LR = tinker_cookbook ``hyperparam_utils.get_lr(base)`` x ``--lr-mult``. Cookbook 0.1.0
  reads ``config.hidden_size``, which Qwen3.5 configs keep under ``text_config``; when
  ``get_lr`` fails the same formula is applied with that hidden size (recorded as
  ``lr_source``);
* loss on the assistant JSON only (prompt + empty think block weight 0), thinking off, via
  the renderer in ``training/tinker_common.py``; its config is written to
  ``training/renderer.json`` (P6 serves with exactly this);
* refuses to start unless ``training/LEXICON_HASH`` equals ``hub.grounding.lexicon_hash()``;
* per epoch: save state + sampler weights, then a DEV evaluation with the trained sampler
  (unsupported-detail rate, fact recall at span Jaccard >= 0.5, fallback rate, valid JSON
  without grammar); metadata in ``training/runs/<run_id>.json``.

Stdout carries counts, ids, metrics and costs only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hub.grounding import check_entry, lexicon_hash
from hub.prompts.builders import prompt_fingerprint
from hub.schemas import EntryOrigin, Fact, parse_writer_output
from training.freeze_lexicon import check_frozen
from training.tinker_common import (
    DATA_DIR,
    DEFAULT_BASE_MODEL,
    LEXICON_HASH_PATH,
    RENDERER_JSON,
    ROOT,
    RUNS_DIR,
    Usage,
    as_dicts,
    atomic_write_json,
    price_for,
    read_jsonl,
    renderer_config,
    write_jsonl,
    writer_messages,
)

LORA_RANK = 32
BATCH_SIZE = 32
MAX_EPOCHS = 3
MAX_LENGTH = 1024
EVAL_MAX_TOKENS = 400
EST_OUT_TOKENS = 220


class Refused(SystemExit):
    pass


# --------------------------------------------------------------------------- config


@dataclass
class TrainConfig:
    base_model: str = DEFAULT_BASE_MODEL
    lr_mult: float = 1.0
    epochs: int = MAX_EPOCHS
    batch_size: int = BATCH_SIZE
    lora_rank: int = LORA_RANK
    max_length: int = MAX_LENGTH
    limit: int | None = None
    dev_limit: int | None = None
    seed: int = 0
    lr_schedule: str = "linear"
    eval_temperature: float = 0.0
    eval_max_tokens: int = EVAL_MAX_TOKENS
    train_path: str = str(DATA_DIR / "sft_train.jsonl")
    dev_items_path: str = str(DATA_DIR / "dev.jsonl")
    phase_budget_usd: float = 2.50  # cap for the WHOLE training phase (all runs in RUNS_DIR)

    def validate(self) -> None:
        if not 1 <= self.epochs <= MAX_EPOCHS:
            raise Refused(f"epochs must be 1..{MAX_EPOCHS}")
        if self.max_length > MAX_LENGTH:
            raise Refused(f"max_length must be <= {MAX_LENGTH}")
        if self.lr_mult <= 0:
            raise Refused("lr_mult must be > 0")


def require_frozen_lexicon(path: Path | None = None) -> str:
    ok, code = check_frozen(path or LEXICON_HASH_PATH)
    if not ok:
        raise Refused(f"refusing to train: {code} (run training/freeze_lexicon.py after review)")
    return lexicon_hash()


def _hidden_size(model: str) -> int:
    from transformers.utils.hub import cached_file

    old = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        path = cached_file(model.split(":")[0], "config.json")
    finally:
        if old is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old
    if path is None:
        raise FileNotFoundError("config.json not in the local HF cache")
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    hs = cfg.get("hidden_size") or (cfg.get("text_config") or {}).get("hidden_size")
    if not isinstance(hs, int):
        raise ValueError("hidden_size not found")
    return hs


def resolve_lr(model: str) -> tuple[float, str]:
    """(lr, source). tinker_cookbook get_lr, or its formula on text_config.hidden_size."""
    try:
        from tinker_cookbook.hyperparam_utils import get_lr  # type: ignore[import-untyped]

        return float(get_lr(model.split(":")[0])), "tinker_cookbook.hyperparam_utils.get_lr"
    except Exception:  # noqa: BLE001 - cookbook 0.1.0 cannot read Qwen3.5 configs
        pass
    hs = _hidden_size(model)
    lr = 5e-05 * 10.0 * (2000 / hs) ** (0.0775 if "qwen" in model.lower() else 0.781)
    return lr, f"get_lr formula (cookbook 0.1.0) with text_config.hidden_size={hs}"


def write_renderer_config(
    tokenizer: Any, tokenizer_id: str, path: Path = RENDERER_JSON
) -> dict[str, Any]:
    cfg = renderer_config(tokenizer, tokenizer_id)
    atomic_write_json(path, cfg)
    return cfg


# --------------------------------------------------------------------------- data


@dataclass
class Example:
    item_id: str
    tokens: list[int]
    weights: list[float]

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


def build_examples(
    rows: list[dict[str, Any]], renderer: Any, max_length: int
) -> tuple[list[Example], int]:
    """Render rows with assistant-only weights; drop (never truncate) overlong rows."""
    out: list[Example] = []
    dropped = 0
    for row in rows:
        tokens, weights = renderer.build_supervised_example(row["messages"])
        if len(tokens) > max_length:
            dropped += 1
            continue
        out.append(
            Example(
                str((row.get("meta") or {}).get("item_id", "?")), tokens.tolist(), weights.tolist()
            )
        )
    return out, dropped


def to_datum(ex: Example) -> Any:
    import torch
    from tinker_cookbook.supervised.common import datum_from_tokens_weights  # type: ignore[import-untyped]

    return datum_from_tokens_weights(torch.tensor(ex.tokens), torch.tensor(ex.weights))


# --------------------------------------------------------------------------- dev eval


@dataclass
class DevMetrics:
    n: int = 0
    valid_json: int = 0
    entries: int = 0  # parsed outputs (denominator of unsupported rate)
    unsupported: int = 0
    fallback: int = 0
    gold_facts: int = 0
    recalled: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def rates(self) -> dict[str, float | None]:
        def r(a: int, b: int) -> float | None:
            return round(a / b, 4) if b else None

        return {
            "unsupported_rate": r(self.unsupported, self.entries),
            "fact_recall": r(self.recalled, self.gold_facts),
            "fallback_rate": r(self.fallback, self.n),
            "valid_json_no_grammar": r(self.valid_json, self.n),
        }


def score_dev(items: Sequence[dict[str, Any]], raws: Sequence[str]) -> DevMetrics:
    """Score first-attempt raw outputs (no grammar, no regen) against generator gold.

    unsupported_rate: parsed entries with >= 1 unsupported detail / parsed entries.
    fact_recall: gold facts matched by a verified predicted fact (eval.metrics.match_facts:
      span token Jaccard >= 0.5 and subject match) / gold facts; unparsed outputs recall 0.
    fallback_rate: outputs that would fall back to the template (unparseable or §8 fail) / n.
    valid_json_no_grammar: outputs that parse into the §9.1 schema / n.
    """
    from eval.metrics import match_facts

    m = DevMetrics()
    for item, raw in zip(items, raws, strict=True):
        m.n += 1
        gold = [Fact.model_validate(f) for f in item["gold"].get("facts") or []]
        m.gold_facts += len(gold)
        parsed = parse_writer_output(raw.strip())
        if parsed.value is None:
            m.fallback += 1
            for c in parsed.reasons:
                m.reasons[c.value] = m.reasons.get(c.value, 0) + 1
            continue
        m.valid_json += 1
        chk = check_entry(
            parsed.value, str(item["transcript"]), str(item["group"]), origin=EntryOrigin.MODEL
        )
        m.entries += 1
        m.unsupported += int(bool(chk.unsupported_details))
        m.fallback += int(not chk.passed)
        m.recalled += len(match_facts(list(chk.verified_facts), gold))  # type: ignore[arg-type]
        for c in chk.reasons:
            m.reasons[c.value] = m.reasons.get(c.value, 0) + 1
    return m


def run_dev_eval(
    items: Sequence[dict[str, Any]],
    prompt_ids: Callable[[list[dict[str, str]]], list[int]],
    submit: Callable[[list[int]], Callable[[], tuple[str, int]]],
    usage: Usage,
) -> tuple[DevMetrics, list[str]]:
    futs = []
    for it in items:
        ids = prompt_ids(as_dicts(writer_messages(it)))
        futs.append((len(ids), submit(ids)))
    raws: list[str] = []
    for n_prompt, fin in futs:
        text, n_out = fin()
        usage.add_sample(n_prompt, 0, 1, n_out)
        raws.append(text)
    return score_dev(items, raws), raws


# --------------------------------------------------------------------------- plan / cost


def estimate_cost(
    train_tokens_per_epoch: int,
    epochs: int,
    dev_prompt_tokens: int,
    dev_items: int,
    model: str,
    est_out: int = EST_OUT_TOKENS,
) -> dict[str, Any]:
    u = Usage(model)
    u.train_tokens = train_tokens_per_epoch * epochs
    for _ in range(epochs):
        if dev_items:
            per = dev_prompt_tokens // dev_items
            for _ in range(dev_items):
                u.add_sample(per, 0, 1, est_out)
    d = u.to_dict()
    return {
        "train_tokens": u.train_tokens,
        "eval_requests": u.requests,
        "eval_prefill_tokens": u.prefill_tokens,
        "eval_sample_tokens": u.sample_tokens,
        "usd": d["estimated_cost_usd"],
    }


def plan(
    cfg: TrainConfig,
    n_train: int,
    mean_tokens: float,
    n_dev: int,
    mean_dev_prompt: float,
    smoke_items: int = 100,
    sweep_items: int = 100,
    sweep_dev: int = 50,
) -> dict[str, Any]:
    """Cost of the whole P5 plan: smoke + 3-LR sweep + full run (<= cfg.epochs)."""
    m = cfg.base_model

    def run(items: int, epochs: int, dev: int) -> dict[str, Any]:
        return estimate_cost(
            round(items * mean_tokens), epochs, round(dev * mean_dev_prompt), dev, m
        )

    smoke = run(min(smoke_items, n_train), 1, min(sweep_dev, n_dev))
    sweep = [run(min(sweep_items, n_train), 1, min(sweep_dev, n_dev)) for _ in (0.5, 1.0, 2.0)]
    full = run(n_train, cfg.epochs, n_dev)
    total = smoke["usd"] + sum(s["usd"] for s in sweep) + full["usd"]
    return {
        "base_model": m,
        "train_items": n_train,
        "mean_example_tokens": round(mean_tokens, 1),
        "dev_items": n_dev,
        "steps_per_epoch_full": math.ceil(n_train / cfg.batch_size),
        "smoke_100x1": smoke,
        "sweep_3x(100x1)": sweep,
        f"full_{cfg.epochs}_epochs": full,
        "total_usd": round(total, 4),
        "note": "checkpoint storage not included (rate not on the token price card)",
    }


# --------------------------------------------------------------------------- budget


def phase_spend(runs_dir: Path = RUNS_DIR, exclude: str | None = None) -> float:
    """Estimated spend of every training run recorded in ``runs_dir`` (run metadata files)."""
    total = 0.0
    if not runs_dir.is_dir():
        return 0.0
    for p in sorted(runs_dir.glob("*.json")):
        if p.name.startswith("_") or p.stem == exclude:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        total += float((d.get("usage") or {}).get("estimated_cost_usd") or 0.0)
    return total


def dev_subset(
    items: list[dict[str, Any]], limit: int | None, seed: int = 0
) -> list[dict[str, Any]]:
    """Deterministic random subset (file order is grouped by seed scenario)."""
    if not limit or limit >= len(items):
        return items
    idx = sorted(random.Random(seed).sample(range(len(items)), limit))
    return [items[i] for i in idx]


# --------------------------------------------------------------------------- selection


def epoch_constraint_codes(dev: dict[str, Any] | None, c_ref: dict[str, Any]) -> list[str]:
    """Why an epoch is (in)eligible under §12.3 vs the base few-shot (C) dev reference."""
    eps = 1e-9
    if not dev:
        return ["NO_DEV_METRICS"]
    codes: list[str] = []
    u, r, f = dev.get("unsupported_rate"), dev.get("fact_recall"), dev.get("fallback_rate")
    if u is None or r is None or f is None:
        codes.append("METRIC_MISSING")
    if r is not None and c_ref.get("fact_recall") is not None and r + eps < c_ref["fact_recall"]:
        codes.append("RECALL_BELOW_C")
    if (
        f is not None
        and c_ref.get("fallback_rate") is not None
        and f - eps > c_ref["fallback_rate"]
    ):
        codes.append("FALLBACK_ABOVE_C")
    return codes or ["ELIGIBLE"]


def shortcut_check(items: Sequence[dict[str, Any]], raws: Sequence[str]) -> dict[str, Any]:
    """P5 shortcut check on dev outputs: fixed fact counts, copied mission nouns, dropped hedges.

    * fact counts: distribution of predicted (raw) fact counts, the modal share, and Spearman
      rho between predicted and gold counts (a fixed-count shortcut gives a high modal share
      and a low rho);
    * copied mission nouns: share of parsed entries containing a lexicon detail that occurs in
      the mission text but NOT in the transcript;
    * dropped hedges: among gold-unsure facts matched by a predicted verified-or-dropped fact
      (span Jaccard >= 0.5 + subject match), the share predicted as sure; plus the share of
      parsed outputs with HEDGE_MISSING_IN_ENTRY / HEDGE_CERTAINTY_MISMATCH.
    Counts and rates only (no text).
    """
    from eval.metrics import match_facts
    from hub.grounding import extract_details
    from training.audit import spearman

    pred_n: list[int] = []
    gold_n: list[int] = []
    copied = parsed = hedge_miss = hedge_mm = 0
    unsure_matched = unsure_sure = 0
    for item, raw in zip(items, raws, strict=True):
        p = parse_writer_output(raw.strip())
        if p.value is None:
            continue
        parsed += 1
        out = p.value
        gold = [Fact.model_validate(f) for f in item["gold"].get("facts") or []]
        pred_n.append(len(out.facts))
        gold_n.append(len(gold))
        mission = item.get("mission_text") or ""
        if mission:
            t_keys = extract_details(str(item["transcript"]), inclusive=True)
            m_keys = {k for k in extract_details(mission, inclusive=True) if ":" not in k}
            e_keys = extract_details(out.entry)
            copied += int(bool((m_keys - t_keys) & e_keys))
        chk = check_entry(
            out, str(item["transcript"]), str(item["group"]), origin=EntryOrigin.MODEL
        )
        codes = {c.value for c in chk.reasons}
        hedge_miss += int("hedge_missing_in_entry" in codes)
        hedge_mm += int("hedge_certainty_mismatch" in codes)
        gold_unsure = [g for g in gold if g.certainty.value == "unsure"]
        for i, _j in match_facts(list(out.facts), gold_unsure):  # type: ignore[arg-type]
            unsure_matched += 1
            unsure_sure += int(out.facts[i].certainty.value == "sure")
    dist: dict[int, int] = {}
    for n in pred_n:
        dist[n] = dist.get(n, 0) + 1
    modal = max(dist.values()) / len(pred_n) if pred_n else None
    rho = spearman(pred_n, gold_n)

    def r(a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None

    return {
        "parsed": parsed,
        "pred_fact_count_dist": dict(sorted(dist.items())),
        "pred_fact_count_modal_share": None if modal is None else round(modal, 4),
        "rho_pred_vs_gold_fact_count": None if rho is None else round(rho, 4),
        "copied_mission_noun_rate": r(copied, parsed),
        "gold_unsure_matched": unsure_matched,
        "gold_unsure_predicted_sure_rate": r(unsure_sure, unsure_matched),
        "hedge_missing_in_entry_rate": r(hedge_miss, parsed),
        "hedge_certainty_mismatch_rate": r(hedge_mm, parsed),
    }


def select_run(run_id: str, c_ref_path: Path) -> dict[str, Any]:
    """§12.3 selection vs C's dev numbers + shortcut check; written into the run file."""
    from training.gates import select_epoch

    path = RUNS_DIR / f"{run_id}.json"
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    c = json.loads(c_ref_path.read_text(encoding="utf-8"))
    c_ref = c["dev"]
    sel = select_epoch(meta["epochs"], c_ref)
    meta["selection"] = {
        "rule": "min dev unsupported_rate s.t. fact_recall >= C and fallback_rate <= C "
        "(PRD §12.3; ties -> earlier epoch; training.gates.select_epoch)",
        "c_ref": {
            k: c_ref.get(k)
            for k in ("unsupported_rate", "fact_recall", "fallback_rate", "valid_json_no_grammar")
        },
        "c_dev_n": c.get("dev_n"),
        "per_epoch": {
            str(ep["epoch"]): epoch_constraint_codes(ep.get("dev"), c_ref) for ep in meta["epochs"]
        },
    }
    meta["selected_epoch"] = sel
    meta["selection"]["reason"] = (
        f"epoch {sel}: lowest unsupported_rate among eligible epochs"
        if sel is not None
        else "NO_EPOCH_ELIGIBLE (G-ship would fail: WRITER_BACKEND=gemma)"
    )
    dev_items = {
        str(it["item_id"]): it for it in read_jsonl(Path(meta["config"]["dev_items_path"]))
    }
    sc: dict[str, Any] = {}
    for ep in meta["epochs"]:
        f = RUNS_DIR / f"{run_id}.dev_e{ep['epoch']}.jsonl"
        if f.is_file():
            rows = read_jsonl(f)
            sc[str(ep["epoch"])] = shortcut_check(
                [dev_items[str(r["item_id"])] for r in rows], [str(r["raw"]) for r in rows]
            )
    meta["shortcut_check"] = sc
    atomic_write_json(path, meta)
    return meta


# --------------------------------------------------------------------------- train


def train(cfg: TrainConfig, run_id: str) -> dict[str, Any]:  # pragma: no cover - network
    import tinker
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier  # type: ignore[import-untyped]

    from training.tinker_common import load_tokenizer, make_renderer

    lex = require_frozen_lexicon()
    tok = load_tokenizer(cfg.base_model)
    rend = make_renderer(tok)
    rcfg = write_renderer_config(tok, cfg.base_model)
    rows = dev_subset(read_jsonl(Path(cfg.train_path)), cfg.limit, cfg.seed)  # seeded sample
    examples, dropped = build_examples(rows, rend, cfg.max_length)
    dev_items = dev_subset(read_jsonl(Path(cfg.dev_items_path)), cfg.dev_limit)
    prior = phase_spend(exclude=run_id)
    rates, _ = price_for(cfg.base_model)
    base_lr, lr_source = resolve_lr(cfg.base_model)
    lr = base_lr * cfg.lr_mult
    steps_per_epoch = math.ceil(len(examples) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    usage = Usage(cfg.base_model)
    meta: dict[str, Any] = {
        "run_id": run_id,
        "config": asdict(cfg),
        "lr": lr,
        "base_lr": base_lr,
        "lr_source": lr_source,
        "lexicon_hash": lex,
        "prompt_fingerprint": prompt_fingerprint("writer"),
        "renderer": rcfg,
        "examples": len(examples),
        "dropped_overlong": dropped,
        "epochs": [],
        "phase_spend_before_usd": round(prior, 6),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    def over_budget(extra_usd: float) -> bool:
        return prior + usage.cost() + extra_usd > cfg.phase_budget_usd

    meta_path = RUNS_DIR / f"{run_id}.json"
    atomic_write_json(meta_path, meta)

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(
        base_model=cfg.base_model,
        rank=cfg.lora_rank,
        seed=cfg.seed,
        user_metadata={"run_id": run_id, "data": "synthetic"},
    )
    rng = random.Random(cfg.seed)
    step = 0
    for epoch in range(cfg.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        nlls: list[float] = []
        for b in range(steps_per_epoch):
            batch = [examples[i] for i in order[b * cfg.batch_size : (b + 1) * cfg.batch_size]]
            if not batch:
                continue
            step_cost = sum(e.n_tokens for e in batch) * rates["train"] / 1e6
            if over_budget(step_cost):
                meta["stopped"] = "PHASE_BUDGET"
                break
            data = [to_datum(e) for e in batch]
            mult = compute_schedule_lr_multiplier(cfg.lr_schedule, step, total_steps)
            fb = tc.forward_backward(data, loss_fn="cross_entropy")
            op = tc.optim_step(
                tinker.AdamParams(learning_rate=lr * mult, beta1=0.9, beta2=0.95, eps=1e-8)
            )
            res = fb.result()
            op.result()
            usage.train_tokens += sum(e.n_tokens for e in batch)
            nlls.append(
                compute_mean_nll(
                    [x["logprobs"] for x in res.loss_fn_outputs],
                    [d.loss_fn_inputs["weights"] for d in data],
                )
            )
            step += 1
            print(f"epoch={epoch + 1} step={step}/{total_steps} nll={nlls[-1]:.4f}")
        if meta.get("stopped"):
            print(f"stopped: phase budget {cfg.phase_budget_usd} USD would be exceeded")
            break
        name = f"{run_id}-e{epoch + 1}"
        state_path = tc.save_state(name).result().path
        sampler_path = tc.save_weights_for_sampler(name).result().path
        samp = sc.create_sampling_client(model_path=sampler_path)
        stop = rend.get_stop_sequences()

        def submit(
            ids: list[int], _s: Any = samp, _stop: list[int] = stop
        ) -> Callable[[], tuple[str, int]]:
            fut = _s.sample(
                prompt=tinker.ModelInput.from_ints(ids),
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=cfg.eval_max_tokens, temperature=cfg.eval_temperature, stop=_stop
                ),
            )

            def fin() -> tuple[str, int]:
                seq = fut.result().sequences[0]
                toks = list(seq.tokens)
                n = len(toks)
                if toks and toks[-1] in _stop:
                    toks = toks[:-1]
                return tok.decode(toks, skip_special_tokens=False), n

            return fin

        eval_cost = (
            len(dev_items) * (600 * rates["prefill"] + cfg.eval_max_tokens * rates["sample"]) / 1e6
        )
        if over_budget(eval_cost):
            meta["stopped"] = "PHASE_BUDGET_BEFORE_EVAL"
            meta["epochs"].append(
                {
                    "epoch": epoch + 1,
                    "state_path": state_path,
                    "sampler_path": sampler_path,
                    "dev": None,
                }
            )
            break
        dm, raws = run_dev_eval(dev_items, rend.prompt_tokens, submit, usage)
        # synthetic dev outputs, local eval output only (never logged)
        write_jsonl(
            RUNS_DIR / f"{run_id}.dev_e{epoch + 1}.jsonl",
            ({"item_id": it["item_id"], "raw": r} for it, r in zip(dev_items, raws, strict=True)),
        )
        ep = {
            "epoch": epoch + 1,
            "state_path": state_path,
            "sampler_path": sampler_path,
            "train_nll_mean": round(sum(nlls) / len(nlls), 5) if nlls else None,
            "dev": dm.rates(),
            "dev_n": dm.n,
            "dev_reasons": dm.reasons,
        }
        meta["epochs"].append(ep)
        meta["usage"] = usage.to_dict()
        atomic_write_json(meta_path, meta)
        print(json.dumps({k: v for k, v in ep.items() if k != "dev_reasons"}))
    meta["usage"] = usage.to_dict()
    meta["phase_spend_after_usd"] = round(prior + usage.cost(), 6)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_write_json(meta_path, meta)
    return meta


def _mean_tokens_from(cfg: TrainConfig, rend: Any) -> tuple[int, float, str]:
    """(n_train, mean tokens per example, source) for the dry-run plan."""
    p = Path(cfg.train_path)
    if p.is_file():
        rows = read_jsonl(p)[: cfg.limit] if cfg.limit else read_jsonl(p)
        ex, _ = build_examples(rows, rend, 10**9)
        if ex:
            return len(ex), sum(e.n_tokens for e in ex) / len(ex), f"measured on {p.name}"
    items_p = DATA_DIR / "train.jsonl"
    if items_p.is_file():
        items = read_jsonl(items_p)[: cfg.limit] if cfg.limit else read_jsonl(items_p)
        pt = [len(rend.prompt_tokens(as_dicts(writer_messages(it)))) for it in items]
        return (
            len(items),
            sum(pt) / len(pt) + EST_OUT_TOKENS + 1,
            f"train.jsonl prompts + {EST_OUT_TOKENS} est. out",
        )
    return 1040, 700.0, "assumed (no data yet): 1040 items x 700 tokens"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sair-Writer LoRA SFT on Tinker (PRD §12.3)")
    ap.add_argument(
        "--base-model", default=os.environ.get("TINKER_BASE_MODEL") or DEFAULT_BASE_MODEL
    )
    ap.add_argument("--lr-mult", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dev-limit", type=int, default=None)
    ap.add_argument("--train", default=str(DATA_DIR / "sft_train.jsonl"))
    ap.add_argument("--dev-items", default=str(DATA_DIR / "dev.jsonl"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase-budget-usd", type=float, default=2.50)
    ap.add_argument("--select", metavar="RUN_ID", default=None, help="§12.3 selection + shortcuts")
    ap.add_argument("--c-ref", type=Path, default=ROOT / "training" / "gates" / "c_dev.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--assume-train-items", type=int, default=None, help="dry run: scale plan")
    ap.add_argument("--assume-dev-items", type=int, default=None, help="dry run: scale plan")
    a = ap.parse_args(argv)
    if a.select:
        m = select_run(a.select, a.c_ref)
        print(
            json.dumps(
                {k: m[k] for k in ("selected_epoch", "selection", "shortcut_check")}, indent=1
            )
        )
        return 0
    cfg = TrainConfig(
        base_model=a.base_model,
        lr_mult=a.lr_mult,
        epochs=a.epochs,
        batch_size=a.batch_size,
        limit=a.limit,
        dev_limit=a.dev_limit,
        seed=a.seed,
        train_path=a.train,
        dev_items_path=a.dev_items,
        phase_budget_usd=a.phase_budget_usd,
    )
    cfg.validate()
    run_id = a.run_id or f"sft-{time.strftime('%Y%m%d-%H%M%S')}-lr{a.lr_mult:g}"

    if a.dry_run:
        from training.tinker_common import load_tokenizer, make_renderer

        ok, code = check_frozen(LEXICON_HASH_PATH)
        tok = load_tokenizer(cfg.base_model)
        rend = make_renderer(tok)
        rc = write_renderer_config(tok, cfg.base_model)
        base_lr, src = resolve_lr(cfg.base_model)
        n_train, mean_tok, source = _mean_tokens_from(cfg, rend)
        dev_p = Path(cfg.dev_items_path)
        if dev_p.is_file():
            dev = read_jsonl(dev_p)[: cfg.dev_limit] if cfg.dev_limit else read_jsonl(dev_p)
            dpt = [len(rend.prompt_tokens(as_dicts(writer_messages(it)))) for it in dev]
            n_dev, mean_dev = len(dev), (sum(dpt) / len(dpt) if dpt else 0.0)
        else:
            n_dev, mean_dev = 130, 480.0
        if a.assume_train_items:
            source += f"; scaled to {a.assume_train_items} train items"
            n_train = a.assume_train_items
        if a.assume_dev_items:
            n_dev = a.assume_dev_items
        rates, known = price_for(cfg.base_model)
        print(
            f"run_id={run_id} base={cfg.base_model} rank={cfg.lora_rank} batch={cfg.batch_size} "
            f"epochs={cfg.epochs} max_length={cfg.max_length}"
        )
        print(f"lr={base_lr * cfg.lr_mult:.3e} (base {base_lr:.3e} x {cfg.lr_mult:g}; {src})")
        print(f"lexicon: {code} current={lexicon_hash()[:16]} (a real run refuses unless OK)")
        print(
            f"renderer: {rc['renderer']} -> {RENDERER_JSON.relative_to(RENDERER_JSON.parents[1])} "
            f"stop_token_ids={rc.get('stop_token_ids')}"
        )
        print(f"data: {source}; dev items={n_dev} mean dev prompt tokens={mean_dev:.0f}")
        print(f"rates (USD/Mtok, known={known}): {rates}")
        print(json.dumps(plan(cfg, n_train, mean_tok, n_dev, mean_dev), indent=1))
        return 0

    train(cfg, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
