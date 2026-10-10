"""Teacher entries on Tinker (PRD §12.1.3): 4 candidates per item, keep the first passing §8.

    (set -a; . ./.env.training; set +a; uv run python training/teacher.py --split train)
    uv run python training/teacher.py --split all --dry-run        # no API: tokens + cost

SYNTHETIC DATA ONLY (NFR-P6): input is the generator's ``training/data/{split}.jsonl`` (or
``--items`` for a hand-written synthetic file). The prompt is exactly the production writer
prompt (``hub.prompts.builders.build_writer_prompt``), rendered with the Qwen3.5 chat
template with thinking off (``training/tinker_common.py``).

Per item, in order, each candidate goes through ``parse_writer_output`` and then
``check_entry(origin=MODEL)``; the first that passes is the gold label, cleaned to what the
checker verified (dropped facts removed, check-7 mission coercion applied, then re-checked).
If none passes, the gold is the code-built template (``template_output``) over the item's
generator gold facts (which the generator already validated with the same checker).

Teacher-only options (never part of the SFT row, which always holds the PLAIN production
prompt + the chosen JSON): ``--regen-round`` re-samples items with no passing candidate
using ``build_writer_prompt(..., regen=<codes of the best failing candidate>)`` (fixed
instruction strings from ``hub.schemas``; previous output text is never sent back);
``--teacher-addendum`` appends the fixed ``TEACHER_ADDENDUM`` rules to the teacher's system
prompt; ``--teacher-glossary`` appends a code-built glossary from the frozen lexicon (English
word for each non-English lexicon term in the transcript, e.g. "चील = kite"). Per item the
report records which round won (``r1:cK``, ``r2:cK`` or ``template``).
``--budget-usd`` stops launching requests when this invocation's estimated spend (published
rates, worst case ``max_tokens`` per in-flight candidate) would exceed the cap.

Raw candidates are cached per item in ``training/cache/teacher/<item_id>.json`` (resume-safe;
a cache entry is reused only for the same model, prompt fingerprint and sampling params).
``_cost.json`` holds the running token usage and the estimated spend (published rate card).
Stdout carries counts, ids, codes and costs only; never transcript, span or entry text.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hub.grounding import (
    check_entry,
    extract_details,
    lexicon_hash,
    normalise,
    template_output,
)
from hub.prompts.builders import (
    MissionText,
    Transcript,
    WalkerGroup,
    build_writer_prompt,
    prompt_fingerprint,
)
from hub.schemas import (
    GROUNDING_CODES,
    EntryOrigin,
    Fact,
    MissionResult,
    ReasonCode,
    WriterOutput,
    parse_writer_output,
)
from training.tinker_common import (
    DATA_DIR,
    DEFAULT_TEACHER_MODEL,
    TEACHER_CACHE,
    Usage,
    as_dicts,
    atomic_write_json,
    gold_json,
    price_for,
    read_jsonl,
    render_prompt_text,
    safe_id,
    write_jsonl,
    writer_messages,
)

SPLITS = ("train", "dev", "test")
DEFAULT_CANDIDATES = 4
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 400  # PRD §9.1
DEFAULT_EST_OUT_TOKENS = 220  # dry-run assumption per candidate (replace with smoke mean)
TEMPLATE = "template"
UNUSABLE = "unusable"
PENDING = "pending"  # budget/request cap hit before this item's rounds finished

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.S)


# --------------------------------------------------------------------------- sampler


@dataclass
class SampleResult:
    texts: list[str]
    out_tokens: list[int]  # generated token count per candidate
    stop_reasons: list[str]
    prompt_tokens: int
    cached_prompt_tokens: int = 0


class Sampler(Protocol):
    model: str

    def prompt_ids(self, messages: list[dict[str, str]]) -> list[int]: ...

    def submit(
        self, prompt_ids: list[int], n: int, temperature: float, max_tokens: int
    ) -> Callable[[], SampleResult]: ...


class TinkerSampler:
    """Real sampler: Tinker SamplingClient on the teacher base model."""

    def __init__(self, model: str, tokenizer_id: str | None = None, renderer: str = "auto") -> None:
        import tinker

        from training.tinker_common import load_tokenizer

        self.model = model
        self._tinker = tinker
        self.tokenizer = load_tokenizer(tokenizer_id or model)
        self._prompt_ids, self.stop_ids = teacher_prompt_fn(self.tokenizer, model, renderer)
        self.client = tinker.ServiceClient().create_sampling_client(base_model=model)

    def prompt_ids(self, messages: list[dict[str, str]]) -> list[int]:
        return self._prompt_ids(messages)

    def submit(
        self, prompt_ids: list[int], n: int, temperature: float, max_tokens: int
    ) -> Callable[[], SampleResult]:
        t = self._tinker
        fut = self.client.sample(
            prompt=t.ModelInput.from_ints(prompt_ids),
            num_samples=n,
            sampling_params=t.SamplingParams(
                max_tokens=max_tokens, temperature=temperature, stop=self.stop_ids
            ),
        )

        def finish() -> SampleResult:
            resp = fut.result()
            texts, outs, stops = [], [], []
            for seq in resp.sequences:
                toks = list(seq.tokens)
                outs.append(len(toks))
                stops.append(str(getattr(seq, "stop_reason", "?")))
                if toks and toks[-1] in self.stop_ids:
                    toks = toks[:-1]
                texts.append(self.tokenizer.decode(toks, skip_special_tokens=False))
            return SampleResult(
                texts=texts,
                out_tokens=outs,
                stop_reasons=stops,
                prompt_tokens=len(prompt_ids),
                cached_prompt_tokens=int(getattr(resp, "prompt_cache_hit_tokens", 0) or 0),
            )

        return finish


def teacher_prompt_fn(
    tokenizer: Any, model: str, renderer: str = "auto"
) -> tuple[Callable[[list[dict[str, str]]], list[int]], list[int]]:
    """(prompt -> ids, stop ids). Qwen3.5 teachers use our renderer; others ("hf") use the
    model's own HF chat template with ``enable_thinking=False`` and stop on its EOS."""
    from training.tinker_common import IM_END, make_renderer

    if renderer == "auto":
        renderer = "qwen3_5" if "qwen3.5" in model.lower() else "hf"
    if renderer == "qwen3_5":
        rend = make_renderer(tokenizer)
        return rend.prompt_tokens, list(rend.get_stop_sequences())
    if renderer != "hf":
        raise ValueError("renderer must be auto, qwen3_5 or hf")

    def ids(messages: list[dict[str, str]]) -> list[int]:
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if hasattr(out, "keys"):
            out = out["input_ids"]
        return [int(x) for x in out]

    stops = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    if isinstance(im_end, int) and im_end != getattr(tokenizer, "unk_token_id", None):
        stops.add(im_end)
    return ids, sorted(stops)


# --------------------------------------------------------------------------- selection


@dataclass
class CandidateEval:
    parse_ok: bool
    passed: bool
    codes: list[str]
    dropped_facts: int = 0
    fenced: bool = False


@dataclass
class Selection:
    winner: int | str  # candidate index, "template" or "unusable"
    gold: WriterOutput | None
    evals: list[CandidateEval] = field(default_factory=list)  # round 1
    gold_codes: list[str] = field(default_factory=list)  # template/unusable diagnostics
    round: int | None = None  # 1 or 2 when a candidate won
    evals_r2: list[CandidateEval] = field(default_factory=list)  # teacher-only regen round
    regen_codes: list[str] = field(default_factory=list)  # codes sent to round 2

    @property
    def label(self) -> str:
        if isinstance(self.winner, int):
            return f"r{self.round or 1}:c{self.winner}"
        return str(self.winner)


def _unfence(raw: str) -> tuple[str, bool]:
    m = _FENCE.match(raw)
    return (m.group(1), True) if m else (raw.strip(), False)


def evaluate_candidate(
    raw: str, transcript: str, group: str
) -> tuple[CandidateEval, WriterOutput | None]:
    """parse -> check_entry(MODEL) -> cleaned (verified facts, coerced mission) -> recheck."""
    text, fenced = _unfence(raw)
    parsed = parse_writer_output(text)
    if parsed.value is None:
        return CandidateEval(False, False, [c.value for c in parsed.reasons], fenced=fenced), None
    chk = check_entry(parsed.value, transcript, group, origin=EntryOrigin.MODEL)
    codes = [c.value for c in chk.reasons]
    if not chk.passed:
        return CandidateEval(True, False, codes, chk.dropped_facts, fenced), None
    cleaned = WriterOutput(
        facts=list(chk.verified_facts),
        mission_result=chk.mission_result,
        mission_span=chk.mission_span,
        entry=parsed.value.entry,
    )
    re_chk = check_entry(cleaned, transcript, group, origin=EntryOrigin.MODEL)
    if not re_chk.passed or re_chk.reasons:
        return CandidateEval(
            True, False, codes + ["RECHECK_FAILED"], chk.dropped_facts, fenced
        ), None
    return CandidateEval(True, True, codes, chk.dropped_facts, fenced), cleaned


def template_gold(item: dict[str, Any]) -> tuple[WriterOutput | None, list[str]]:
    """§12.1.3 fallback: template over the item's generator gold facts (checks 1-3, 7 in code)."""
    g = item["gold"]
    try:
        facts = [Fact.model_validate(f) for f in g.get("facts") or []]
        base = WriterOutput(
            facts=facts,
            mission_result=MissionResult(g.get("mission_result") or "not_mentioned"),
            mission_span=g.get("mission_span"),
            entry="",
        )
    except (ValueError, TypeError):
        return None, ["GOLD_SCHEMA_INVALID"]
    transcript, group = str(item["transcript"]), str(item["group"])
    chk = check_entry(base, transcript, group, origin=EntryOrigin.TEMPLATE)
    out = template_output(group, chk)
    final = check_entry(out, transcript, group, origin=EntryOrigin.TEMPLATE)
    codes = [c.value for c in chk.reasons]
    if not final.passed:
        return None, codes + ["TEMPLATE_FAILED"] + [c.value for c in final.reasons]
    return out, codes


def select_gold(item: dict[str, Any], candidates: Sequence[str]) -> Selection:
    transcript, group = str(item["transcript"]), str(item["group"])
    evals: list[CandidateEval] = []
    for i, raw in enumerate(candidates):
        ev, cleaned = evaluate_candidate(raw, transcript, group)
        evals.append(ev)
        if cleaned is not None:
            return Selection(i, cleaned, evals, round=1)
    out, codes = template_gold(item)
    if out is None:
        return Selection(UNUSABLE, None, evals, codes)
    return Selection(TEMPLATE, out, evals, codes)


# --------------------------------------------------------------------------- cache


def prompt_fp(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(render_prompt_text(messages).encode("utf-8")).hexdigest()


def load_cache(path: Path, key: dict[str, Any]) -> list[str]:
    """Cached candidate texts for this exact (model, prompt, params) key; else []."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if any(data.get(k) != v for k, v in key.items()):
        return []
    return [str(c["text"]) for c in data.get("candidates", [])]


def save_cache(path: Path, key: dict[str, Any], cands: list[dict[str, Any]]) -> None:
    atomic_write_json(path, {**key, "candidates": cands})


# --------------------------------------------------------------------------- run

# TEACHER-ONLY system addendum (behind --teacher-addendum). Fixed text, no content. The SFT
# training row always keeps the plain production prompt (sft_row uses writer_messages).
TEACHER_ADDENDUM = (
    "Extra rules for this reply:\n"
    "- Each span is copied character for character from the transcript: same words, same "
    "script, same spelling. Do not shorten a span in the middle of what it says.\n"
    "- Choose spans long enough to contain the subject and the place. The subject and the "
    "place must be words that appear inside that same span; if the place is not in the span, "
    "leave place out.\n"
    "- In the entry, only name things (birds, trees, animals, colours, weather, places, "
    "numbers) that appear in a fact's span, subject or place. If the transcript is in Hindi, "
    "you may translate a fact's subject into English in the entry, but add nothing else."
)


@functools.cache
def _english_words() -> dict[str, tuple[str, tuple[str, ...]]]:
    """lexicon id -> (English word = first alias, other aliases), from data/lexicon.yaml."""
    import yaml

    from hub.grounding import DEFAULT_LEXICON

    raw = yaml.safe_load(DEFAULT_LEXICON.read_text(encoding="utf-8"))
    out: dict[str, tuple[str, tuple[str, ...]]] = {}
    for e in raw.get("entries") or []:
        aliases = [str(a) for a in e.get("aliases") or []]
        if aliases and e.get("category") != "number":
            out[str(e["id"])] = (aliases[0], tuple(aliases[1:]))
    return out


# Devanagari aliases never glossed: Hindi function words ("बस" = "that's all") and terms
# whose lexicon id is one of several meanings (धूप: lexicon "incense", usually "sunshine").
_GLOSSARY_STOP = frozenset({"बस", "धूप"})
_DEVA_RE = re.compile(r"[\u0900-\u097F]")


def teacher_glossary(transcript: str) -> str | None:
    """TEACHER-ONLY: code-built glossary (from the frozen lexicon) of the English word for
    each Devanagari lexicon term in this transcript, e.g. "चील = kite". None if empty.

    Devanagari aliases only: romanised/English aliases are skipped because they are mostly
    English inflections ("sunny" -> sun) or ambiguous ("dhoop" -> incense), so the glossary
    is a no-op for English and Hinglish (Latin-script) transcripts."""
    found = extract_details(transcript, inclusive=True)
    t_norm = normalise(transcript)
    pairs: list[str] = []
    for lid, (english, others) in sorted(_english_words().items()):
        if lid not in found:
            continue
        here = [
            a
            for a in others
            if _DEVA_RE.search(a)
            and not re.search(r"[A-Za-z]", a)
            and normalise(a) in t_norm
            and a not in _GLOSSARY_STOP
        ]
        if here:
            pairs.append(f"{' / '.join(here[:3])} = {english}")
    if not pairs:
        return None
    return (
        "Glossary for this note (English words to use in the entry for things it names; "
        "every thing you name in the entry must also be one of your facts): " + "; ".join(pairs)
    )


@dataclass
class RunConfig:
    model: str = DEFAULT_TEACHER_MODEL
    candidates: int = DEFAULT_CANDIDATES
    first_stage: int = DEFAULT_CANDIDATES  # < candidates => staged sampling (cost saver)
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    concurrency: int = 8
    max_requests: int | None = None  # hard cap on API requests in this invocation
    regen_round: bool = False  # teacher-only round 2 with fixed regen instructions
    regen_candidates: int = 2
    addendum: bool = False  # teacher-only system addendum (TEACHER_ADDENDUM)
    glossary: bool = False  # teacher-only lexicon glossary (teacher_glossary)
    budget_usd: float | None = None  # cap on estimated spend in this invocation
    raise_on_stop: bool = True  # False (CLI): report finished items, mark the rest pending


def teacher_messages(
    item: dict[str, Any],
    regen: Sequence[ReasonCode] = (),
    addendum: bool = False,
    glossary: bool = False,
) -> list[dict[str, str]]:
    """Teacher prompt: the production builder (optionally with its fixed regen block) plus
    the optional teacher-only addendum. Never contains previous model output."""
    mission = item.get("mission_text")
    msgs = as_dicts(
        build_writer_prompt(
            WalkerGroup(str(item["group"])),
            MissionText(str(mission)) if mission else None,
            Transcript(str(item["transcript"])),
            regen=list(regen),
        )
    )
    extra = [TEACHER_ADDENDUM] if addendum else []
    if glossary and (g := teacher_glossary(str(item["transcript"]))):
        extra.append(g)
    if extra:
        msgs[0] = {
            "role": "system",
            "content": msgs[0]["content"].rstrip() + "\n\n" + "\n\n".join(extra),
        }
    return msgs


def _cache_key(cfg: RunConfig, item_id: str, fp: str, rnd: int) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "model": cfg.model,
        "round": rnd,
        "addendum": cfg.addendum,
        "glossary": cfg.glossary,
        "prompt_fp": fp,
        "prompt_version_fp": prompt_fingerprint("writer"),
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }


def _cache_path(cache_dir: Path, iid: str, rnd: int, cfg: RunConfig) -> Path:
    tag = (".add" if cfg.addendum else "") + (".gl" if cfg.glossary else "")
    return cache_dir / (iid + tag + (".r2" if rnd == 2 else "") + ".json")


class Budget(Exception):
    """Raised when --max-requests or --budget-usd would be exceeded."""


_DROP_OR_FAIL = frozenset(c.value for c in GROUNDING_CODES)


def regen_codes(evals: Sequence[CandidateEval]) -> list[ReasonCode]:
    """Codes of the best failing candidate (parsed first, then fewest codes), enum order."""
    if not evals:
        return []
    best = min(evals, key=lambda e: (not e.parse_ok, len(e.codes)))
    have = {c for c in best.codes if c in _DROP_OR_FAIL}
    return [c for c in ReasonCode if c.value in have]


def _first_pass(
    item: dict[str, Any], texts: Sequence[str]
) -> tuple[int | None, WriterOutput | None, list[CandidateEval]]:
    transcript, group = str(item["transcript"]), str(item["group"])
    evals: list[CandidateEval] = []
    for i, raw in enumerate(texts):
        ev, cleaned = evaluate_candidate(raw, transcript, group)
        evals.append(ev)
        if cleaned is not None:
            return i, cleaned, evals
    return None, None, evals


class _Runner:
    def __init__(
        self,
        sampler: Sampler | None,
        cfg: RunConfig,
        usage: Usage,
        on_usage: Callable[[Usage], None] | None,
    ) -> None:
        self.sampler, self.cfg, self.usage, self.on_usage = sampler, cfg, usage, on_usage
        self.requests = 0
        self.stopped: str | None = None  # reason, once a cap was hit (no more launches)
        self.start_cost = usage.cost()
        self.rates, _ = price_for(cfg.model)

    def _est(self, n_prompt: int, n: int) -> float:
        r = self.rates
        return (
            n_prompt * r["prefill"]
            + n_prompt * (n - 1) * r["prefill_cached"]
            + n * self.cfg.max_tokens * r["sample"]
        ) / 1e6

    def fill(self, states: list[dict[str, Any]], target: int) -> None:
        """Sample until each state has ``target`` candidates (cached ones count)."""
        if self.stopped:
            return
        try:
            self._fill(states, target)
        except Budget as e:
            self.stopped = str(e)

    def _fill(self, states: list[dict[str, Any]], target: int) -> None:
        cfg = self.cfg
        pending = [st for st in states if len(st["cands"]) < target]
        step = max(1, cfg.concurrency)
        for start in range(0, len(pending), step):
            if self.sampler is None:
                raise Budget("no sampler (dry run or cache-only) but candidates are missing")
            futs: list[tuple[dict[str, Any], int, Callable[[], SampleResult]]] = []
            inflight = 0.0
            for st in pending[start : start + step]:
                if cfg.max_requests is not None and self.requests >= cfg.max_requests:
                    raise Budget("max requests reached")
                need = target - len(st["cands"])
                ids = self.sampler.prompt_ids(st["msgs"])
                est = self._est(len(ids), need)
                spent = self.usage.cost() - self.start_cost
                if cfg.budget_usd is not None and spent + inflight + est > cfg.budget_usd:
                    self._drain(futs)
                    raise Budget(f"budget cap {cfg.budget_usd:.4f} USD would be exceeded")
                futs.append(
                    (st, need, self.sampler.submit(ids, need, cfg.temperature, cfg.max_tokens))
                )
                inflight += est
                self.requests += 1
            self._drain(futs)

    def _drain(self, futs: list[tuple[dict[str, Any], int, Callable[[], SampleResult]]]) -> None:
        for st, need, fin in futs:
            t0 = time.monotonic()
            res = fin()
            self.usage.add_sample(
                res.prompt_tokens, res.cached_prompt_tokens, need, sum(res.out_tokens)
            )
            for text, n_out, stop in zip(res.texts, res.out_tokens, res.stop_reasons, strict=True):
                st["cands"].append({"text": text, "out_tokens": n_out, "stop_reason": stop})
            st["key"]["last_latency_s"] = round(time.monotonic() - t0, 3)
            save_cache(st["path"], {**st["key"], "prompt_tokens": res.prompt_tokens}, st["cands"])
            if self.on_usage is not None:
                self.on_usage(self.usage)
        futs.clear()


def _state(
    item: dict[str, Any], cfg: RunConfig, cache_dir: Path, rnd: int, regen: Sequence[ReasonCode]
) -> dict[str, Any]:
    iid = safe_id(str(item["item_id"]))
    msgs = teacher_messages(item, regen, cfg.addendum, cfg.glossary)
    key = _cache_key(cfg, iid, prompt_fp(msgs), rnd)
    path = _cache_path(cache_dir, iid, rnd, cfg)
    cands: list[dict[str, Any]] = []
    if load_cache(path, key):
        cands = json.loads(path.read_text(encoding="utf-8")).get("candidates", [])
    return {"item": item, "msgs": msgs, "key": key, "path": path, "cands": cands}


def run_items(
    items: list[dict[str, Any]],
    sampler: Sampler | None,
    cfg: RunConfig,
    cache_dir: Path = TEACHER_CACHE,
    usage: Usage | None = None,
    on_usage: Callable[[Usage], None] | None = None,
) -> dict[str, Selection]:
    """Round 1 (+ optional teacher-only regen round 2), then template. Resume-safe."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    runner = _Runner(sampler, cfg, usage or Usage(cfg.model), on_usage)
    ids = [safe_id(str(it["item_id"])) for it in items]
    r1 = {iid: _state(it, cfg, cache_dir, 1, ()) for iid, it in zip(ids, items, strict=True)}
    stages = (
        [cfg.first_stage, cfg.candidates] if cfg.first_stage < cfg.candidates else [cfg.candidates]
    )
    won: dict[str, tuple[int, int, WriterOutput]] = {}  # iid -> (round, index, gold)
    evals1: dict[str, list[CandidateEval]] = {}
    for target in stages:
        open_ = [r1[i] for i in ids if i not in won]
        runner.fill(open_, target)
        for iid in ids:
            if iid in won:
                continue
            idx, gold, ev = _first_pass(r1[iid]["item"], [c["text"] for c in r1[iid]["cands"]])
            evals1[iid] = ev
            if idx is not None and gold is not None:
                won[iid] = (1, idx, gold)
    evals2: dict[str, list[CandidateEval]] = {}
    codes2: dict[str, list[str]] = {}
    if cfg.regen_round:
        r2: dict[str, dict[str, Any]] = {}
        for iid, it in zip(ids, items, strict=True):
            if iid in won:
                continue
            codes = regen_codes(evals1.get(iid, []))
            codes2[iid] = [c.value for c in codes]
            r2[iid] = _state(it, cfg, cache_dir, 2, codes)
        runner.fill(list(r2.values()), cfg.regen_candidates)
        for iid, st in r2.items():
            idx, gold, ev = _first_pass(st["item"], [c["text"] for c in st["cands"]])
            evals2[iid] = ev
            if idx is not None and gold is not None:
                won[iid] = (2, idx, gold)
    results: dict[str, Selection] = {}
    for iid, it in zip(ids, items, strict=True):
        ev1, ev2 = evals1.get(iid, []), evals2.get(iid, [])
        unfinished = len(r1[iid]["cands"]) < cfg.candidates or (
            cfg.regen_round and iid not in won and len(ev2) < cfg.regen_candidates
        )
        if iid not in won and unfinished:
            results[iid] = Selection(PENDING, None, ev1, [], None, ev2, codes2.get(iid, []))
            continue
        if iid in won:
            rnd, idx, gold = won[iid]
            results[iid] = Selection(idx, gold, ev1, [], rnd, ev2, codes2.get(iid, []))
            continue
        out, gcodes = template_gold(it)
        w = UNUSABLE if out is None else TEMPLATE
        results[iid] = Selection(w, out, ev1, gcodes, None, ev2, codes2.get(iid, []))
    if runner.stopped:
        if cfg.raise_on_stop:
            raise Budget(runner.stopped)
        print(f"stopped early: {runner.stopped}")
    return results


# --------------------------------------------------------------------------- outputs


def sft_row(item: dict[str, Any], sel: Selection) -> dict[str, Any] | None:
    if sel.gold is None:
        return None
    msgs = as_dicts(writer_messages(item))
    return {
        "messages": msgs + [{"role": "assistant", "content": gold_json(sel.gold)}],
        "meta": {
            "item_id": item["item_id"],
            "seed_id": item.get("seed_id"),
            "source": sel.label,
            "language_mix": (item.get("meta") or {}).get("language_mix"),
        },
    }


def summarise(items: list[dict[str, Any]], results: dict[str, Selection]) -> dict[str, Any]:
    by_mix: dict[str, Counter[str]] = defaultdict(Counter)
    winners: Counter[str] = Counter()
    by_round: Counter[str] = Counter()
    fail_codes: dict[str, Counter[str]] = {"r1": Counter(), "r2": Counter()}
    cand: dict[str, Counter[str]] = {"r1": Counter(), "r2": Counter()}
    for item in items:
        sel = results.get(str(item["item_id"]))
        if sel is None:
            continue
        if sel.winner == PENDING:
            by_round[PENDING] += 1
            continue
        mix = str((item.get("meta") or {}).get("language_mix", "?"))
        winners[sel.label] += 1
        by_round[f"r{sel.round}" if sel.round else str(sel.winner)] += 1
        by_mix[mix]["items"] += 1
        by_mix[mix]["pass_r1"] += int(sel.round == 1)
        by_mix[mix]["pass_any"] += int(sel.round is not None)
        for rk, evs in (("r1", sel.evals), ("r2", sel.evals_r2)):
            for ev in evs:
                cand[rk]["n"] += 1
                cand[rk]["parsed"] += int(ev.parse_ok)
                cand[rk]["passed"] += int(ev.passed)
                cand[rk]["fenced"] += int(ev.fenced)
                if not ev.passed:
                    fail_codes[rk].update(ev.codes)
    total = sum(winners.values())

    def r(a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None

    return {
        "items": total,
        "winners": dict(sorted(winners.items())),
        "by_round": dict(sorted(by_round.items())),
        "item_pass_rate_r1": r(by_round["r1"], total),
        "item_pass_rate": r(by_round["r1"] + by_round["r2"], total),
        "r2_items_attempted": sum(
            1 for it in items if (s := results.get(str(it["item_id"]))) and s.evals_r2
        ),
        "template_share": r(winners[TEMPLATE], total),
        "candidates": {
            rk: {
                "n": c["n"],
                "parse_rate": r(c["parsed"], c["n"]),
                "pass_rate": r(c["passed"], c["n"]),
                "fenced": c["fenced"],
            }
            for rk, c in cand.items()
        },
        "pass_rate_by_language_mix": {
            m: {
                "items": c["items"],
                "r1": r(c["pass_r1"], c["items"]),
                "after_r2": r(c["pass_any"], c["items"]),
            }
            for m, c in sorted(by_mix.items())
        },
        "failing_candidate_codes": {rk: dict(c.most_common()) for rk, c in fail_codes.items()},
    }


def estimate(
    items: list[dict[str, Any]],
    cfg: RunConfig,
    est_out: int,
    prompt_len: Callable[[list[dict[str, str]]], int],
    first_stage_pass: float = 0.6,
) -> dict[str, Any]:
    """Dry-run token/cost estimate (no API). Assumes no cross-request prefix-cache hits."""
    p_tokens = [prompt_len(as_dicts(writer_messages(it))) for it in items]
    total_p = sum(p_tokens)
    rates, known = price_for(cfg.model)

    def cost(n_first: int, n_second: int, p_second_share: float) -> dict[str, float]:
        u = Usage(cfg.model)
        for pt in p_tokens:
            u.add_sample(pt, 0, n_first, est_out * n_first)
        if n_second:
            n_items_2 = round(len(p_tokens) * p_second_share)
            for pt in p_tokens[:n_items_2]:
                u.add_sample(pt, 0, n_second, est_out * n_second)
        return {
            "requests": u.requests,
            "prefill_tokens": u.prefill_tokens,
            "prefill_cached_tokens": u.prefill_cached_tokens,
            "sample_tokens": u.sample_tokens,
            "usd": round(u.cost(), 4),
        }

    out: dict[str, Any] = {
        "model": cfg.model,
        "items": len(items),
        "prompt_tokens_total": total_p,
        "prompt_tokens_mean": round(total_p / len(items), 1) if items else 0,
        "assumed_out_tokens_per_candidate": est_out,
        "rates_usd_per_mtok": rates,
        "rates_known": known,
        "single_shot_4": cost(4, 0, 0.0),
        "staged_2_then_2": cost(2, 2, 1.0 - first_stage_pass),
        "staged_assumed_first_stage_item_pass": first_stage_pass,
        "single_shot_2": cost(2, 0, 0.0),
    }
    return out


def load_split_items(split: str, items_path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if items_path is not None:
        return {"custom": read_jsonl(items_path)}
    names = SPLITS if split == "all" else (split,)
    out: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        p = DATA_DIR / f"{name}.jsonl"
        if not p.is_file():
            raise SystemExit(f"missing split file training/data/{name}.jsonl")
        out[name] = read_jsonl(p)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import os

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=[*SPLITS, "all"], default="train")
    ap.add_argument("--items", type=Path, default=None, help="explicit synthetic items JSONL")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--cache-dir", type=Path, default=TEACHER_CACHE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--model", default=os.environ.get("TINKER_TEACHER_MODEL", DEFAULT_TEACHER_MODEL)
    )
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer id (default: the model)")
    ap.add_argument("--renderer", choices=["auto", "qwen3_5", "hf"], default="auto")
    ap.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    ap.add_argument(
        "--first-stage",
        type=int,
        default=None,
        help="staged: sample k first, rest only if none pass",
    )
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--budget-usd", type=float, default=None, help="stop before exceeding")
    ap.add_argument("--regen-round", action="store_true", help="teacher-only round 2")
    ap.add_argument("--regen-candidates", type=int, default=2)
    ap.add_argument("--teacher-addendum", action="store_true", help="teacher-only system rules")
    ap.add_argument("--teacher-glossary", action="store_true", help="teacher-only lexicon glossary")
    ap.add_argument("--language-mix", default=None, help="only items with this meta.language_mix")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--est-out-tokens", type=int, default=DEFAULT_EST_OUT_TOKENS)
    ap.add_argument("--assume-items", type=int, default=None, help="dry run: scale to N items")
    ap.add_argument(
        "--first-stage-pass", type=float, default=0.6, help="dry run: staged assumption"
    )
    ap.add_argument("--cache-only", action="store_true", help="never call the API")
    a = ap.parse_args(argv)

    cfg = RunConfig(
        model=a.model,
        candidates=a.candidates,
        first_stage=a.first_stage or a.candidates,
        temperature=a.temperature,
        max_tokens=a.max_tokens,
        concurrency=a.concurrency,
        max_requests=a.max_requests,
        regen_round=a.regen_round,
        regen_candidates=a.regen_candidates,
        addendum=a.teacher_addendum,
        glossary=a.teacher_glossary,
        budget_usd=a.budget_usd,
        raise_on_stop=False,
    )
    splits = load_split_items(a.split, a.items)
    if a.language_mix:
        for k in list(splits):
            splits[k] = [
                it
                for it in splits[k]
                if (it.get("meta") or {}).get("language_mix") == a.language_mix
            ]
    if a.limit is not None:
        left = a.limit
        for k in list(splits):
            splits[k] = splits[k][: max(0, left)]
            left -= len(splits[k])
    all_items = [it for v in splits.values() for it in v]
    print(
        f"teacher model={cfg.model} items={len(all_items)} "
        f"splits={ {k: len(v) for k, v in splits.items()} }"
    )
    print(
        f"prompt_fingerprint(writer)={prompt_fingerprint('writer')[:16]} "
        f"lexicon_hash={lexicon_hash()[:16]}"
    )

    if a.dry_run:
        from training.tinker_common import load_tokenizer, make_renderer

        tok = load_tokenizer("Qwen/Qwen3.5-4B")  # same Qwen3.5 tokenizer family; local cache
        rend = make_renderer(tok)
        est = estimate(
            all_items,
            cfg,
            a.est_out_tokens,
            lambda m: len(rend.prompt_tokens(m)),
            first_stage_pass=a.first_stage_pass,
        )
        if a.assume_items and all_items:
            f = a.assume_items / len(all_items)
            est["scaled_to_items"] = a.assume_items
            for k in ("single_shot_4", "staged_2_then_2", "single_shot_2"):
                est[k] = {
                    kk: (round(v * f, 4) if kk == "usd" else round(v * f))
                    for kk, v in est[k].items()
                }
        print(json.dumps(est, indent=1))
        return 0

    cost_path = a.cache_dir / "_cost.json"
    prev: dict[str, Any] = {}
    if cost_path.is_file():
        prev = json.loads(cost_path.read_text(encoding="utf-8"))
    usage = (
        Usage.from_dict(prev["by_model"][cfg.model])
        if cfg.model in prev.get("by_model", {})
        else Usage(cfg.model)
    )

    def persist(u: Usage) -> None:
        by_model = dict(prev.get("by_model", {}))
        by_model[cfg.model] = u.to_dict()
        total = round(sum(float(v["estimated_cost_usd"]) for v in by_model.values()), 6)
        atomic_write_json(
            cost_path,
            {
                "by_model": by_model,
                "estimated_total_usd": total,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    sampler: Sampler | None = None
    if not a.cache_only:
        try:
            sampler = TinkerSampler(cfg.model, a.tokenizer, a.renderer)
        except Exception as e:  # noqa: BLE001 - report type/status only (no content)
            print(
                f"sampler init failed: {type(e).__name__} status={getattr(e, 'status_code', '?')}"
            )
            return 2
    t0 = time.monotonic()
    try:
        results = run_items(all_items, sampler, cfg, a.cache_dir, usage, persist)
    except Budget as e:
        print(f"stopped: {e}")
        persist(usage)
        return 3
    persist(usage)
    elapsed = time.monotonic() - t0

    report: dict[str, Any] = {"model": cfg.model, "config": cfg.__dict__, "splits": {}}
    for name, items in splits.items():
        rows = [r for it in items if (r := sft_row(it, results[str(it["item_id"])])) is not None]
        out_name = f"sft_{name}.jsonl"
        n = write_jsonl(a.out_dir / out_name, rows)
        summ = summarise(items, results)
        summ["sft_rows"] = n
        summ["per_item"] = {
            str(it["item_id"]): {
                "winner": results[str(it["item_id"])].label,
                "regen_codes": results[str(it["item_id"])].regen_codes,
                "candidate_codes_r2": [ev.codes for ev in results[str(it["item_id"])].evals_r2],
                "candidate_codes": [ev.codes for ev in results[str(it["item_id"])].evals],
                "gold_codes": results[str(it["item_id"])].gold_codes,
            }
            for it in items
        }
        report["splits"][name] = summ
        print(f"[{name}] wrote {out_name} rows={n}")
        print(json.dumps({k: v for k, v in summ.items() if k != "per_item"}, indent=1))
    report["usage"] = usage.to_dict()
    report["elapsed_s"] = round(elapsed, 2)
    atomic_write_json(a.cache_dir / "_report.json", report)
    print(f"usage: {json.dumps(usage.to_dict())}")
    print(f"elapsed_s={elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
