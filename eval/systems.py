"""Systems under evaluation (PRD §13.2) and the per-item record they produce.

* **A**: template only. Every item gets the *no-facts* template
  ``{Group} went for their evening walk.`` (PRD §8), facts ``[]``, mission_result
  ``not_mentioned``. This is the trivially faithful floor: zero unsupported details by
  construction, zero fact recall, and mission accuracy equal to the share of gold
  ``not_mentioned``. A cannot extract facts (no model), and a template built from GOLD
  facts would leak labels into a baseline, so it is not used.
* **B**: Gemma 4 E2B zero-shot, constrained by the JSON schema (``grammar=True``).
* **C**: Qwen3.5-4B base, few-shot (the few-shot examples live in the renderer supplied by
  the training code), GBNF grammar.
* **D**: Sair-Writer (Qwen3.5-4B + LoRA), GBNF grammar.
* **B+/C+/D+**: generate → parse → ``check_entry``; if it fails, ONE regeneration whose
  prompt carries the reason-code instructions (``build_writer_prompt(..., regen=codes)``);
  check again; if that fails too → ``template_output`` built from the verified facts of the
  last *parseable* attempt (no parseable attempt → the no-facts template).

Plain B/C/D have no fallback: their final entry is the single model output (whether or
not it passes the check). An unparseable output has no entry (``final_origin=invalid``).
Every record keeps the raw outputs and the check of each attempt. Records hold item
text and are written only to local eval output (never logged).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from eval.backends import CacheMiss, WriterBackend
from eval.items import EvalItem
from hub.grounding import CheckResult, build_template_entry, check_entry, template_output
from hub.prompts.builders import MissionText, Transcript, WalkerGroup, build_writer_prompt
from hub.schemas import (
    EntryOrigin,
    Fact,
    MissionResult,
    ReasonCode,
    WriterOutput,
    parse_writer_output,
)

__all__ = [
    "ALL_SYSTEMS",
    "BASE_OF",
    "Attempt",
    "CheckSummary",
    "ItemRecord",
    "is_plus",
    "run_item",
]

ALL_SYSTEMS: tuple[str, ...] = ("A", "B", "C", "D", "B+", "C+", "D+")
# writer backend key per system (the CLI maps keys to concrete backends)
BASE_OF: dict[str, str] = {
    "A": "template",
    "B": "gemma",
    "C": "qwen_base",
    "D": "sair_writer",
    "B+": "gemma",
    "C+": "qwen_base",
    "D+": "sair_writer",
}


def is_plus(system: str) -> bool:
    return system.endswith("+")


@dataclass(frozen=True)
class CheckSummary:
    passed: bool
    reasons: list[str]
    verified_facts: list[dict[str, Any]]
    dropped_facts: int
    mission_result: str
    mission_span: str | None
    unsupported_details: list[str]  # local eval output only

    @classmethod
    def of(cls, c: CheckResult) -> CheckSummary:
        return cls(
            passed=c.passed,
            reasons=[r.value for r in c.reasons],
            verified_facts=[f.model_dump(mode="json") for f in c.verified_facts],
            dropped_facts=c.dropped_facts,
            mission_result=c.mission_result.value,
            mission_span=c.mission_span,
            unsupported_details=list(c.unsupported_details),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CheckSummary:
        return cls(**d)

    def facts(self) -> list[Fact]:
        return [Fact.model_validate(f) for f in self.verified_facts]


@dataclass
class Attempt:
    raw: str
    parse_ok: bool
    parse_reasons: list[str]
    regen_codes: list[str]  # codes whose instructions were in this attempt's prompt
    check: CheckSummary | None
    latency_ms: float
    check_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    cached: bool = False


@dataclass
class ItemRecord:
    item_id: str
    seed: str
    system: str
    grammar: bool
    attempts: list[Attempt] = field(default_factory=list)
    final_origin: str = "invalid"  # model | template | invalid | error
    regen_used: bool = False
    final: dict[str, Any] | None = None  # WriterOutput as JSON
    final_check: CheckSummary | None = None
    latency: dict[str, float] = field(default_factory=dict)  # node -> ms, plus "e2e"
    error: str | None = None  # exception type name only

    @property
    def has_entry(self) -> bool:
        return self.final is not None and self.final_check is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ItemRecord:
        d = dict(d)
        attempts = []
        for a in d.pop("attempts", []):
            a = dict(a)
            chk = a.pop("check", None)
            attempts.append(Attempt(check=CheckSummary.from_dict(chk) if chk else None, **a))
        fc = d.pop("final_check", None)
        return cls(
            attempts=attempts,
            final_check=CheckSummary.from_dict(fc) if fc else None,
            **d,
        )


def _prompt(item: EvalItem, regen: list[ReasonCode]) -> list[Any]:
    return build_writer_prompt(
        WalkerGroup(item.group),
        MissionText(item.mission) if item.mission and item.mission.strip() else None,
        Transcript(item.transcript),
        regen=regen,
    )


def _attempt(
    writer: WriterBackend,
    item: EvalItem,
    *,
    grammar: bool,
    regen: list[ReasonCode],
    origin: EntryOrigin,
) -> tuple[Attempt, WriterOutput | None, CheckResult | None]:
    raw = writer.generate(_prompt(item, regen), grammar=grammar)
    stats = writer.last_call
    parsed = parse_writer_output(raw)
    t0 = time.perf_counter()
    check = (
        check_entry(parsed.value, item.transcript, item.group, origin=origin)
        if parsed.value
        else None
    )
    check_ms = (time.perf_counter() - t0) * 1000
    att = Attempt(
        raw=raw,
        parse_ok=parsed.ok,
        parse_reasons=[r.value for r in parsed.reasons],
        regen_codes=[r.value for r in regen],
        check=CheckSummary.of(check) if check else None,
        latency_ms=stats.latency_ms if stats else 0.0,
        check_ms=check_ms,
        tokens_in=stats.tokens_in if stats else None,
        tokens_out=stats.tokens_out if stats else None,
        cached=stats.cached if stats else False,
    )
    return att, parsed.value, check


def _no_facts_template(group: str) -> WriterOutput:
    return WriterOutput(
        facts=[],
        mission_result=MissionResult.NOT_MENTIONED,
        mission_span=None,
        entry=build_template_entry(group, [], False),
    )


def run_item(
    system: str, item: EvalItem, writer: WriterBackend, *, grammar: bool = True
) -> ItemRecord:
    """Run one system on one item. ``CacheMiss`` propagates; other backend errors are
    recorded by exception type only."""
    if system not in BASE_OF:
        raise ValueError("unknown system")
    rec = ItemRecord(item_id=item.id, seed=item.seed, system=system, grammar=grammar)
    try:
        _run(system, item, writer, grammar, rec)
    except CacheMiss:
        raise
    except Exception as exc:  # noqa: BLE001 - recorded by type only (NFR-P4)
        rec.final_origin = "error"
        rec.final = None
        rec.final_check = None
        rec.error = type(exc).__name__
    rec.latency["e2e"] = sum(v for k, v in rec.latency.items() if k != "e2e")
    return rec


def _run(
    system: str, item: EvalItem, writer: WriterBackend, grammar: bool, rec: ItemRecord
) -> None:
    if system == "A":
        att, out, chk = _attempt(
            writer, item, grammar=grammar, regen=[], origin=EntryOrigin.TEMPLATE
        )
        rec.attempts.append(att)
        rec.latency["template"] = att.latency_ms + att.check_ms
        if out is not None and chk is not None:
            rec.final_origin = "template"
            rec.final = out.model_dump(mode="json")
            rec.final_check = att.check
        return

    att1, out1, chk1 = _attempt(writer, item, grammar=grammar, regen=[], origin=EntryOrigin.MODEL)
    rec.attempts.append(att1)
    rec.latency["writer"] = att1.latency_ms
    rec.latency["check"] = att1.check_ms

    if not is_plus(system):
        if out1 is not None and chk1 is not None:
            rec.final_origin = "model"
            rec.final = out1.model_dump(mode="json")
            rec.final_check = att1.check
        return

    if out1 is not None and chk1 is not None and chk1.passed:
        rec.final_origin = "model"
        rec.final = out1.model_dump(mode="json")
        rec.final_check = att1.check
        return

    codes = list(chk1.reasons) if chk1 is not None else [ReasonCode(r) for r in att1.parse_reasons]
    att2, out2, chk2 = _attempt(
        writer, item, grammar=grammar, regen=codes, origin=EntryOrigin.MODEL
    )
    rec.attempts.append(att2)
    rec.regen_used = True
    rec.latency["writer_regen"] = att2.latency_ms
    rec.latency["check_regen"] = att2.check_ms
    if out2 is not None and chk2 is not None and chk2.passed:
        rec.final_origin = "model"
        rec.final = out2.model_dump(mode="json")
        rec.final_check = att2.check
        return

    t0 = time.perf_counter()
    last = chk2 if chk2 is not None else chk1
    tmpl = template_output(item.group, last) if last is not None else _no_facts_template(item.group)
    tchk = check_entry(tmpl, item.transcript, item.group, origin=EntryOrigin.TEMPLATE)
    rec.latency["template"] = (time.perf_counter() - t0) * 1000
    rec.final_origin = "template"
    rec.final = tmpl.model_dump(mode="json")
    rec.final_check = CheckSummary.of(tchk)
