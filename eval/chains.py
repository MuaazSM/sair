"""Contamination experiment (PRD §13.4): week-long chains, prose vs fact memory.

Chain file format (``eval/data/chains/*.yaml``, one chain per file; synthetic only,
generic places, never derived from real notes)::

    version: 1
    id: c01                       # unique chain id (also the bootstrap unit)
    group: "Mom and Dad"
    place_type: "a lane"          # generic place, informational
    start_date: 2026-10-05        # date of day 1
    first_mission: "Listen for one bird you can hear from the path."   # day-1 mission
    days:                         # 7 entries for real chains (examples may be shorter)
      - transcript: "..."         # what the walkers said on day d (required)
        note: "..."               # optional, informational

Loop, for each day d = 1..N:

1. writer (system A-D or a "+" variant) on (group, mission_d, transcript_d) → entry + check;
2. memory: ``fact`` = the verified facts of the last 7 walks (``VerifiedFact`` records via
   the production ``build_planner_prompt``), ``prose`` = the previous entries' TEXT via
   the eval-only ``build_prose_planner_prompt`` (violates FR-M3 by design);
3. planner (fixed, Gemma) → mission for day d+1 → ``check_mission`` with the FR-M4
   provenance filter OFF (headline arm) or ON (production arm); a failure gets ONE
   regeneration with reason codes, then a fixed fallback from ``data/fallback_missions.yaml``;
4. the served mission becomes mission_{d+1}.

Contamination (mission planned after day p): the mission's concrete details
(``extract_details`` strict, entry-side rules) minus generic walk words (``walk_words.yaml``
generic nouns, non-number lexicon ids) and weekday names, that are absent from ALL
transcripts of days 1..p (``extract_details`` inclusive, alias-aware; person-name details
by boundaried presence in the normalised transcripts). ``echo_only`` marks a contaminated
mission whose orphan details all appear in a mission the planner was shown (so they did not
come from memory).

Which mission is measured (PRD §13.4). Each day is scored twice:

* ``*_raw``: the planner's RAW output, i.e. its first parsed mission for that day, before
  any safety regeneration or fallback (``None`` when no attempt parsed);
* ``*_served``: the mission actually served (after regen/fallback), which also feeds the
  next day's writer and the planner's "recent missions".

The HEADLINE fields (``orphans``/``contaminated``/``echo_only`` per day,
``first_contaminated_day`` per chain, and ``contaminated_by_day`` /
``first_contaminated_day`` / ``pct_contaminated_by_last_day`` in the summary) use the RAW
output for provenance-OFF runs ("headline runs use the raw planner output with the FR-M4
provenance filter OFF in both arms") and the SERVED mission for provenance-ON runs ("what
production does"). The other measure is always kept under ``raw_*`` / ``served_*`` names;
each summary row says which one its headline uses in ``headline_measure``.

Resumable: every model call is cached under ``eval/cache/chains/`` keyed by
(backend id, prompt sha) and every finished (chain, system, memory, provenance) run is
checkpointed as ``<out>/runs/<key>.json``; ``--resume`` skips finished runs and replays
cached calls, so a killed run restarts cheaply.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from pydantic import BaseModel, ConfigDict, Field

from eval.backends import CacheMiss, CacheMode, PlannerBackend, WriterBackend
from eval.bootstrap import plain_bootstrap
from eval.items import EvalItem
from eval.metrics import Ratio
from eval.prose_planner import build_prose_planner_prompt
from eval.systems import BASE_OF, run_item
from hub.grounding import extract_details, load_lexicon, normalise
from hub.prompts.builders import (
    MAX_RECENT_MISSIONS,
    ChatMessage,
    MissionText,
    WalkerGroup,
    build_planner_prompt,
)
from hub.safety import check_mission
from hub.schemas import Fact, MissionPlan, ReasonCode, VerifiedFact, parse_mission_plan

__all__ = [
    "ChainDay",
    "ChainResult",
    "ChainSpec",
    "DayRecord",
    "contamination",
    "load_chain",
    "load_chains",
    "main",
    "run_chain",
    "run_chains",
    "stub_plan",
    "summarise",
]

ROOT = Path(__file__).resolve().parent.parent
CHAINS_DIR = ROOT / "eval" / "data" / "chains"
CACHE_DIR = ROOT / "eval" / "cache" / "chains"
OUT_DIR = ROOT / "eval" / "results" / "chains"
MEMORY_WINDOW = 7  # last 7 walks (FR-M2)

_CFG = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ChainDay(BaseModel):
    model_config = _CFG
    transcript: str = Field(min_length=1)
    note: str | None = None


class ChainSpec(BaseModel):
    model_config = _CFG
    version: int = 1
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    group: str = Field(min_length=1)
    place_type: str | None = None
    start_date: dt.date
    first_mission: str = Field(min_length=1)
    days: list[ChainDay] = Field(min_length=1, max_length=14)


def load_chain(path: Path) -> ChainSpec:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ChainSpec.model_validate(data)
    except (OSError, ValueError, yaml.YAMLError):
        raise ValueError("invalid chain file") from None


def load_chains(directory: Path) -> list[ChainSpec]:
    specs = [load_chain(p) for p in sorted(directory.glob("*.yaml"))]
    ids = [s.id for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chain ids")
    return specs


# --------------------------------------------------------------------------- contamination


@lru_cache(maxsize=1)
def _generic_keys() -> frozenset[str]:
    ww = yaml.safe_load((ROOT / "data" / "walk_words.yaml").read_text(encoding="utf-8"))
    nouns = [str(n) for n in ww.get("generic_nouns", [])]
    lex = load_lexicon()
    keys = extract_details(" , ".join(nouns), inclusive=True)
    return frozenset(k for k in keys if lex.categories.get(k) not in (None, "number"))


def _name_present(name: str, texts: Sequence[str]) -> bool:
    pat = re.compile(rf"(?<![0-9a-zà-ɏ]){re.escape(name)}(?:'s)?(?![0-9a-zà-ɏ])")
    return any(pat.search(normalise(t)) for t in texts)


def contamination(
    mission: str,
    transcripts: Sequence[str],
    group: str,
    *,
    shown_missions: Sequence[str] = (),
) -> tuple[list[str], bool]:
    """(orphan detail keys, echo_only) for one mission. Keys are canonical ids, not text."""
    details = extract_details(mission, group=group)
    generic = _generic_keys()
    evidence: set[str] = set()
    for t in transcripts:
        evidence |= extract_details(t, inclusive=True)
    orphans: list[str] = []
    for key in sorted(details):
        if key in generic or key.startswith("day:"):
            continue
        if key.startswith("name:"):
            if not _name_present(key.removeprefix("name:"), transcripts):
                orphans.append(key)
        elif key not in evidence:
            orphans.append(key)
    if not orphans or not shown_missions:
        return orphans, False
    shown: set[str] = set()
    for m in shown_missions:
        shown |= extract_details(m, inclusive=True)
    echo = all(
        (
            _name_present(o.removeprefix("name:"), shown_missions)
            if o.startswith("name:")
            else o in shown
        )
        for o in orphans
    )
    return orphans, echo


# --------------------------------------------------------------------------- records


@dataclass
class PlannerAttempt:
    raw: str
    parse_ok: bool
    reasons: list[str]  # parse or safety codes
    passed: bool


@dataclass
class DayRecord:
    day: int
    mission_in: str
    writer_origin: str
    writer_passed: bool | None
    writer_unsupported: int
    entry: str | None
    verified_facts: int
    planner_attempts: list[PlannerAttempt] = field(default_factory=list)
    mission_out: str = ""
    mission_out_source: str = "fallback"  # model | fallback
    # headline measure: raw planner output when provenance is off, served mission when on
    orphans: list[str] = field(default_factory=list)
    contaminated: bool = False
    echo_only: bool = False
    # raw planner output (first parsed mission of the day; None = nothing parsed)
    raw_orphans: list[str] | None = None
    contaminated_raw: bool | None = None
    raw_echo_only: bool = False
    # served mission (after safety regen / fallback)
    served_orphans: list[str] = field(default_factory=list)
    contaminated_served: bool = False
    served_echo_only: bool = False


@dataclass
class ChainResult:
    chain_id: str
    system: str
    memory: str
    provenance: bool
    days: list[DayRecord] = field(default_factory=list)
    error: str | None = None

    @property
    def headline_measure(self) -> str:
        """PRD §13.4: raw planner output with the filter off; served mission with it on."""
        return "served_mission" if self.provenance else "raw_planner_output"

    @property
    def first_contaminated_day(self) -> int | None:
        """Headline: first day whose headline-measured mission is contaminated."""
        return next((d.day for d in self.days if d.contaminated), None)

    @property
    def first_contaminated_day_raw(self) -> int | None:
        return next((d.day for d in self.days if d.contaminated_raw), None)

    @property
    def first_contaminated_day_served(self) -> int | None:
        return next((d.day for d in self.days if d.contaminated_served), None)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["headline_measure"] = self.headline_measure
        d["first_contaminated_day"] = self.first_contaminated_day
        d["first_contaminated_day_raw"] = self.first_contaminated_day_raw
        d["first_contaminated_day_served"] = self.first_contaminated_day_served
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChainResult:
        days = []
        for day in d.get("days", []):
            day = dict(day)
            atts = [PlannerAttempt(**a) for a in day.pop("planner_attempts", [])]
            days.append(DayRecord(planner_attempts=atts, **day))
        return cls(
            chain_id=d["chain_id"],
            system=d["system"],
            memory=d["memory"],
            provenance=d["provenance"],
            days=days,
            error=d.get("error"),
        )


@lru_cache(maxsize=1)
def _fallbacks() -> tuple[str, ...]:
    data = yaml.safe_load((ROOT / "data" / "fallback_missions.yaml").read_text(encoding="utf-8"))
    return tuple(str(m["text"]) for m in data["missions"])


def _fallback(day: int, recent: Sequence[str]) -> str:
    fb = _fallbacks()
    for k in range(len(fb)):
        cand = fb[(day - 1 + k) % len(fb)]
        if cand not in recent:
            return cand
    return fb[(day - 1) % len(fb)]


def _verified(
    spec: ChainSpec, day: int, date: dt.date, facts: Sequence[Fact]
) -> list[VerifiedFact]:
    return [
        VerifiedFact(
            id=f"f{day}_{i}",
            walk_id=f"w{day}",
            walk_date=date,
            group=spec.group,
            kind=f.kind,
            subject=f.subject,
            place=f.place,
            certainty=f.certainty,
            span=f.span,
        )
        for i, f in enumerate(facts, start=1)
    ]


def run_chain(
    spec: ChainSpec,
    system: str,
    writer: WriterBackend,
    planner: PlannerBackend,
    *,
    memory: str,
    provenance: bool,
) -> ChainResult:
    """Run one chain end to end. ``CacheMiss`` propagates."""
    if memory not in ("prose", "fact"):
        raise ValueError("memory must be prose or fact")
    res = ChainResult(spec.id, system, memory, provenance)
    group = WalkerGroup(spec.group)
    mission = spec.first_mission
    served: list[str] = [mission]
    transcripts: list[str] = []
    entries: list[tuple[dt.date, str]] = []
    facts_by_walk: list[list[VerifiedFact]] = []
    for d, day in enumerate(spec.days, start=1):
        date = spec.start_date + dt.timedelta(days=d - 1)
        item = EvalItem(
            id=f"{spec.id}-d{d}",
            seed=spec.id,
            group=spec.group,
            mission=mission,
            transcript=day.transcript,
        )
        rec = run_item(system, item, writer)
        transcripts.append(day.transcript)
        entry = str(rec.final["entry"]) if rec.final is not None else None
        chk = rec.final_check
        facts = _verified(spec, d, date, chk.facts()) if chk is not None else []
        facts_by_walk.append(facts)
        if entry is not None:
            entries.append((date, entry))
        dr = DayRecord(
            day=d,
            mission_in=mission,
            writer_origin=rec.final_origin,
            writer_passed=chk.passed if chk is not None else None,
            writer_unsupported=len(chk.unsupported_details) if chk is not None else 0,
            entry=entry,
            verified_facts=len(facts),
        )

        memory_facts = [f for walk in facts_by_walk[-MEMORY_WINDOW:] for f in walk]
        facts_by_id = {f.id: f for f in memory_facts}
        recent = served[-MAX_RECENT_MISSIONS:]
        recent_mt = [MissionText(m) for m in recent]
        today = date + dt.timedelta(days=1)

        def prompt(
            regen: Sequence[ReasonCode],
            facts: list[VerifiedFact] = memory_facts,
            recent_m: list[MissionText] = recent_mt,
            day_date: dt.date = today,
        ) -> list[ChatMessage]:
            if memory == "fact":
                return build_planner_prompt(
                    facts, recent_m, group=group, today=day_date, regen=regen
                )
            return build_prose_planner_prompt(
                entries, recent_m, group=group, today=day_date, regen=regen
            )

        chosen: str | None = None
        first_text: str | None = None
        regen: list[ReasonCode] = []
        for _ in range(2):
            raw = planner.plan(prompt(regen))
            parsed = parse_mission_plan(raw)
            plan: MissionPlan | None = parsed.value
            if plan is None:
                dr.planner_attempts.append(
                    PlannerAttempt(raw, False, [r.value for r in parsed.reasons], False)
                )
                regen = list(parsed.reasons)
                continue
            if first_text is None:
                first_text = plan.text
            safety = check_mission(plan, facts_by_id, recent_missions=recent, provenance=provenance)
            dr.planner_attempts.append(
                PlannerAttempt(raw, True, [r.value for r in safety.reasons], safety.passed)
            )
            if safety.passed:
                chosen = plan.text
                break
            regen = list(safety.reasons)
        if chosen is not None:
            dr.mission_out, dr.mission_out_source = chosen, "model"
        else:
            dr.mission_out, dr.mission_out_source = _fallback(d, recent), "fallback"

        s_orphans, s_echo = contamination(
            dr.mission_out, transcripts, spec.group, shown_missions=recent
        )
        dr.served_orphans, dr.contaminated_served, dr.served_echo_only = (
            s_orphans,
            bool(s_orphans),
            s_echo,
        )
        if first_text is not None:
            r_orphans, r_echo = contamination(
                first_text, transcripts, spec.group, shown_missions=recent
            )
            dr.raw_orphans, dr.contaminated_raw, dr.raw_echo_only = (
                r_orphans,
                bool(r_orphans),
                r_echo,
            )
        # PRD §13.4: headline = raw planner output (filter off) / served mission (filter on)
        if provenance:
            dr.orphans, dr.contaminated, dr.echo_only = list(s_orphans), bool(s_orphans), s_echo
        else:
            dr.orphans = list(dr.raw_orphans or [])
            dr.contaminated = bool(dr.contaminated_raw)
            dr.echo_only = dr.raw_echo_only
        res.days.append(dr)
        mission = dr.mission_out
        served.append(mission)
    return res


# --------------------------------------------------------------------------- batch + summary


def _run_key(chain_id: str, system: str, memory: str, provenance: bool) -> str:
    sys_tag = system.replace("+", "plus")
    return f"{chain_id}__{sys_tag}__{memory}__prov-{'on' if provenance else 'off'}"


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def run_chains(
    specs: Sequence[ChainSpec],
    systems: Sequence[str],
    writers: dict[str, WriterBackend],
    planner: PlannerBackend,
    *,
    memories: Sequence[str] = ("prose", "fact"),
    provenances: Sequence[bool] = (False, True),
    out_dir: Path | None = None,
    resume: bool = False,
    progress: bool = False,
) -> list[ChainResult]:
    results: list[ChainResult] = []
    total = len(specs) * len(systems) * len(memories) * len(provenances)
    done = 0
    for spec in specs:
        for system in systems:
            for memory in memories:
                for prov in provenances:
                    done += 1
                    key = _run_key(spec.id, system, memory, prov)
                    path = out_dir / "runs" / f"{key}.json" if out_dir is not None else None
                    if resume and path is not None and path.is_file():
                        results.append(ChainResult.from_dict(json.loads(path.read_text("utf-8"))))
                        continue
                    res = run_chain(
                        spec,
                        system,
                        writers[BASE_OF[system]],
                        planner,
                        memory=memory,
                        provenance=prov,
                    )
                    if path is not None:
                        _atomic_write(path, res.to_dict())
                    results.append(res)
                    if progress:
                        print(f"[{done}/{total}] {key}", file=sys.stderr, flush=True)
    return results


_MEASURES: dict[str, tuple[str, str]] = {
    # measure -> (DayRecord flag, ChainResult first-day property)
    "headline": ("contaminated", "first_contaminated_day"),
    "raw": ("contaminated_raw", "first_contaminated_day_raw"),
    "served": ("contaminated_served", "first_contaminated_day_served"),
}


def _measure(
    rs: Sequence[ChainResult], n_days: int, measure: str, n_resamples: int
) -> dict[str, Any]:
    flag, first_attr = _MEASURES[measure]
    firsts: list[int | None] = [getattr(r, first_attr) for r in rs]
    by_day: dict[int, float] = {}
    per_day: dict[int, float] = {}
    for p in range(1, n_days + 1):
        by_day[p] = sum(1 for f in firsts if f is not None and f <= p) / len(rs)
        on_day = [d for r in rs for d in r.days if d.day == p]
        per_day[p] = sum(1 for d in on_day if getattr(d, flag)) / len(on_day) if on_day else 0.0
    final = [Ratio(1.0 if f is not None else 0.0, 1) for f in firsts]
    ci = plain_bootstrap(final, n_resamples=n_resamples)
    return {
        "pct_contaminated_by_last_day": ci.to_dict(),
        "contaminated_by_day": by_day,
        "mission_contamination_rate_by_day": per_day,
        "first_contaminated_day": firsts,
        "contaminated_missions": sum(1 for r in rs for d in r.days if getattr(d, flag)),
    }


def summarise(results: Sequence[ChainResult], *, n_resamples: int = 10_000) -> list[dict[str, Any]]:
    """Per (system, memory, provenance): % chains contaminated by each planning day, day of
    first contamination, per-day mission contamination rate; plain bootstrap over chains.

    PRD §13.4: the unprefixed (headline) fields measure the RAW planner output when the
    provenance filter is off and the SERVED mission when it is on (``headline_measure``).
    ``raw_*`` and ``served_*`` carry both measures for every row."""
    groups: dict[tuple[str, str, bool], list[ChainResult]] = {}
    for r in results:
        groups.setdefault((r.system, r.memory, r.provenance), []).append(r)
    rows: list[dict[str, Any]] = []
    for (system, memory, prov), rs in sorted(groups.items()):
        n_days = max((len(r.days) for r in rs), default=0)
        head = _measure(rs, n_days, "headline", n_resamples)
        raw = _measure(rs, n_days, "raw", n_resamples)
        served = _measure(rs, n_days, "served", n_resamples)
        row: dict[str, Any] = {
            "system": system,
            "memory": memory,
            "provenance": "on" if prov else "off",
            "headline": not prov,
            "headline_measure": "served_mission" if prov else "raw_planner_output",
            "n_chains": len(rs),
            "n_days": n_days,
            **{k: v for k, v in head.items() if k != "contaminated_missions"},
        }
        for prefix, m in (("raw", raw), ("served", served)):
            for k, v in m.items():
                row[f"{prefix}_{k}"] = v
        row["raw_unparsed_missions"] = sum(
            1 for r in rs for d in r.days if d.contaminated_raw is None
        )
        row["echo_only_missions"] = sum(1 for r in rs for d in r.days if d.echo_only)
        row["raw_echo_only_missions"] = sum(1 for r in rs for d in r.days if d.raw_echo_only)
        row["served_echo_only_missions"] = sum(1 for r in rs for d in r.days if d.served_echo_only)
        row["fallback_missions"] = sum(
            1 for r in rs for d in r.days if d.mission_out_source == "fallback"
        )
        row["missions"] = sum(len(r.days) for r in rs)
        rows.append(row)
    return rows


def _write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    _atomic_write(out_dir / "contamination.json", {"rows": rows})
    lines = [
        "system,memory,provenance,headline_measure,n_chains,day,pct_chains_contaminated,"
        "mission_rate,raw_pct_chains_contaminated,served_pct_chains_contaminated"
    ]
    for row in rows:
        for p, v in row["contaminated_by_day"].items():
            rate = row["mission_contamination_rate_by_day"][p]
            raw = row["raw_contaminated_by_day"][p]
            served = row["served_contaminated_by_day"][p]
            lines.append(
                f"{row['system']},{row['memory']},{row['provenance']},"
                f"{row['headline_measure']},{row['n_chains']},{p},{v:.6f},{rate:.6f},"
                f"{raw:.6f},{served:.6f}"
            )
    (out_dir / "contamination.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


# Deterministic stub planner for dry runs (``--stub``): no model, generic sense missions only.
STUB_MISSIONS: tuple[str, ...] = (
    "Listen for the softest sound you can hear.",
    "Look at the colour of the sky above you.",
    "Notice the wind as it moves the leaves around you.",
    "Find one leaf with a shape you like.",
    "Count the trees you pass along your way.",
    "Spot one thing on your street that you have never noticed before.",
    "Watch the clouds move across the sky above you.",
)


def stub_plan(messages: list[ChatMessage]) -> str:
    """First stub mission not among the prompt's recent missions (deterministic)."""
    user = messages[-1].content if messages else ""
    recent = user.split("Recent missions (do not repeat):", 1)[-1]
    text = next((m for m in STUB_MISSIONS if m not in recent), STUB_MISSIONS[0])
    return json.dumps({"type": "sense", "references": [], "text": text})


def main(
    argv: Sequence[str] | None = None,
    *,
    writers: dict[str, WriterBackend] | None = None,
    planner: PlannerBackend | None = None,
) -> int:
    ap = argparse.ArgumentParser(
        prog="eval/chains.py", description="PRD §13.4 contamination chains"
    )
    ap.add_argument("--chains-dir", type=Path, default=CHAINS_DIR)
    ap.add_argument("--systems", default="A,B,C,D")
    ap.add_argument("--memory", default="prose,fact")
    ap.add_argument("--provenance", default="off,on", help="off = headline, on = production")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--resume", action="store_true", help="skip finished runs, replay cached calls")
    ap.add_argument("--cached", action="store_true", help="replay cache only; fail on a miss")
    ap.add_argument(
        "--renderer", default=None, help="module:attr renderer, fallback for both C and D"
    )
    ap.add_argument("--renderer-c", default=None, help="module:attr renderer for system C")
    ap.add_argument("--renderer-d", default=None, help="module:attr renderer for system D")
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument(
        "--stub",
        action="store_true",
        help="dry run, no models: system A template writer + deterministic stub planner",
    )
    args = ap.parse_args(argv)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if any(s not in BASE_OF for s in systems):
        print("unknown system", file=sys.stderr)
        return 2
    memories = [m.strip() for m in args.memory.split(",") if m.strip()]
    provs = [p.strip() == "on" for p in args.provenance.split(",") if p.strip()]
    specs = load_chains(args.chains_dir)
    if not specs:
        print("no chain files found", file=sys.stderr)
        return 2
    if args.stub:
        if any(s != "A" for s in systems):
            print("--stub supports system A only", file=sys.stderr)
            return 2
        from eval.backends import StubPlanner, TemplateWriter

        writers = writers or {"template": TemplateWriter()}
        planner = planner or StubPlanner(stub_plan, backend_id="stub-planner:dry")
    mode = CacheMode.CACHED if args.cached else CacheMode.RESUME
    if writers is None or planner is None:
        from eval.config import build_planner, build_writers

        writers = writers or build_writers(
            [BASE_OF[s] for s in systems],
            cache_dir=args.cache_dir,
            mode=mode,
            renderer_spec=args.renderer,
            renderer_c=args.renderer_c,
            renderer_d=args.renderer_d,
        )
        planner = planner or build_planner(cache_dir=args.cache_dir, mode=mode)
    try:
        results = run_chains(
            specs,
            systems,
            writers,
            planner,
            memories=memories,
            provenances=provs,
            out_dir=args.out,
            resume=args.resume or args.cached,
            progress=True,
        )
    except CacheMiss as miss:
        print(f"{miss}; run without --cached to fill the cache", file=sys.stderr)
        return 4
    rows = summarise(results, n_resamples=args.resamples)
    _write_summary(rows, args.out)
    for row in rows:
        ci = row["pct_contaminated_by_last_day"]
        print(
            f"{row['system']:<3} memory={row['memory']:<5} provenance={row['provenance']:<3} "
            f"measure={row['headline_measure']} n_chains={row['n_chains']} "
            f"contaminated_by_day{row['n_days']}={ci['point']:.3f} "
            f"[{ci['lo']:.3f}, {ci['hi']:.3f}] "
            f"(raw={row['raw_pct_contaminated_by_last_day']['point']:.3f} "
            f"served={row['served_pct_contaminated_by_last_day']['point']:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
