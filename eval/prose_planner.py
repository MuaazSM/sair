"""EVAL ONLY: the "prose memory" planner prompt for the contamination experiment (PRD §13.4).

This builder deliberately violates FR-M3: the planner sees the TEXT of previous diary
entries instead of verified fact records. It exists only to measure how invented entry
details leak into later missions, and it lives in ``eval/`` so production code can never
import it (hub/ never depends on eval/). Production always uses
``hub.prompts.builders.build_planner_prompt`` (verified facts only).

To keep the two arms comparable, the system prompt is the production planner system
prompt verbatim and the user block has the same layout; only the memory section differs
("Diary entries ..." with one dated entry per line instead of fact records). The prose
planner is never given fact ids, so with the provenance filter ON a prose-arm mission can
only pass FR-M4 with generic walk words.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from hub.prompts.builders import (
    MAX_RECENT_MISSIONS,
    REGEN_HEADER,
    ChatMessage,
    MissionText,
    WalkerGroup,
    planner_system_prompt,
)
from hub.schemas import ReasonCode, reasons_to_instructions

__all__ = ["MAX_PROSE_ENTRIES", "build_prose_planner_prompt"]

MAX_PROSE_ENTRIES = 7  # same window as fact memory (last 7 walks, FR-M2)
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _day(d: dt.date) -> str:
    return f"{_WEEKDAYS[d.weekday()]} {d.isoformat()}"


def build_prose_planner_prompt(
    entries: Sequence[tuple[dt.date, str]],
    recent_missions: Sequence[MissionText],
    *,
    group: WalkerGroup,
    today: dt.date,
    regen: Sequence[ReasonCode] = (),
) -> list[ChatMessage]:
    """Planner prompt whose memory is previous entries' prose (eval-only, violates FR-M3)."""
    if not isinstance(group, WalkerGroup):
        raise TypeError("build_prose_planner_prompt: group must be WalkerGroup")
    if isinstance(today, dt.datetime) or not isinstance(today, dt.date):
        raise TypeError("build_prose_planner_prompt: today must be a date")
    recent = list(recent_missions)
    if len(recent) > MAX_RECENT_MISSIONS:
        raise ValueError("build_prose_planner_prompt: at most 5 recent missions")
    if any(not isinstance(m, MissionText) for m in recent):
        raise TypeError("build_prose_planner_prompt: recent_missions must be MissionText")
    window = list(entries)[-MAX_PROSE_ENTRIES:]
    entry_lines = [f"- {_day(d)}: {' '.join(text.split())}" for d, text in window]
    recent_lines = [f"- {m.text}" for m in recent]
    user = (
        f"Walkers: {group.name}\n"
        f"Today: {_day(today)}\n"
        "Diary entries from recent walks (one per line, oldest first):\n"
        f"{chr(10).join(entry_lines) if entry_lines else '(none)'}\n"
        "Recent missions (do not repeat):\n"
        f"{chr(10).join(recent_lines) if recent_lines else '(none)'}"
    )
    lines = reasons_to_instructions(list(regen))
    if lines:
        user += "\n\n" + REGEN_HEADER + "\n" + "\n".join(f"- {line}" for line in lines)
    return [ChatMessage("system", planner_system_prompt()), ChatMessage("user", user)]
