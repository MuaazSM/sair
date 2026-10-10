"""Eval item schema and loaders (synthetic test set, real set, human labels).

Synthetic items (one JSON object per line, ``eval/data/synthetic/test.jsonl`` by default;
written later by the synthetic-data tooling from the held-out seeds)::

    {"id": "s012-07", "seed": "s012", "group": "Mom and Dad",
     "mission": "Listen for one bird you can hear from the path." | null,
     "transcript": "...",
     "gold": {"facts": [{"span": "...", "subject": "...", "kind": "sound",
                         "certainty": "unsure", "place": null}],
              "mission_result": "done", "mission_span": "..." | null},
     "reference_transcript": null,          # real set only (WER/CER)
     "human": null | {"has_unsupported": true, "unsupported": ["parrot"]}}

``seed`` is the cluster id for the cluster bootstrap (PRD §13.3). The synthetic-data
generator (``training/split.py``) writes ``item_id`` / ``seed_id`` / ``mission_text`` and a
``meta`` block; those are accepted as aliases of ``id`` / ``seed`` / ``mission`` (a file
that gives both spellings with different values is rejected), and ``meta`` is kept as-is.
The real set uses the same schema; its file lives under the real-set root
(``eval/real/``), which this code only ever opens when the owner runs ``eval/run.py --set real``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hub.schemas import Certainty, FactKind, MissionResult

__all__ = [
    "FIELD_ALIASES",
    "EvalItem",
    "Gold",
    "GoldFact",
    "HumanLabels",
    "ItemsError",
    "load_items",
]

_CFG = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ItemsError(ValueError):
    """Fixed-message error for unreadable item files (no content in the message)."""


class GoldFact(BaseModel):
    model_config = _CFG

    span: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    kind: FactKind
    certainty: Certainty
    place: str | None = None


class Gold(BaseModel):
    model_config = _CFG

    facts: list[GoldFact] = Field(default_factory=list)
    mission_result: MissionResult = MissionResult.NOT_MENTIONED
    mission_span: str | None = None


class HumanLabels(BaseModel):
    """Human unsupported-detail annotation for one entry of one system (owner-written)."""

    model_config = _CFG

    has_unsupported: bool | None = None
    unsupported: list[str] | None = None


# generator spelling -> eval spelling (training/split.py writes the left-hand keys)
FIELD_ALIASES: dict[str, str] = {"item_id": "id", "seed_id": "seed", "mission_text": "mission"}


class EvalItem(BaseModel):
    model_config = _CFG

    @model_validator(mode="before")
    @classmethod
    def _accept_generator_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for alias, name in FIELD_ALIASES.items():
            if alias not in out:
                continue
            value = out.pop(alias)
            if name in out and out[name] != value:
                raise ValueError("conflicting item fields")
            out[name] = value
        return out

    id: str = Field(min_length=1)
    seed: str = Field(min_length=1)
    group: str = Field(min_length=1)
    mission: str | None = None
    transcript: str = Field(min_length=1)
    gold: Gold = Field(default_factory=Gold)
    reference_transcript: str | None = None
    human: dict[str, HumanLabels] | None = None  # keyed by system name
    meta: dict[str, Any] | None = None  # generator metadata (language mix, style, ...)


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line


def load_items(path: Path) -> list[EvalItem]:
    """Load a JSONL item file. Errors carry the line number only, never content."""
    if not path.is_file():
        raise ItemsError("item file not found")
    items: list[EvalItem] = []
    seen: set[str] = set()
    for n, line in enumerate(_iter_lines(path), start=1):
        try:
            item = EvalItem.model_validate(json.loads(line))
        except (ValueError, TypeError):
            raise ItemsError(f"invalid item on line {n}") from None
        if item.id in seen:
            raise ItemsError(f"duplicate item id on line {n}")
        seen.add(item.id)
        items.append(item)
    return items
