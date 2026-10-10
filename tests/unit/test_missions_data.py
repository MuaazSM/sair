"""Fixed fallback and seed missions (PRD 11.2, FR-M6, FR-M6b)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORDS: dict[str, Any] = yaml.safe_load((ROOT / "data" / "walk_words.yaml").read_text("utf-8"))
FALLBACKS: list[dict[str, Any]] = yaml.safe_load(
    (ROOT / "data" / "fallback_missions.yaml").read_text("utf-8")
)["missions"]
SEEDS: list[dict[str, Any]] = yaml.safe_load(
    (ROOT / "data" / "seed_missions.yaml").read_text("utf-8")
)["missions"]
ALL = FALLBACKS + SEEDS

_GAMIFIED = re.compile(r"\b(challenge|streak|points?|win|score|badge|reward|level)\b", re.I)
_YOU = re.compile(r"\b(you|your)\b", re.I)
_TOKEN = re.compile(r"[A-Za-z']+")


def _time_regex() -> re.Pattern[str]:
    latin = [w for w in WORDS["time_of_day_words"] if w.isascii()]
    other = [re.escape(w) for w in WORDS["time_of_day_words"] if not w.isascii()]
    alts = [r"\b" + re.escape(w) + r"\b" for w in latin] + other
    return re.compile("|".join(alts), re.I)


TIME = _time_regex()


def test_counts() -> None:
    assert len(FALLBACKS) >= 15
    assert len(SEEDS) == 3


def test_unique_ids() -> None:
    ids = [m["id"] for m in ALL]
    assert len(ids) == len(set(ids))


def test_types() -> None:
    assert {m["type"] for m in FALLBACKS} == {"sense", "count", "change"}
    assert all(m["type"] == "seed" for m in SEEDS)


@pytest.mark.parametrize("m", ALL, ids=lambda m: m["id"])
def test_text_rules(m: dict[str, Any]) -> None:
    text: str = m["text"]
    assert len(text.split()) <= 30
    assert _YOU.search(text)
    assert not TIME.search(text)
    assert not _GAMIFIED.search(text)
    assert text.count(".") + text.count("?") + text.count("!") == 1  # one sentence, one task
    for item in WORDS["blocklist"]:
        for pat in item["patterns"]:
            assert not re.search(pat, text, re.I), item["id"]
    stems = tuple(WORDS["verb_allowlist"])
    first = _TOKEN.findall(text)[0].lower()
    assert first.startswith(tuple(s[:-1] if s.endswith("e") else s for s in stems)), first


def test_safety_filter_accepts_all() -> None:
    safety = pytest.importorskip("hub.safety")
    for m in ALL:
        assert safety.check_fallback_text(m["text"]).passed, m["id"]
