"""FR-O4: health keywords in three scripts trigger a text-free ping."""

from __future__ import annotations

from pathlib import Path

import pytest

from hub.graphs import journal as J
from tests.unit._graphfix import EN_20S, GROUP, RecNotifier, journal_deps


@pytest.mark.parametrize(
    "text",
    [
        "We stopped by the gate because I felt dizzy for a minute.",  # English
        "Had chest pain near the park, then it went away.",
        "Aaj thoda chakkar aaya tha bench ke paas.",  # Hinglish
        "Ghutne mein dard tha toh hum jaldi laut aaye.",
        "आज पार्क में सीने में दर्द हुआ।",  # Devanagari
        "घुटने में दर्द था।",
    ],
)
def test_three_scripts_trigger(text: str) -> None:
    assert J.has_health_keyword(text)


@pytest.mark.parametrize(
    "text",
    [
        EN_20S,
        "Painted walls near the school looked bright today.",  # "pain" only inside a word
        "Hum park tak gaye aur koyal suni.",
        "मंदिर के पास कोयल बोल रही थी।",
    ],
)
def test_normal_text_does_not_trigger(text: str) -> None:
    assert not J.has_health_keyword(text)


def test_notifier_gets_no_text(tmp_path: Path) -> None:
    n = RecNotifier()
    deps = journal_deps(tmp_path, notifier=n)
    marker = "CANARY-health-7f3a I felt dizzy"
    J.notify({"walk_id": "w1", "group": GROUP, "transcript": marker, "facts": 0}, deps)
    assert n.health == [(GROUP,)]
    flat = repr(n.health) + repr(n.walks)
    assert "CANARY" not in flat and "dizzy" not in flat
