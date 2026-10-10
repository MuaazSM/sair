"""training/audit.py: each PRD §12.2 rule passes on good data and trips on crafted bad data."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from training import audit

HINGLISH = [
    "aaj",
    "hum",
    "park",
    "gaye",
    "aur",
    "bahut",
    "accha",
    "laga",
    "hawa",
    "thi",
    "ki",
    "ke",
]
ENGLISH = [
    "we",
    "walked",
    "down",
    "the",
    "lane",
    "and",
    "saw",
    "some",
    "birds",
    "by",
    "the",
    "old",
    "gate",
]
DEVA = ["आज", "हम", "पार्क", "गए", "और", "बहुत", "अच्छा", "लगा", "हवा", "ठंडी", "थी"]
POSITIONS = ["start", "middle", "end"]


def _transcript(rng: random.Random, mix: str, words: int) -> str:
    vocab = {"hinglish": HINGLISH, "english": ENGLISH, "devanagari_heavy": DEVA}[mix]
    return " ".join(rng.choice(vocab) for _ in range(words))


def good_items(n: int = 140, seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    items = []
    for i in range(n):
        mix = "hinglish" if i % 20 < 8 else ("devanagari_heavy" if i % 20 < 11 else "english")
        n_facts = i % 7
        words = rng.randint(15, 120)  # independent of the fact count
        facts = [
            {"span": f"span {i} {j}", "kind": "sight", "subject": f"thing{i}x{j}",
             "certainty": "unsure" if j == 0 and i % 2 else "sure", "place": None}
            for j in range(n_facts)
        ]  # fmt: skip
        hedges = [POSITIONS[i % 3]] if facts and i % 2 else []
        items.append(
            {
                "item_id": f"s{i:03d}-00",
                "seed_id": f"s{i:03d}",
                "group": "Mom and Dad",
                "mission_text": None if i % 5 == 0 else "Notice one tree.",
                "transcript": _transcript(rng, mix, words),
                "gold": {"facts": facts, "mission_result": "not_mentioned", "mission_span": None},
                "meta": {"language_mix": mix, "style": [], "hard": i % 5 == 1,
                         "lures": [], "hedge_positions": hedges},
            }
        )  # fmt: skip
    return items


def sft_for(
    items: list[dict[str, Any]], entry_words: Any = None, template_every: int = 0
) -> list[dict[str, Any]]:
    rng = random.Random(3)
    rows = []
    for k, it in enumerate(items):
        n = entry_words(it) if entry_words else rng.randint(8, 40)
        gold = {"facts": [], "mission_result": "not_mentioned", "mission_span": None,
                "entry": " ".join(["word"] * n)}  # fmt: skip
        src = "template" if template_every and k % template_every == 0 else "candidate:0"
        rows.append(
            {
                "messages": [{"role": "assistant", "content": json.dumps(gold)}],
                "meta": {"item_id": it["item_id"], "source": src},
            }
        )
    return rows


def status(rules: list[audit.Rule]) -> dict[str, str]:
    return {r.name: r.status for r in rules}


def test_good_dataset_passes_every_rule() -> None:
    items = good_items()
    st = status(audit.audit(items, sft_for(items)))
    assert "FAIL" not in st.values(), st
    assert st["length_leak_entry"] == "PASS"
    assert st["template_gold_share"] == "INFO"


def test_length_leak_facts_trips() -> None:
    items = good_items()
    for it in items:
        it["transcript"] = " ".join(["hum"] * (10 + 25 * len(it["gold"]["facts"])))
    st = status(audit.audit(items))
    assert st["length_leak_facts"] == "FAIL"
    assert st["length_predictor"] == "FAIL"


def test_length_leak_entry_trips() -> None:
    items = good_items()
    rows = sft_for(items, entry_words=lambda it: len(it["transcript"].split()) // 2 + 5)
    assert status(audit.audit(items, rows))["length_leak_entry"] == "FAIL"


def test_fact_count_spread_trips() -> None:
    items = [it for it in good_items() if len(it["gold"]["facts"]) != 6]
    assert status(audit.audit(items))["fact_count_spread"] == "FAIL"


def test_subject_share_trips() -> None:
    items = good_items()
    for it in items[:20]:  # 20/140 = 14% > 5%
        it["gold"]["facts"].append(
            {"span": "x", "kind": "animal", "subject": "Koel", "certainty": "sure"}
        )
    assert status(audit.audit(items))["subject_share"] == "FAIL"


def test_hedge_positions_trip() -> None:
    items = good_items()
    for it in items:
        if it["meta"]["hedge_positions"]:
            it["meta"]["hedge_positions"] = ["start"]
    assert status(audit.audit(items))["hedge_positions"] == "FAIL"


def test_hedge_positions_fallback_without_meta() -> None:
    items = good_items()
    for it in items:
        it["meta"].pop("hedge_positions")
    # synthetic spans have no hedge cues -> no positions -> FAIL (not silently PASS)
    assert status(audit.audit(items))["hedge_positions"] == "FAIL"
    assert audit.hedge_position("shayad koel thi") == "start"
    assert audit.hedge_position("koel thi shayad") == "end"
    assert audit.hedge_position("koel maybe near the gate") == "middle"


def test_hinglish_meta_and_script_trip() -> None:
    items = good_items()
    for it in items:
        if it["meta"]["language_mix"] == "hinglish":
            it["meta"]["language_mix"] = "english"
    st = status(audit.audit(items))
    assert st["hinglish_meta"] == "FAIL" and st["hinglish_script"] == "PASS"
    items = good_items()
    rng = random.Random(1)
    for it in items:
        if it["meta"]["language_mix"] == "hinglish":
            it["transcript"] = _transcript(rng, "english", 30)  # labelled hinglish, English text
    st = status(audit.audit(items))
    assert st["hinglish_meta"] == "PASS" and st["hinglish_script"] == "FAIL"


def test_devanagari_meta_and_script_trip() -> None:
    items = good_items()
    rng = random.Random(2)
    for it in items:
        if it["meta"]["language_mix"] == "devanagari_heavy":
            it["transcript"] = _transcript(rng, "hinglish", 30)  # romanised, not Devanagari
    st = status(audit.audit(items))
    assert st["devanagari_meta"] == "PASS" and st["devanagari_script"] == "FAIL"
    items = good_items()
    for it in items:
        if it["meta"]["language_mix"] == "devanagari_heavy":
            it["meta"]["language_mix"] = "mixed"
    assert status(audit.audit(items))["devanagari_meta"] == "FAIL"


@pytest.mark.parametrize("every", [1, 2, 5, 100])
def test_mission_dropout_band(every: int) -> None:
    items = good_items()
    for i, it in enumerate(items):
        it["mission_text"] = None if i % every == 0 else "Notice one tree."
    expected = "PASS" if every == 5 else "FAIL"
    assert status(audit.audit(items))["mission_dropout"] == expected


def test_hard_share_band() -> None:
    items = good_items()
    for it in items:
        it["meta"]["hard"] = False
    assert status(audit.audit(items))["hard_share"] == "FAIL"


def test_template_share_is_reported_and_warns() -> None:
    items = good_items()
    st = status(audit.audit(items, sft_for(items, template_every=2)))
    assert st["template_gold_share"] == "WARN"
    assert "FAIL" not in st.values()


def test_spearman_basics() -> None:
    assert audit.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert audit.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert audit.spearman([1, 1, 1], [1, 2, 3]) is None
    assert audit.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9487, abs=1e-3)


def test_cli_strict_exit_codes(tmp_path: Path) -> None:
    good = tmp_path / "good.jsonl"
    good.write_text("\n".join(json.dumps(it) for it in good_items()) + "\n", encoding="utf-8")
    assert audit.main(["--items", str(good), "--sft", "--strict"]) == 0
    bad_items = copy.deepcopy(good_items())
    for it in bad_items:
        it["mission_text"] = None
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(json.dumps(it) for it in bad_items) + "\n", encoding="utf-8")
    assert audit.main(["--items", str(bad), "--sft", "--strict"]) == 1
    assert audit.main(["--items", str(bad), "--sft"]) == 0  # non-strict reports only
