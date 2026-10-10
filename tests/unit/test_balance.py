"""Unit tests for training/balance.py (synthetic items only)."""

from __future__ import annotations

import random
from typing import Any

from training import balance


def make(i: int, words: int, subjects: list[str], hard: bool = False) -> dict[str, Any]:
    return {
        "item_id": f"x{i:04d}",
        "seed_id": f"s{i % 20:03d}",
        "transcript": " ".join(["w"] * words),
        "gold": {
            "facts": [{"subject": s, "span": "x"} for s in subjects],
            "mission_result": "done",
            "mission_span": None,
        },
        "meta": {"hard": hard},
    }


def corpus(n: int = 600, seed: int = 3) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pool = [f"subj{k}" for k in range(400)]
    items = []
    for i in range(n):
        k = rng.randint(0, 6)
        words = 10 + 8 * k + rng.randint(0, 20)  # strong length / fact-count leak
        subs = ["clouds"] if rng.random() < 0.2 else []
        subs += rng.sample(pool, max(k - len(subs), 0))
        items.append(make(i, words, subs, hard=rng.random() < 0.2))
    return items


def test_subject_cap_enforced_and_deterministic() -> None:
    items = corpus()
    assert balance.subject_counts(items)["clouds"] / len(items) > 0.05
    a, dropped_a = balance.cap_subjects(items)
    b, dropped_b = balance.cap_subjects(list(reversed(items)))
    top = balance.subject_counts(a).most_common(1)[0][1] / len(a)
    assert top <= balance.SUBJECT_CAP
    assert dropped_a and {i for i, _ in dropped_a} == {i for i, _ in dropped_b}
    assert {i["item_id"] for i in a} == {i["item_id"] for i in b}


def test_non_hard_dropped_first() -> None:
    items = corpus()
    _, dropped = balance.cap_subjects(items)
    by_id = {i["item_id"]: i for i in items}
    assert dropped
    assert sum(bool(by_id[d]["meta"]["hard"]) for d, _ in dropped) == 0


def test_length_leak_reduced_below_threshold() -> None:
    items = corpus(1200)
    before = balance.rho(items)
    assert before is not None and before > balance.RHO_MAX
    kept, dropped, ok = balance.cut_length_leak(items, max_share=0.9)
    after = balance.rho(kept)
    assert ok and after is not None and after <= balance.RHO_MAX
    assert len(kept) + len(dropped) == len(items)


def test_leak_drop_cap_reports_failure() -> None:
    items = corpus(300)
    kept, dropped, ok = balance.cut_length_leak(items, max_share=0.01)
    assert not ok and len(dropped) == 3


def test_balance_does_not_mutate_input_and_is_deterministic() -> None:
    items = corpus(800)
    snapshot = [dict(i) for i in items]
    k1, d1, _ = balance.balance(items)
    k2, d2, _ = balance.balance(items)
    assert items == snapshot
    assert d1 == d2 and [i["item_id"] for i in k1] == [i["item_id"] for i in k2]
