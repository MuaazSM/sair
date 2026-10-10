"""Unit tests for training/split.py (synthetic items only)."""

from __future__ import annotations

from typing import Any

from training import split

N_SEEDS = 80
PER_SEED = 17


def make_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for s in range(N_SEEDS):
        for v in range(PER_SEED):
            items.append(
                {
                    "item_id": f"s{s:03d}-{v:02d}",
                    "seed_id": f"s{s:03d}",
                    "group": "Mom",
                    "mission_text": "Listen for birds.",
                    "transcript": "x",
                    "gold": {"facts": [], "mission_result": "done", "mission_span": "we did"},
                    "meta": {"language_mix": "english", "style": [], "hard": False, "lures": []},
                }
            )
    return items


def test_split_is_by_seed() -> None:
    splits = split.run(make_items())
    seen: dict[str, str] = {}
    for name, rows in splits.items():
        for r in rows:
            assert seen.setdefault(r["seed_id"], name) == name
    assert sum(len(r) for r in splits.values()) == N_SEEDS * PER_SEED


def test_split_80_10_10_seeds() -> None:
    splits = split.run(make_items())
    counts = {n: len({r["seed_id"] for r in rows}) for n, rows in splits.items()}
    assert counts == {"train": 64, "dev": 8, "test": 8}


def test_split_deterministic_and_rng_dependent() -> None:
    items = make_items()
    a = split.split_seeds(i["seed_id"] for i in items)
    assert a == split.split_seeds(i["seed_id"] for i in reversed(items))
    assert a != split.split_seeds((i["seed_id"] for i in items), rng_seed=1)


def test_dropout_fraction_and_gold() -> None:
    out = split.apply_dropout(make_items())
    dropped = [i for i in out if i["meta"]["mission_dropout"]]
    frac = len(dropped) / len(out)
    assert abs(frac - 0.20) <= 0.03
    for i in dropped:
        assert i["mission_text"] is None
        assert i["gold"]["mission_result"] == "not_mentioned"
        assert i["gold"]["mission_span"] is None
    for i in out:
        if not i["meta"]["mission_dropout"]:
            assert i["mission_text"] and i["gold"]["mission_result"] == "done"


def test_dropout_does_not_mutate_input_and_is_deterministic() -> None:
    items = make_items()
    a = split.apply_dropout(items)
    assert items[0]["mission_text"] == "Listen for birds."
    assert a == split.apply_dropout(items)
