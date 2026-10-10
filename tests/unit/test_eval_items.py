"""eval/items.py: the generator's ``item_id``/``seed_id``/``mission_text`` spelling is accepted."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import run
from eval.backends import TemplateWriter
from eval.items import EvalItem, ItemsError, load_items

GOLD = {
    "facts": [
        {
            "span": "we heard a crow near the gate",
            "kind": "sound",
            "subject": "crow",
            "certainty": "sure",
            "place": None,
        }
    ],
    "mission_result": "not_mentioned",
    "mission_span": None,
}


def _generator_row(i: int, seed: str) -> dict[str, object]:
    """Shape written by training/split.py (synthetic, generic place)."""
    return {
        "item_id": f"{seed}-{i:02d}",
        "seed_id": seed,
        "group": "Mom and Dad",
        "mission_text": "Listen for one bird you can hear from the path.",
        "transcript": f"we heard a crow near the gate, bas itna hi {i}",
        "gold": GOLD,
        "meta": {"language_mix": "hinglish", "style": ["very_short"], "hard": False},
    }


def test_generator_keys_alias_eval_keys() -> None:
    gen = EvalItem.model_validate(_generator_row(0, "s031"))
    assert (gen.id, gen.seed) == ("s031-00", "s031")
    assert gen.mission == "Listen for one bird you can hear from the path."
    assert gen.meta is not None and gen.meta["language_mix"] == "hinglish"
    legacy = EvalItem.model_validate(
        {"id": "s031-00", "seed": "s031", "group": "Mom and Dad", "transcript": "x"}
    )
    assert (legacy.id, legacy.seed, legacy.mission, legacy.meta) == ("s031-00", "s031", None, None)
    # both spellings agreeing is fine; disagreeing is rejected
    same = dict(_generator_row(1, "s031"), seed="s031")
    assert EvalItem.model_validate(same).seed == "s031"
    with pytest.raises(ValueError):
        EvalItem.model_validate(dict(_generator_row(1, "s031"), seed="s999"))
    # unknown keys are still rejected (extra="forbid")
    with pytest.raises(ValueError):
        EvalItem.model_validate(dict(_generator_row(1, "s031"), surprise=1))


def test_load_items_reads_generator_file_and_conflict_message_has_no_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "test.jsonl"
    rows = [_generator_row(i, f"s{s:03d}") for s in range(2) for i in range(2)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    items = load_items(path)
    assert [it.seed for it in items] == ["s000", "s000", "s001", "s001"]

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(dict(rows[0], seed="other")) + "\n", encoding="utf-8")
    with pytest.raises(ItemsError) as exc:
        load_items(bad)
    assert str(exc.value) == "invalid item on line 1"


def test_checked_in_synthetic_test_split_loads() -> None:
    path = Path(run.__file__).resolve().parent / "data" / "synthetic" / "test.jsonl"
    if not path.is_file():
        pytest.skip("no synthetic test split yet")
    items = load_items(path)
    assert items and all(it.seed for it in items)


def test_run_cli_on_generator_split(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    rows = [_generator_row(i, f"s{s:03d}") for s in range(3) for i in range(2)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    code = run.main(
        ["--systems", "A", "--items", str(path), "--out", str(out), "--resamples", "100"],
        writers={"template": TemplateWriter()},
    )
    assert code == 0
    assert any(out.iterdir())
