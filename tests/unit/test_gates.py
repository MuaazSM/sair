"""G-export / G-ship decision rules with fakes (no models, synthetic toy items)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.items import EvalItem
from eval.systems import CheckSummary, ItemRecord
from training import gates

BASE = {
    "unsupported_rate": 0.10,
    "fact_recall": 0.80,
    "fallback_rate": 0.05,
    "valid_json_no_grammar": 0.95,
}


# --------------------------------------------------------------------------- G-export


def test_export_pass_exactly_at_2pp_boundary() -> None:
    g = {
        "unsupported_rate": 0.12,
        "fact_recall": 0.78,
        "fallback_rate": 0.07,
        "valid_json_no_grammar": 0.93,
    }
    r = gates.compare_export(BASE, g)
    assert r["pass"] is True
    assert {m: row["delta_pp"] for m, row in r["metrics"].items()} == {
        "unsupported_rate": 2.0,
        "fact_recall": -2.0,
        "fallback_rate": 2.0,
        "valid_json_no_grammar": -2.0,
    }


@pytest.mark.parametrize("metric", gates.EXPORT_METRICS)
def test_export_fails_just_past_boundary(metric: str) -> None:
    g = dict(BASE)
    g[metric] = round(BASE[metric] + 0.0201, 4)
    r = gates.compare_export(BASE, g)
    assert r["pass"] is False and r["metrics"][metric]["pass"] is False
    assert all(row["pass"] for m, row in r["metrics"].items() if m != metric)


def test_export_none_handling() -> None:
    t = dict(BASE, fact_recall=None)
    assert gates.compare_export(t, dict(BASE, fact_recall=None))["pass"] is True  # n/a both
    assert gates.compare_export(t, BASE)["pass"] is False  # one side missing


def test_select_epoch_rule() -> None:
    eps = [
        {"epoch": 1, "dev": {"unsupported_rate": 0.05, "fact_recall": 0.60, "fallback_rate": 0.1}},
        {"epoch": 2, "dev": {"unsupported_rate": 0.08, "fact_recall": 0.75, "fallback_rate": 0.1}},
        {"epoch": 3, "dev": {"unsupported_rate": 0.08, "fact_recall": 0.80, "fallback_rate": 0.1}},
    ]
    c = {"fact_recall": 0.70, "fallback_rate": 0.10}
    assert gates.select_epoch(eps, c) == 2  # e1 fails recall; e2/e3 tie -> earlier
    assert gates.select_epoch(eps, {"fact_recall": 0.9, "fallback_rate": 0.1}) is None


ITEM = {
    "item_id": "s001-00",
    "seed_id": "s001",
    "group": "Mom and Dad",
    "mission_text": None,
    "transcript": "Aaj walk pe koel was calling near the temple. Bas itna hi.",
    "gold": {
        "facts": [
            {
                "span": "koel was calling near the temple",
                "kind": "sound",
                "subject": "koel",
                "certainty": "sure",
                "place": None,
            }
        ],
        "mission_result": "not_mentioned",
        "mission_span": None,
    },
}
GOOD = json.dumps(
    {
        "facts": ITEM["gold"]["facts"],
        "mission_result": "not_mentioned",  # type: ignore[index]
        "mission_span": None,
        "entry": "Mom and Dad heard a koel calling near the temple.",
    }
)


def _run_file(tmp: Path, dev: dict[str, Any], n: int) -> None:
    (tmp / "r1.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "config": {},
                "epochs": [
                    {
                        "epoch": 1,
                        "sampler_path": "tinker://x/sampler_weights/r1-e1",
                        "dev": dev,
                        "dev_n": n,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_g_export_end_to_end_with_fake_server(tmp_path: Path) -> None:
    items = [dict(ITEM, item_id=f"s001-0{i}") for i in range(4)]
    raws = [GOOD, GOOD, GOOD, "not json"]
    seen: list[Any] = []

    def gen(messages: list[Any]) -> str:
        seen.append(messages)
        return raws[len(seen) - 1]

    # what train.py would have stored for the same outputs
    from training.train import score_dev

    tinker = score_dev(items, raws).rates()
    _run_file(tmp_path, tinker, 4)
    out = tmp_path / "g_export.json"
    r = gates.g_export(
        "r1",
        epoch=1,
        generate=gen,
        runs_dir=tmp_path,
        out=out,
        items=items,
        comparable=lambda run: None,
    )
    assert r["pass"] is True and len(seen) == 4
    assert json.loads(out.read_text(encoding="utf-8"))["pass"] is True
    assert r["metrics"]["fallback_rate"]["gguf"] == 0.25

    _run_file(tmp_path, dict(tinker, fallback_rate=0.0), 4)  # Tinker had no fallbacks
    seen.clear()
    r = gates.g_export(
        "r1",
        epoch=1,
        generate=gen,
        runs_dir=tmp_path,
        out=out,
        items=items,
        comparable=lambda run: None,
    )
    assert r["pass"] is False and r["metrics"]["fallback_rate"]["delta_pp"] == 25.0


def test_g_export_refusals(tmp_path: Path) -> None:
    _run_file(tmp_path, BASE, 3)
    with pytest.raises(gates.Refused, match="dev item count"):
        gates.g_export(
            "r1",
            epoch=1,
            generate=lambda m: GOOD,
            runs_dir=tmp_path,
            out=tmp_path / "o.json",
            items=[ITEM],
            comparable=lambda run: None,
        )
    with pytest.raises(gates.Refused, match="--epoch"):
        gates.g_export(
            "r1",
            generate=lambda m: GOOD,
            runs_dir=tmp_path,
            out=tmp_path / "o.json",
            items=[ITEM],
            comparable=lambda run: None,
            c_ref_path=tmp_path / "none.json",
        )
    with pytest.raises(gates.Refused, match="lexicon"):
        gates.check_comparable({"lexicon_hash": "0" * 64, "prompt_fingerprint": "x"})


# --------------------------------------------------------------------------- G-ship table

C = {"unsupported_rate": 0.20, "fact_recall": 0.70, "template_fallback_rate": 0.10}


@pytest.mark.parametrize(
    ("d", "gx", "ci", "rec", "reasons", "borderline"),
    [
        (dict(C, unsupported_rate=0.10), True, (-0.15, -0.05), "llama", [], False),
        (dict(C, unsupported_rate=0.10), True, (-0.15, 0.02), "llama", [], True),
        (dict(C, unsupported_rate=0.10), True, None, "llama", [], True),  # degenerate CI
        (dict(C), True, (-0.05, 0.05), "gemma", ["unsupported_not_lower"], True),
        (
            dict(C, unsupported_rate=0.25),
            True,
            (0.01, 0.1),
            "gemma",
            ["unsupported_not_lower"],
            False,
        ),
        (
            dict(C, unsupported_rate=0.10, fact_recall=0.69),
            True,
            (-0.2, -0.01),
            "gemma",
            ["recall_below_c"],
            False,
        ),
        (
            dict(C, unsupported_rate=0.10, template_fallback_rate=0.11),
            True,
            (-0.2, -0.01),
            "gemma",
            ["fallback_above_c"],
            False,
        ),
        (
            dict(C, unsupported_rate=0.10, fact_recall=0.70, template_fallback_rate=0.10),
            False,
            (-0.2, -0.01),
            "gemma",
            ["g_export_failed"],
            False,
        ),
        (dict(C, unsupported_rate=0.10), None, (-0.2, -0.01), "gemma", ["g_export_missing"], False),
        (
            dict(C, unsupported_rate=None),
            True,
            None,
            "gemma",
            ["metric_unavailable:unsupported_rate"],
            True,
        ),
        (
            dict(C, unsupported_rate=0.10, fact_recall=None),
            True,
            (-0.2, -0.01),
            "gemma",
            ["metric_unavailable:fact_recall"],
            False,
        ),
    ],
)
def test_ship_decision_table(
    d: dict[str, Any], gx: bool | None, ci: Any, rec: str, reasons: list[str], borderline: bool
) -> None:
    r = gates.ship_decision(d, C, g_export_passed=gx, unsupported_diff_ci=ci)
    assert (r["recommendation"], r["reasons"], r["borderline"]) == (rec, reasons, borderline)


def test_ship_equal_recall_and_fallback_allowed() -> None:
    r = gates.ship_decision(
        dict(C, unsupported_rate=0.19), C, g_export_passed=True, unsupported_diff_ci=(-0.03, -0.001)
    )
    assert r["recommendation"] == "llama"


def _summary(unsupported: bool) -> CheckSummary:
    return CheckSummary(
        passed=not unsupported,
        reasons=[],
        verified_facts=[],
        dropped_facts=0,
        mission_result="not_mentioned",
        mission_span=None,
        unsupported_details=["x"] if unsupported else [],
    )


def _rec(
    system: str, iid: str, seed: str, unsupported: bool, origin: str = "model"
) -> dict[str, Any]:
    r = ItemRecord(
        item_id=iid,
        seed=seed,
        system=system,
        grammar=True,
        final_origin=origin,
        final={"entry": "e"},
        final_check=_summary(unsupported),
    )
    return r.to_dict()


def test_g_ship_end_to_end(tmp_path: Path) -> None:
    items = [
        EvalItem.model_validate({"id": f"s{k}-0", "seed": f"s{k}", "group": "G", "transcript": "t"})
        for k in range(6)
    ]
    ip = tmp_path / "dev.jsonl"
    ip.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")
    c_unsup = [True, True, True, False, False, False]
    d_unsup = [False, False, True, False, False, False]
    rc = tmp_path / "c.jsonl"
    rd = tmp_path / "d.jsonl"
    rc.write_text(
        "".join(
            json.dumps(_rec("C+", i.id, i.seed, u)) + "\n"
            for i, u in zip(items, c_unsup, strict=True)
        ),
        encoding="utf-8",
    )
    rd.write_text(
        "".join(
            json.dumps(_rec("D+", i.id, i.seed, u)) + "\n"
            for i, u in zip(items, d_unsup, strict=True)
        ),
        encoding="utf-8",
    )
    gx = tmp_path / "g_export.json"
    gx.write_text(json.dumps({"pass": True}), encoding="utf-8")
    out = tmp_path / "g_ship.json"
    r = gates.g_ship([rc, rd], items_path=ip, g_export_path=gx, out=out, resamples=500)
    m = r["metrics"]
    assert m["systems"]["D+"]["unsupported_rate"]["point"] == pytest.approx(1 / 6)
    assert m["systems"]["C+"]["unsupported_rate"]["point"] == pytest.approx(0.5)
    assert m["paired"]["unsupported_rate"]["point"] == pytest.approx(1 / 6 - 0.5)
    assert r["recommendation"] == "llama" and r["reasons"] == []
    assert isinstance(r["borderline"], bool)
    assert json.loads(out.read_text(encoding="utf-8"))["recommendation"] == "llama"

    gx.unlink()
    assert gates.g_ship([rc, rd], items_path=ip, g_export_path=gx, out=out, resamples=200)[
        "reasons"
    ] == ["g_export_missing"]
    with pytest.raises(gates.Refused, match="both"):
        gates.g_ship([rc], items_path=ip, g_export_path=gx, out=out, resamples=200)


def test_gates_never_touch_env_files() -> None:
    import re

    src = Path(gates.__file__).read_text(encoding="utf-8")
    assert not re.search(r"[\'\"/]\.env", src)
