"""eval/run.py on synthetic toy items: outputs, --cached, --assert-baseline, no-data exits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import run
from eval.backends import CachedWriter, CacheMode, StubWriter, TemplateWriter

GOOD = json.dumps(
    {
        "facts": [
            {
                "span": "we heard a crow near the gate",
                "kind": "sound",
                "subject": "crow",
                "certainty": "sure",
                "place": "near the gate",
            }
        ],
        "mission_result": "not_mentioned",
        "mission_span": None,
        "entry": "Mom and Dad heard a crow near the gate.",
    }
)


def _items(path: Path, n_seeds: int = 3, per: int = 2) -> Path:
    rows = []
    for s in range(n_seeds):
        for k in range(per):
            rows.append(
                {
                    "id": f"s{s}-{k}",
                    "seed": f"s{s}",
                    "group": "Mom and Dad",
                    "mission": None,
                    "transcript": f"we heard a crow near the gate, item {s} {k}",
                    "gold": {
                        "facts": [
                            {
                                "span": "we heard a crow near the gate",
                                "subject": "crow",
                                "kind": "sound",
                                "certainty": "sure",
                            }
                        ],
                        "mission_result": "not_mentioned",
                    },
                }
            )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_synthetic_run_writes_outputs(tmp_path: Path) -> None:
    items = _items(tmp_path / "test.jsonl")
    out = tmp_path / "out"
    writers = {
        "template": TemplateWriter(),
        "sair_writer": StubWriter([GOOD]),
        "qwen_base": StubWriter([GOOD]),
    }
    code = run.main(
        ["--items", str(items), "--systems", "A,C+,D+", "--out", str(out), "--resamples", "100"],
        writers=writers,
    )
    assert code == 0
    for name in ("metrics.json", "metrics.csv", "paired.csv", "latency.csv", "items.jsonl"):
        assert (out / name).is_file()
    m = json.loads((out / "metrics.json").read_text("utf-8"))
    assert m["n_items"] == 6 and m["n_clusters"] == 3 and m["bootstrap"] == "cluster-by-seed"
    dp = m["systems"]["D+"]["metrics"]
    assert dp["fact_recall"]["point"] == 1.0 and dp["unsupported_rate"]["point"] == 0.0
    assert dp["fact_recall"]["n_clusters"] == 3
    assert m["systems"]["A"]["metrics"]["fact_recall"]["point"] == 0.0
    assert "D+ - C+" in m["paired"]
    assert len((out / "items.jsonl").read_text("utf-8").splitlines()) == 18


def test_cached_replay_and_baseline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    items = _items(tmp_path / "test.jsonl")
    cache = tmp_path / "cache"
    live = {"sair_writer": CachedWriter(StubWriter([GOOD]), cache, CacheMode.RESUME)}
    args = ["--items", str(items), "--systems", "D+", "--resamples", "100"]
    assert run.main(args + ["--out", str(tmp_path / "a")], writers=live) == 0
    baseline = tmp_path / "a" / "metrics.json"

    replay = {"sair_writer": CachedWriter(StubWriter([]), cache, CacheMode.CACHED)}
    code = run.main(
        args + ["--cached", "--out", str(tmp_path / "b"), "--assert-baseline", str(baseline)],
        writers=replay,
    )
    assert code == 0 and "baseline: ok" in capsys.readouterr().out

    data = json.loads(baseline.read_text("utf-8"))
    data["systems"]["D+"]["metrics"]["fact_recall"]["point"] = 0.5
    baseline.write_text(json.dumps(data), encoding="utf-8")
    replay = {"sair_writer": CachedWriter(StubWriter([]), cache, CacheMode.CACHED)}
    code = run.main(
        args + ["--cached", "--out", str(tmp_path / "c"), "--assert-baseline", str(baseline)],
        writers=replay,
    )
    assert code == 5 and "fact_recall" in capsys.readouterr().err


def test_cached_mode_missing_entry_fails_clearly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    items = _items(tmp_path / "test.jsonl")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "unrelated.json").write_text("{}", encoding="utf-8")
    replay = {"sair_writer": CachedWriter(StubWriter([GOOD]), cache, CacheMode.CACHED)}
    code = run.main(
        ["--items", str(items), "--systems", "D+", "--cached", "--out", str(tmp_path / "o")],
        writers=replay,
    )
    err = capsys.readouterr().err
    assert code == 4 and "cache miss" in err and "crow" not in err


def test_cached_mode_without_any_cache_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run, "SYNTHETIC_CACHE", tmp_path / "empty-cache")
    monkeypatch.setattr(run, "SYNTHETIC_ITEMS", tmp_path / "missing.jsonl")
    assert (
        run.main(
            ["--cached"],
            writers={
                k: StubWriter([GOOD]) for k in ("template", "gemma", "qwen_base", "sair_writer")
            },
        )
        == 2
    )
    assert "no cached results yet" in capsys.readouterr().err

    items = _items(tmp_path / "test.jsonl")
    code = run.main(
        ["--cached", "--items", str(items), "--systems", "D"],
        writers={"sair_writer": StubWriter([GOOD])},
    )
    assert code == 2 and "no cached results yet" in capsys.readouterr().err


def test_no_grammar_pass_reports_valid_json(tmp_path: Path) -> None:
    items = _items(tmp_path / "test.jsonl")
    w = StubWriter(lambda msgs, grammar: GOOD if grammar else "Sure! Here is the JSON: {")
    code = run.main(
        [
            "--items",
            str(items),
            "--systems",
            "B",
            "--no-grammar-pass",
            "--resamples",
            "100",
            "--out",
            str(tmp_path / "o"),
        ],
        writers={"gemma": w},
    )
    assert code == 0
    m = json.loads((tmp_path / "o" / "metrics.json").read_text("utf-8"))
    assert m["systems"]["B-nogrammar"]["metrics"]["valid_json_no_grammar"]["point"] == 0.0
    assert "valid_json_no_grammar" not in m["systems"]["B"]["metrics"]


def test_unknown_system_rejected() -> None:
    assert run.main(["--systems", "E"], writers={}) == 2
