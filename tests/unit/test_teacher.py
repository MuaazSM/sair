"""training/teacher.py with a fake sampler (no network). Synthetic fixtures only."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from hub.grounding import build_template_entry, check_entry
from hub.prompts.builders import MissionText, Transcript, WalkerGroup, build_writer_prompt
from hub.schemas import EntryOrigin, Fact, parse_writer_output
from training import teacher
from training.tinker_common import Usage, as_dicts, gold_json

GROUP = "Mom and Dad"
TRANSCRIPT = (
    "Aaj walk pe koel was calling near the temple. Shayad ek parrot tha neem pe. "
    "Mission done, gulmohar pe phool aa gaye."
)
F1 = {
    "span": "koel was calling near the temple",
    "kind": "sound",
    "subject": "koel",
    "certainty": "sure",
    "place": "near the temple",
}
F2 = {
    "span": "Shayad ek parrot tha neem pe",
    "kind": "animal",
    "subject": "parrot",
    "certainty": "unsure",
    "place": "neem",
}
BASE = {"facts": [F1, F2], "mission_result": "done", "mission_span": "gulmohar pe phool aa gaye"}
GOOD = json.dumps(
    {
        **BASE,
        "entry": "Mom and Dad heard a koel calling near the temple. "
        "They maybe saw a parrot on the neem.",
    }
)
GOOD_2 = json.dumps({**BASE, "entry": "Mom and Dad heard a koel near the temple."})
BAD_DETAIL = json.dumps(
    {**BASE, "entry": "Mom and Dad heard a koel near the temple. They saw a red parrot."}
)
BAD_HEDGE = json.dumps({**BASE, "entry": "Mom and Dad heard a koel and saw a parrot."})
NOT_JSON = "Sure! Here is the entry: Mom and Dad went walking."
# passes, but one fact has an invented span -> cleaned gold must drop it
DROPS = json.dumps(
    {
        **BASE,
        "facts": [F1, {**F2, "span": "a span that is not there"}],
        "entry": "Mom and Dad heard a koel calling near the temple.",
    }
)


def make_item(
    item_id: str = "seed01-00", mission: str | None = "Notice the gulmohar."
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "seed_id": "seed01",
        "group": GROUP,
        "mission_text": mission,
        "transcript": TRANSCRIPT,
        "gold": dict(BASE),
        "meta": {"language_mix": "hinglish", "style": [], "hard": True, "lures": []},
    }


class FakeSampler:
    """Returns scripted candidate texts per item; records every prompt it is given."""

    def __init__(self, script: dict[str, list[str]], model: str = "fake/teacher") -> None:
        self.model = model
        self.script = {k: list(v) for k, v in script.items()}
        self.prompts: list[list[dict[str, str]]] = []
        self.requests: list[int] = []
        self._by_ids: dict[int, str] = {}

    def prompt_ids(self, messages: list[dict[str, str]]) -> list[int]:
        self.prompts.append(messages)
        key = len(self.prompts)
        user = messages[1]["content"]
        item = next(k for k in self.script if f"<<<\n{TRANSCRIPT}" in user and k)
        self._by_ids[key] = item
        return [key] * 10

    def submit(
        self, prompt_ids: list[int], n: int, temperature: float, max_tokens: int
    ) -> Callable[[], teacher.SampleResult]:
        self.requests.append(n)
        item = self._by_ids[prompt_ids[0]]
        texts = [self.script[item].pop(0) for _ in range(n)]

        def fin() -> teacher.SampleResult:
            return teacher.SampleResult(
                texts=texts, out_tokens=[50] * n, stop_reasons=["stop"] * n, prompt_tokens=10
            )

        return fin


class ExplodingSampler(FakeSampler):
    def submit(self, *a: Any, **k: Any) -> Callable[[], teacher.SampleResult]:
        raise AssertionError("must not call the API when the cache is complete")


def test_first_passing_candidate_is_kept_and_cleaned() -> None:
    sel = teacher.select_gold(make_item(), [NOT_JSON, BAD_DETAIL, GOOD, GOOD_2])
    assert sel.winner == 2
    assert [e.passed for e in sel.evals] == [False, False, True]
    assert sel.evals[0].codes == ["invalid_json"]
    assert "unsupported_detail" in sel.evals[1].codes
    assert sel.gold is not None
    assert sel.gold.entry == json.loads(GOOD)["entry"]
    chk = check_entry(sel.gold, TRANSCRIPT, GROUP, origin=EntryOrigin.MODEL)
    assert chk.passed and chk.reasons == ()


def test_passing_candidate_with_dropped_fact_is_cleaned() -> None:
    sel = teacher.select_gold(make_item(), [DROPS])
    assert sel.winner == 0 and sel.gold is not None
    assert [f.span for f in sel.gold.facts] == [F1["span"]]
    assert sel.evals[0].codes == ["span_not_in_transcript"]


def test_template_when_no_candidate_passes() -> None:
    sel = teacher.select_gold(make_item(), [NOT_JSON, BAD_DETAIL, BAD_HEDGE, BAD_HEDGE])
    assert sel.winner == teacher.TEMPLATE
    assert sel.gold is not None
    facts = [Fact.model_validate(f) for f in BASE["facts"]]
    assert sel.gold.entry == build_template_entry(GROUP, facts, True)
    assert sel.gold.mission_result.value == "done"
    assert check_entry(sel.gold, TRANSCRIPT, GROUP, origin=EntryOrigin.TEMPLATE).passed


def test_fenced_json_is_accepted() -> None:
    sel = teacher.select_gold(make_item(), ["```json\n" + GOOD + "\n```"])
    assert sel.winner == 0 and sel.evals[0].fenced


def test_gold_json_is_compact_with_key_order() -> None:
    out = parse_writer_output(GOOD).value
    assert out is not None
    s = gold_json(out)
    assert ", " not in s.split('"entry"')[0] and ": " not in s.split('"entry"')[0]
    assert list(json.loads(s)) == ["facts", "mission_result", "mission_span", "entry"]
    assert parse_writer_output(s).value == out


def test_production_prompt_used_verbatim(tmp_path: Path) -> None:
    items = [make_item("a-00"), make_item("b-00", mission=None)]
    fake = FakeSampler({"a-00": [GOOD] * 4, "b-00": [GOOD] * 4})
    # route by order: both items share the transcript, so map ids by prompt order
    fake.prompt_ids = _ordered_prompt_ids(fake, ["a-00", "b-00"])  # type: ignore[method-assign]
    cfg = teacher.RunConfig(model=fake.model)
    res = teacher.run_items(items, fake, cfg, cache_dir=tmp_path)
    expected_a = as_dicts(
        build_writer_prompt(
            WalkerGroup(GROUP), MissionText("Notice the gulmohar."), Transcript(TRANSCRIPT)
        )
    )
    expected_b = as_dicts(build_writer_prompt(WalkerGroup(GROUP), None, Transcript(TRANSCRIPT)))
    assert fake.prompts == [expected_a, expected_b]
    assert "Today's mission: (none)" in expected_b[1]["content"]
    row = teacher.sft_row(items[0], res["a-00"])
    assert row is not None
    assert row["messages"][:2] == expected_a
    assert row["messages"][2]["role"] == "assistant"
    assert row["meta"]["source"] == "candidate:0"


def _ordered_prompt_ids(
    fake: FakeSampler, order: list[str]
) -> Callable[[list[dict[str, str]]], list[int]]:
    def prompt_ids(messages: list[dict[str, str]]) -> list[int]:
        fake.prompts.append(messages)
        key = len(fake.prompts)
        fake._by_ids[key] = order[key - 1]
        return [key] * 10

    return prompt_ids


def test_cache_resume_skips_api(tmp_path: Path) -> None:
    items = [make_item("a-00")]
    cfg = teacher.RunConfig(model="fake/teacher")
    first = FakeSampler({"a-00": [BAD_DETAIL, NOT_JSON, GOOD, GOOD_2]})
    r1 = teacher.run_items(items, first, cfg, cache_dir=tmp_path)
    assert first.requests == [4]
    cached = json.loads((tmp_path / "a-00.json").read_text(encoding="utf-8"))
    assert len(cached["candidates"]) == 4 and cached["model"] == "fake/teacher"
    again = ExplodingSampler({"a-00": []})
    r2 = teacher.run_items(items, again, cfg, cache_dir=tmp_path)
    assert r1["a-00"].winner == r2["a-00"].winner == 2
    assert teacher.run_items(items, None, cfg, cache_dir=tmp_path)["a-00"].winner == 2
    # a different model is a different cache key -> resample
    other = FakeSampler({"a-00": [GOOD] * 4}, model="fake/other")
    teacher.run_items(items, other, teacher.RunConfig(model="fake/other"), cache_dir=tmp_path)
    assert other.requests == [4]


def test_staged_sampling_only_tops_up_failures(tmp_path: Path) -> None:
    items = [make_item("a-00")]
    cfg = teacher.RunConfig(model="fake/teacher", candidates=4, first_stage=2)
    fake = FakeSampler({"a-00": [BAD_DETAIL, NOT_JSON, BAD_HEDGE, GOOD]})
    res = teacher.run_items(items, fake, cfg, cache_dir=tmp_path)
    assert fake.requests == [2, 2]
    assert res["a-00"].winner == 3
    fake2 = FakeSampler({"a-00": [GOOD, GOOD]})
    res2 = teacher.run_items(items, fake2, cfg, cache_dir=tmp_path / "b")
    assert fake2.requests == [2] and res2["a-00"].winner == 0


def test_max_requests_cap(tmp_path: Path) -> None:
    items = [make_item("a-00"), make_item("b-00")]
    fake = FakeSampler({"a-00": [GOOD] * 4, "b-00": [GOOD] * 4})
    fake.prompt_ids = _ordered_prompt_ids(fake, ["a-00", "b-00"])  # type: ignore[method-assign]
    cfg = teacher.RunConfig(model=fake.model, max_requests=1, concurrency=1)
    with pytest.raises(teacher.Budget):
        teacher.run_items(items, fake, cfg, cache_dir=tmp_path)
    assert fake.requests == [4]


def test_unsafe_item_id_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        teacher.run_items([make_item("../x")], None, teacher.RunConfig(), cache_dir=tmp_path)


def test_usage_cost_uses_published_rates() -> None:
    u = Usage("Qwen/Qwen3.5-397B-A17B")
    u.add_sample(prompt_tokens=1_000_000, cached=0, num_samples=4, out=1_000_000)
    # 1M full prefill + 3M cached prefill + 1M sampled
    assert u.cost() == pytest.approx(1.65 + 3 * 0.33 + 7.50)
    unknown = Usage("someone/unknown-model")
    unknown.add_sample(1_000_000, 0, 1, 0)
    assert unknown.to_dict()["rates_known"] is False and unknown.cost() == pytest.approx(3.60)


def test_summary_pass_rates_by_language_mix() -> None:
    a, b = make_item("a-00"), make_item("b-00")
    b["meta"]["language_mix"] = "english"
    res = {"a-00": teacher.select_gold(a, [GOOD]), "b-00": teacher.select_gold(b, [NOT_JSON])}
    s = teacher.summarise([a, b], res)
    assert s["pass_rate_by_language_mix"] == {
        "english": {"items": 1, "pass_rate": 0.0},
        "hinglish": {"items": 1, "pass_rate": 1.0},
    }
    assert s["template_share"] == 0.5 and s["item_pass_rate"] == 0.5
