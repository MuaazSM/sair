"""Systems A-D / B+-D+ control flow and the call cache (stub backends only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.backends import (
    CachedWriter,
    CacheMiss,
    CacheMode,
    StubWriter,
    TemplateWriter,
    WriterBackend,
)
from eval.items import EvalItem
from eval.systems import run_item
from hub.prompts.builders import REGEN_HEADER

G = "Mom and Dad"
T = "we heard a crow near the gate, bas itna hi"
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
BAD = json.dumps(
    {
        "facts": [],
        "mission_result": "not_mentioned",
        "mission_span": None,
        "entry": "Mom and Dad saw a parrot.",
    }
)


def _item() -> EvalItem:
    return EvalItem(id="i1", seed="s1", group=G, mission="Listen for one bird.", transcript=T)


def test_backends_satisfy_protocol_and_expose_base_url() -> None:
    for b in (TemplateWriter(), StubWriter([GOOD])):
        assert isinstance(b, WriterBackend)
        assert b.base_url is None


def test_plain_system_single_attempt_even_if_check_fails() -> None:
    w = StubWriter([BAD, GOOD])
    r = run_item("D", _item(), w)
    assert len(w.calls) == 1 and len(r.attempts) == 1
    assert r.final_origin == "model" and r.final_check is not None and not r.final_check.passed


def test_plus_system_regen_then_model() -> None:
    w = StubWriter([BAD, GOOD])
    r = run_item("D+", _item(), w)
    assert r.regen_used and r.final_origin == "model"
    assert r.attempts[0].check is not None and not r.attempts[0].check.passed
    assert r.attempts[1].check is not None and r.attempts[1].check.passed
    regen_prompt = w.calls[1][0][-1].content
    assert (
        REGEN_HEADER in regen_prompt
        and "Mention only things that are in the facts." in regen_prompt
    )
    assert "parrot" not in regen_prompt  # previous output is never passed back
    assert r.attempts[1].regen_codes == ["unsupported_detail"]
    assert {"writer", "check", "writer_regen", "check_regen", "e2e"} <= set(r.latency)


def test_plus_system_first_attempt_passes_no_regen() -> None:
    w = StubWriter([GOOD])
    r = run_item("C+", _item(), w)
    assert not r.regen_used and len(w.calls) == 1 and r.final_origin == "model"


def test_plus_system_template_keeps_verified_facts_of_last_parseable_attempt() -> None:
    bad_with_fact = json.dumps(
        {**json.loads(GOOD), "entry": "Mom and Dad heard a crow and a parrot."}
    )
    r = run_item("B+", _item(), StubWriter([bad_with_fact, "not json"]))
    assert r.final_origin == "template"
    assert (
        r.final is not None
        and r.final["entry"] == "Mom and Dad went for a walk. They noticed crow."
    )
    assert r.final_check is not None and r.final_check.passed


def test_plus_system_two_unparseable_attempts_give_no_facts_template() -> None:
    w = StubWriter(["nope", "still nope"])
    r = run_item("B+", _item(), w)
    assert w.calls[1][0][-1].content.endswith("- Reply with one valid JSON object only.")
    assert r.final is not None and r.final["entry"] == "Mom and Dad went for their evening walk."


def test_grammar_flag_passed_through() -> None:
    w = StubWriter([GOOD])
    run_item("B", _item(), w, grammar=False)
    assert w.calls[0][1] is False


def test_backend_error_recorded_by_type_only() -> None:
    def boom(messages: object, grammar: bool) -> str:
        raise TimeoutError("secret transcript words")

    r = run_item("D+", _item(), StubWriter(boom))
    assert r.final_origin == "error" and r.error == "TimeoutError"
    assert "secret" not in json.dumps(r.to_dict())


def test_cache_resume_and_cached_modes(tmp_path: Path) -> None:
    inner = StubWriter([GOOD])
    cw = CachedWriter(inner, tmp_path, CacheMode.RESUME)
    run_item("D", _item(), cw)
    run_item("D", _item(), cw)
    assert len(inner.calls) == 1 and cw.store.hits == 1 and cw.store.writes == 1
    replay = CachedWriter(StubWriter([]), tmp_path, CacheMode.CACHED)
    r = run_item("D", _item(), replay)
    assert r.attempts[0].cached and r.final_origin == "model"


def test_cached_mode_missing_entry_raises_cache_miss(tmp_path: Path) -> None:
    cw = CachedWriter(StubWriter([GOOD]), tmp_path, CacheMode.CACHED)
    with pytest.raises(CacheMiss) as exc:
        run_item("D", _item(), cw)
    assert "crow" not in str(exc.value) and "prompt_sha=" in str(exc.value)


def test_cache_key_depends_on_backend_and_grammar(tmp_path: Path) -> None:
    a = CachedWriter(StubWriter([GOOD], backend_id="x"), tmp_path, CacheMode.RESUME)
    b = CachedWriter(StubWriter([GOOD], backend_id="y"), tmp_path, CacheMode.RESUME)
    run_item("D", _item(), a)
    run_item("D", _item(), b)
    run_item("D", _item(), a, grammar=False)
    assert len(list(tmp_path.glob("*.json"))) == 3
