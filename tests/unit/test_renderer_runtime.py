"""Runtime renderer == training render (string and tokens), few-shot for baseline C.

Synthetic items only. The tokenizer checks use the local HF cache (HF_HUB_OFFLINE=1) and
skip when it is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from hub.prompts.builders import ChatMessage
from training import renderer_runtime as rr
from training.tinker_common import as_dicts, render_prompt_text, writer_messages

G = "Mom and Dad"


def _item(
    iid: str,
    seed: str,
    transcript: str,
    facts: list[dict[str, Any]],
    mission: str | None,
    mr: str = "not_mentioned",
    span: str | None = None,
    mix: str = "english",
    hard: bool = False,
) -> dict[str, Any]:
    return {
        "item_id": iid,
        "seed_id": seed,
        "group": G,
        "mission_text": mission,
        "transcript": transcript,
        "gold": {"facts": facts, "mission_result": mr, "mission_span": span},
        "meta": {"language_mix": mix, "hard": hard},
    }


def _fact(span: str, subject: str, kind: str = "sound") -> dict[str, Any]:
    return {"span": span, "kind": kind, "subject": subject, "certainty": "sure", "place": None}


ITEMS = [
    _item(
        "s001-00",
        "s001",
        "Aaj walk pe koel was calling near the temple. Bas itna hi.",
        [_fact("koel was calling near the temple", "koel")],
        "Listen for one bird.",
        mix="hinglish",
    ),
    _item(
        "s002-03",
        "s002",
        "We saw a big rain tree by the lane. It was windy.",
        [_fact("We saw a big rain tree by the lane", "rain tree", "plant")],
        None,
    ),
    _item(
        "s003-01",
        "s003",
        "  {curly} braces <b>and</b> a crow cawing  ",
        [_fact("a crow cawing", "crow")],
        "Notice one sound {x}.",
        mix="english",
    ),
]

MESSAGE_LISTS = [as_dicts(writer_messages(it)) for it in ITEMS]


def _tokenizer() -> Any:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen/Qwen3.5-4B tokenizer not in the local HF cache")


# --------------------------------------------------------------------------- zero-shot (D)


@pytest.mark.parametrize("msgs", MESSAGE_LISTS, ids=["hinglish", "no-mission", "braces"])
def test_string_equals_training_render(msgs: list[dict[str, str]]) -> None:
    r = rr.RuntimeRenderer()
    as_chat = [ChatMessage(m["role"], m["content"]) for m in msgs]  # type: ignore[arg-type]
    assert r.render(as_chat) == render_prompt_text(msgs)
    assert r(as_chat) == render_prompt_text(msgs)  # hub Renderer protocol (callable)
    assert r.render(msgs) == render_prompt_text(msgs)


def test_handwritten_shape() -> None:
    r = rr.RuntimeRenderer()
    s = r.render([ChatMessage("system", " SYS \n"), ChatMessage("user", "\nUSR ")])
    assert s == (
        "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nUSR<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert r.stop_strings == ["<|im_end|>"]


@pytest.mark.parametrize("msgs", MESSAGE_LISTS, ids=["hinglish", "no-mission", "braces"])
def test_tokens_equal_training_prompt_tokens(msgs: list[dict[str, str]]) -> None:
    tok = _tokenizer()
    ours = tok.encode(rr.RuntimeRenderer().render(msgs), add_special_tokens=False)
    assert ours == tok.encode(render_prompt_text(msgs), add_special_tokens=False)
    try:
        from training.tinker_common import make_renderer

        train_r = make_renderer(tok)
    except Exception:  # noqa: BLE001 - tinker/torch not importable here
        return
    assert ours == train_r.prompt_tokens(msgs)
    hf = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert ours == tok.encode(hf, add_special_tokens=False)


def test_rejects_non_prompt_messages() -> None:
    r = rr.RuntimeRenderer()
    with pytest.raises(ValueError):
        r.render([ChatMessage("user", "a"), ChatMessage("system", "b")])
    with pytest.raises(ValueError):
        r.render([ChatMessage("system", "a"), ChatMessage("assistant", "b")])
    with pytest.raises(ValueError):
        r.render([])


def test_config_validation(tmp_path: Path) -> None:
    p = tmp_path / "renderer.json"
    p.write_text(json.dumps({"renderer": "other"}), encoding="utf-8")
    with pytest.raises(rr.RendererConfigError):
        rr.RuntimeRenderer(p)
    with pytest.raises(rr.RendererConfigError):
        rr.RuntimeRenderer(tmp_path / "missing.json")


def test_eval_can_load_both_renderers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from eval.config import load_renderer

    r = load_renderer("training.renderer_runtime:renderer")
    assert r.render(MESSAGE_LISTS[0]) == render_prompt_text(MESSAGE_LISTS[0])  # type: ignore[arg-type]
    fs = tmp_path / "few_shot.json"
    ex = rr.select_few_shot(ITEMS)
    fs.write_text(
        json.dumps({"examples": [e.__dict__ for e in ex], "sha256": rr._examples_sha(ex)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(rr, "FEW_SHOT_JSON", fs)
    c = load_renderer("training.renderer_runtime:few_shot_renderer")
    assert isinstance(c, rr.FewShotRenderer)


# --------------------------------------------------------------------------- few-shot (C)


def test_select_few_shot_is_deterministic_and_grounded() -> None:
    a = rr.select_few_shot(ITEMS)
    b = rr.select_few_shot(list(reversed(ITEMS)))
    assert [e.item_id for e in a] == [e.item_id for e in b]
    assert len(a) == 2 and len({e.item_id.split("-")[0] for e in a}) == 2
    for e in a:
        out = json.loads(e.assistant)
        assert set(out) == {"facts", "mission_result", "mission_span", "entry"}
        assert (
            e.user
            == writer_messages(next(i for i in ITEMS if i["item_id"] == e.item_id))[-1].content
        )


def test_select_prefers_sft_gold() -> None:
    gold = json.dumps(
        {
            "facts": [_fact("koel was calling near the temple", "koel")],
            "mission_result": "not_mentioned",
            "mission_span": None,
            "entry": "Mom and Dad heard a koel calling near the temple.",
        },
        separators=(",", ":"),
    )
    sft = [
        {
            "meta": {"item_id": "s001-00"},
            "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": gold}],
        }
    ]
    ex = rr.select_few_shot(ITEMS, sft)
    assert next(e for e in ex if e.item_id == "s001-00").assistant == gold


def test_few_shot_render_structure() -> None:
    ex = [rr.FewShotExample("a", "U1 ", " A1"), rr.FewShotExample("b", "U2", "A2")]
    r = rr.FewShotRenderer(examples=ex)
    s = r.render([ChatMessage("system", "S"), ChatMessage("user", "U3")])
    assert s == (
        "<|im_start|>system\nS<|im_end|>\n"
        "<|im_start|>user\nU1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n"
        "<|im_start|>user\nU2<|im_end|>\n<|im_start|>assistant\nA2<|im_end|>\n"
        "<|im_start|>user\nU3<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert r.fingerprint() != rr.RuntimeRenderer().fingerprint()


def test_few_shot_matches_hf_template_history() -> None:
    tok = _tokenizer()
    ex = rr.select_few_shot(ITEMS)
    msgs = MESSAGE_LISTS[2]
    r = rr.FewShotRenderer(examples=ex)
    hf_msgs = [msgs[0]]
    for e in ex:
        hf_msgs += [
            {"role": "user", "content": e.user},
            {"role": "assistant", "content": e.assistant},
        ]
    hf_msgs += msgs[1:]
    hf = tok.apply_chat_template(
        hf_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert r.render(msgs) == hf


def test_few_shot_file_roundtrip_and_tamper(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text("".join(json.dumps(i) + "\n" for i in ITEMS), encoding="utf-8")
    out = tmp_path / "few_shot.json"
    ex = rr.freeze_few_shot(train, tmp_path / "none.jsonl", out)
    assert [e.item_id for e in rr.load_few_shot(out)] == [e.item_id for e in ex]
    data = json.loads(out.read_text(encoding="utf-8"))
    data["examples"][0]["assistant"] += " "
    out.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(rr.RendererConfigError, match="sha256"):
        rr.load_few_shot(out)
