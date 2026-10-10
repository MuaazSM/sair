"""training/train.py + freeze_lexicon.py + the renderer (no network). Synthetic data only."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from hub.grounding import lexicon_hash
from training import freeze_lexicon, train
from training.tinker_common import (
    IM_END,
    as_dicts,
    render_prompt_text,
    renderer_config,
    writer_messages,
)

ITEM = {
    "item_id": "seed01-00",
    "seed_id": "seed01",
    "group": "Mom and Dad",
    "mission_text": "Notice the gulmohar.",
    "transcript": "Aaj walk pe koel was calling near the temple. Mission done, "
    "gulmohar pe phool aa gaye.",
    "gold": {
        "facts": [
            {
                "span": "koel was calling near the temple",
                "kind": "sound",
                "subject": "koel",
                "certainty": "sure",
                "place": "near the temple",
            }
        ],
        "mission_result": "done",
        "mission_span": "gulmohar pe phool aa gaye",
    },
}
GOLD = json.dumps(
    {
        "facts": ITEM["gold"]["facts"],  # type: ignore[index]
        "mission_result": "done",
        "mission_span": "gulmohar pe phool aa gaye",
        "entry": "Mom and Dad heard a koel calling near the temple.",
    },
    separators=(",", ":"),
)


def _tokenizer() -> Any:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen/Qwen3.5-4B tokenizer not in the local HF cache")


# --------------------------------------------------------------------------- lexicon freeze


def test_refuses_without_frozen_lexicon(tmp_path: Path) -> None:
    with pytest.raises(train.Refused, match="LEXICON_NOT_FROZEN"):
        train.require_frozen_lexicon(tmp_path / "LEXICON_HASH")


def test_refuses_on_hash_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "LEXICON_HASH"
    p.write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(train.Refused, match="LEXICON_HASH_MISMATCH"):
        train.require_frozen_lexicon(p)


def test_accepts_matching_hash_and_freeze_script(tmp_path: Path) -> None:
    p = tmp_path / "LEXICON_HASH"
    assert freeze_lexicon.main(["--out", str(p)]) == 0
    assert p.read_text(encoding="utf-8").strip() == lexicon_hash()
    assert freeze_lexicon.main(["--out", str(p), "--check"]) == 0
    assert train.require_frozen_lexicon(p) == lexicon_hash()
    p.write_text("deadbeef\n", encoding="utf-8")
    assert freeze_lexicon.main(["--out", str(p), "--check"]) == 1


def test_train_refuses_before_any_api_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "LEXICON_HASH_PATH", tmp_path / "missing")

    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("no API client may be created when the lexicon is not frozen")

    import tinker

    monkeypatch.setattr(tinker, "ServiceClient", boom)
    with pytest.raises(train.Refused):
        train.train(train.TrainConfig(), "test-run")


def test_config_limits() -> None:
    for bad in (train.TrainConfig(epochs=4), train.TrainConfig(max_length=2048),
                train.TrainConfig(lr_mult=0)):  # fmt: skip
        with pytest.raises(train.Refused):
            bad.validate()
    train.TrainConfig().validate()
    assert (train.LORA_RANK, train.BATCH_SIZE, train.MAX_LENGTH) == (32, 32, 1024)


# --------------------------------------------------------------------------- renderer


def test_renderer_config_written(tmp_path: Path) -> None:
    out = tmp_path / "renderer.json"
    tok = _tokenizer()
    cfg = train.write_renderer_config(tok, "Qwen/Qwen3.5-4B", out)
    disk = json.loads(out.read_text(encoding="utf-8"))
    assert disk == cfg
    assert disk["renderer"] == "qwen3_5_no_think" and disk["thinking"] == "off"
    assert disk["generation_prefix"] == "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert disk["stop_strings"] == [IM_END]
    assert disk["stop_token_ids"] == tok.encode(IM_END, add_special_tokens=False)
    assert len(disk["hf_chat_template_sha256"]) == 64


def test_renderer_config_without_tokenizer() -> None:
    cfg = renderer_config(None)
    assert "stop_token_ids" not in cfg
    assert cfg["prompt_example"].endswith("<think>\n\n</think>\n\n")


def test_prompt_matches_hf_chat_template_thinking_off() -> None:
    tok = _tokenizer()
    msgs = as_dicts(writer_messages(ITEM))
    ours = render_prompt_text(msgs)
    hf = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert ours == hf
    full_hf = tok.apply_chat_template(
        [*msgs, {"role": "assistant", "content": GOLD}], tokenize=False, enable_thinking=False
    )
    assert full_hf == ours + GOLD + IM_END + "\n"


def test_assistant_only_loss_mask_with_real_tokenizer() -> None:
    from training.tinker_common import make_renderer

    tok = _tokenizer()
    rend = make_renderer(tok)
    msgs = [*as_dicts(writer_messages(ITEM)), {"role": "assistant", "content": GOLD}]
    tokens, weights = rend.build_supervised_example(msgs)
    toks, w = tokens.tolist(), weights.tolist()
    n_prompt = len(rend.prompt_tokens(msgs[:-1]))
    assert set(w[:n_prompt]) == {0.0} and set(w[n_prompt:]) == {1.0}
    trained = tok.decode([t for t, x in zip(toks, w, strict=True) if x == 1.0])
    assert trained == GOLD + IM_END
    assert "<think>" not in trained and "Transcript:" not in trained
    # whole sequence equals the HF template tokenisation (minus the trailing newline)
    hf_ids = tok.apply_chat_template(msgs, tokenize=True, enable_thinking=False)
    if hasattr(hf_ids, "keys"):
        hf_ids = hf_ids["input_ids"]
    assert toks == list(hf_ids)[: len(toks)]
    datum = train.to_datum(train.Example("x", toks, w))
    assert len(datum.loss_fn_inputs["weights"].data) == len(toks) - 1
    assert sum(datum.loss_fn_inputs["weights"].data) == sum(w)  # first token is prompt (w=0)


def test_overlong_examples_dropped_not_truncated() -> None:
    from training.tinker_common import make_renderer

    rend = make_renderer(_tokenizer())
    row = {"messages": [*as_dicts(writer_messages(ITEM)), {"role": "assistant", "content": GOLD}],
           "meta": {"item_id": "seed01-00"}}  # fmt: skip
    kept, dropped = train.build_examples([row], rend, 1024)
    assert (len(kept), dropped) == (1, 0)
    kept, dropped = train.build_examples([row], rend, 50)
    assert (len(kept), dropped) == (0, 1)


def test_resolve_lr_qwen35() -> None:
    _tokenizer()  # skip unless the local HF cache has the model files
    lr, source = train.resolve_lr("Qwen/Qwen3.5-4B")
    assert lr == pytest.approx(5e-4 * (2000 / 2560) ** 0.0775, rel=1e-6)
    assert "get_lr" in source


# --------------------------------------------------------------------------- dev scoring


def test_score_dev_metrics() -> None:
    bad_detail = json.loads(GOLD)
    bad_detail["entry"] = "Mom and Dad heard a koel near the temple and saw three crows."
    raws = [GOLD, "not json at all", json.dumps(bad_detail)]
    m = train.score_dev([ITEM, ITEM, ITEM], raws)
    r = m.rates()
    assert r["valid_json_no_grammar"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["unsupported_rate"] == pytest.approx(0.5)
    assert r["fallback_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["fact_recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m.reasons.get("invalid_json") == 1


def test_plan_has_all_stages() -> None:
    p = train.plan(
        train.TrainConfig(), n_train=1040, mean_tokens=750, n_dev=130, mean_dev_prompt=520
    )
    assert set(p) >= {"smoke_100x1", "sweep_3x(100x1)", "full_3_epochs", "total_usd"}
    assert p["full_3_epochs"]["train_tokens"] == 1040 * 750 * 3
    assert p["steps_per_epoch_full"] == 33


def test_teacher_hf_renderer_matches_qwen_renderer() -> None:
    from training.teacher import teacher_prompt_fn

    tok = _tokenizer()
    msgs = as_dicts(writer_messages(ITEM))
    ours, ours_stop = teacher_prompt_fn(tok, "Qwen/Qwen3.5-397B-A17B")
    hf, hf_stop = teacher_prompt_fn(tok, "Qwen/Qwen3.5-397B-A17B", "hf")
    assert ours(msgs) == hf(msgs)
    assert set(ours_stop) <= set(hf_stop)
