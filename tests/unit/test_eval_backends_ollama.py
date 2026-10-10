"""The eval Ollama adapter must disable thinking, as production does (hub/models/ollama.py)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from eval.backends import OllamaPlanner, OllamaWriter
from hub.prompts.builders import ChatMessage


class _Resp:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"message": {"content": "{}"}, "prompt_eval_count": 1, "eval_count": 1}


@pytest.mark.parametrize("grammar", [True, False])
def test_writer_sends_think_false(monkeypatch: pytest.MonkeyPatch, grammar: bool) -> None:
    seen: dict[str, Any] = {}

    def fake_post(url: str, *, json: dict[str, Any], timeout: float) -> _Resp:
        seen.update(json)
        return _Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    w = OllamaWriter("http://127.0.0.1:11434", "gemma4:e2b")
    assert w.generate([ChatMessage(role="user", content="x")], grammar=grammar) == "{}"
    assert seen["think"] is False
    assert seen["options"]["temperature"] == 0


def test_planner_sends_think_false(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_post(url: str, *, json: dict[str, Any], timeout: float) -> _Resp:
        seen.update(json)
        return _Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    OllamaPlanner("http://127.0.0.1:11434", "gemma4:e2b").plan(
        [ChatMessage(role="user", content="x")]
    )
    assert seen["think"] is False
    assert "format" in seen
