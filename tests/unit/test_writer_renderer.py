"""PRD §12.4: with WRITER_BACKEND=llama the hub renders the writer prompt with the training
renderer (training/renderer.json), and refuses to start when that file is missing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import respx

from hub.api.deps import WriterRendererError, build_deps
from hub.graphs import journal
from hub.models.llama_server import render_qwen_chat
from hub.prompts.builders import MissionText, Transcript, WalkerGroup, build_writer_prompt
from tests.unit._graphfix import GOOD_ENTRY, RecNotifier, StubOllama, StubTts, make_settings
from training.renderer_runtime import RENDERER_JSON, RuntimeRenderer

GROUP = "Mom and Dad"
TRANSCRIPT = "we heard a crow near the gate"
MISSION = "Listen for one bird you can name."


def _deps(tmp: Path, **kw: Any) -> Any:
    s = make_settings(tmp, writer_backend="llama", **kw)
    return build_deps(
        s,
        ollama=StubOllama(),
        tts=StubTts(),
        notifier=RecNotifier(),
        run_workers=False,
        plan_on_startup=False,
    )


def _state() -> Any:
    return {
        "walk_id": "w1",
        "group": GROUP,
        "mission_text": MISSION,
        "transcript": TRANSCRIPT,
    }


def _expected(cfg: Path) -> str:
    msgs = build_writer_prompt(
        WalkerGroup(GROUP), MissionText(MISSION), Transcript(TRANSCRIPT), regen=()
    )
    return RuntimeRenderer(cfg).render(msgs)


@respx.mock
def test_llama_backend_sends_training_renderer_prompt(tmp_path: Path) -> None:
    deps = _deps(tmp_path)
    route = respx.post(f"{deps.settings.writer_url}/completion").respond(
        json={"content": GOOD_ENTRY}
    )
    try:
        upd = journal._call_writer(_state(), deps.journal, ())
    finally:
        deps.close()
    assert upd["writer_error"] is None
    sent = json.loads(route.calls.last.request.content)["prompt"]
    assert sent == _expected(RENDERER_JSON)
    msgs = build_writer_prompt(
        WalkerGroup(GROUP), MissionText(MISSION), Transcript(TRANSCRIPT), regen=()
    )
    assert sent != render_qwen_chat(msgs)


@respx.mock
def test_renderer_path_setting_is_honoured(tmp_path: Path) -> None:
    cfg = json.loads(RENDERER_JSON.read_text("utf-8"))
    cfg["message_template"] = "<<{role}>>{content_stripped}<</>>\n"
    alt = tmp_path / "renderer.json"
    alt.write_text(json.dumps(cfg), encoding="utf-8")
    deps = _deps(tmp_path, writer_renderer_path=alt)
    route = respx.post(f"{deps.settings.writer_url}/completion").respond(
        json={"content": GOOD_ENTRY}
    )
    try:
        journal._call_writer(_state(), deps.journal, ())
    finally:
        deps.close()
    sent = json.loads(route.calls.last.request.content)["prompt"]
    assert sent == _expected(alt) and sent.startswith("<<system>>")


def test_missing_renderer_file_refuses_start_with_code_only(tmp_path: Path) -> None:
    with pytest.raises(WriterRendererError) as ei:
        _deps(tmp_path, writer_renderer_path=tmp_path / "nope.json")
    assert str(ei.value) == "writer_renderer_unavailable"
    assert "nope" not in str(ei.value) and str(tmp_path) not in str(ei.value)


def test_gemma_backend_does_not_need_renderer_file(tmp_path: Path) -> None:
    s = make_settings(tmp_path, writer_renderer_path=tmp_path / "nope.json")
    deps = build_deps(
        s, ollama=StubOllama(), tts=StubTts(), notifier=RecNotifier(), run_workers=False
    )
    deps.close()
