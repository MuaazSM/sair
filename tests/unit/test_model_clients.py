"""Ollama + llama-server clients (respx mocks): request shapes, error mapping, no text leaks."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
import respx

from hub.models import audio
from hub.models.errors import ModelError, Permanent, Transient
from hub.models.llama_server import LlamaServerClient, render_qwen_chat
from hub.models.ollama import TRANSCRIBE_PROMPT, OllamaClient
from hub.prompts.builders import ChatMessage
from hub.schemas import ReasonCode
from hub.settings import Settings

FX = Path(__file__).parent.parent / "fixtures" / "audio"
MARK = "CANARY-MARKER-7f3a"
OLLAMA = "http://127.0.0.1:11434"
LLAMA = "http://127.0.0.1:8091"


def settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[arg-type]


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    return audio.decode_to_wav16k(FX / "en_20s.m4a", tmp_path)


def chat_reply(text: str = "hello") -> dict[str, object]:
    return {
        "message": {"role": "assistant", "content": text},
        "done_reason": "stop",
        "prompt_eval_count": 111,
        "eval_count": 22,
        "load_duration": 500_000_000,
    }


# ------------------------------------------------------------------ ollama requests


@respx.mock
def test_transcribe_request_shape(wav: Path) -> None:
    route = respx.post(f"{OLLAMA}/api/chat").respond(json=chat_reply("  a transcript \n"))
    c = OllamaClient(settings())
    text, st = c.transcribe_with_stats(wav)
    assert text == "a transcript"
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "gemma4:e2b"
    assert body["think"] is False and body["stream"] is False
    assert body["options"]["temperature"] == 0
    msg = body["messages"][0]
    assert msg["content"] == TRANSCRIBE_PROMPT
    assert base64.b64decode(msg["images"][0]) == wav.read_bytes()
    assert (st.prompt_tokens, st.completion_tokens, st.load_s) == (111, 22, 0.5)
    assert st.latency_s >= 0 and not st.hit_limit


@respx.mock
def test_transcribe_chunked_in_order(wav: Path) -> None:
    replies = iter(["one", " two ", "three"])
    respx.post(f"{OLLAMA}/api/chat").mock(
        side_effect=lambda r: httpx.Response(200, json=chat_reply(next(replies)))
    )
    c = OllamaClient(settings(transcribe_chunk_s=10))
    assert c.transcribe(wav) == "one two three"
    assert len(respx.calls) == 3


@respx.mock
def test_chat_json_writer_and_planner_shapes() -> None:
    route = respx.post(f"{OLLAMA}/api/chat").respond(json=chat_reply('{"a":1}'))
    c = OllamaClient(settings())
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    msgs = [ChatMessage("system", "sys"), ChatMessage("user", "usr")]
    text, st = c.chat_json(msgs, schema)
    assert text == '{"a":1}' and st.completion_tokens == 22
    b = json.loads(route.calls.last.request.content)
    assert b["format"] == schema and b["think"] is False and b["options"]["temperature"] == 0
    assert b["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    c.chat_json(msgs, schema, role="planner")
    assert json.loads(route.calls.last.request.content)["options"]["temperature"] == 0.7
    c.chat_json(msgs, schema, role="planner", temperature=0.2)
    assert json.loads(route.calls.last.request.content)["options"]["temperature"] == 0.2


@respx.mock
def test_chat_json_returns_invalid_json_untouched() -> None:
    respx.post(f"{OLLAMA}/api/chat").respond(json=chat_reply("not json {"))
    text, _ = OllamaClient(settings()).chat_json([ChatMessage("user", "x")], {"type": "object"})
    assert text == "not json {"


@respx.mock
def test_truncated_flag() -> None:
    r = chat_reply("{")
    r["done_reason"] = "length"
    respx.post(f"{OLLAMA}/api/chat").respond(json=r)
    _, st = OllamaClient(settings()).chat_json([ChatMessage("user", "x")], {})
    assert st.hit_limit


@respx.mock
def test_warm_and_health() -> None:
    route = respx.post(f"{OLLAMA}/api/chat").respond(json={"done": True})
    respx.get(f"{OLLAMA}/api/tags").respond(json={"models": []})
    c = OllamaClient(settings())
    assert c.warm() and c.health()
    b = json.loads(route.calls.last.request.content)
    assert b["messages"] == [] and b["keep_alive"] == "30m"


@respx.mock
def test_warm_and_health_never_raise() -> None:
    respx.post(f"{OLLAMA}/api/chat").mock(side_effect=httpx.ConnectError("x"))
    respx.get(f"{OLLAMA}/api/tags").mock(side_effect=httpx.ConnectError("x"))
    c = OllamaClient(settings())
    assert c.warm() is False and c.health() is False


# ------------------------------------------------------------------ error mapping

CASES: list[tuple[str, object, type[ModelError]]] = [
    ("connect", httpx.ConnectError("refused"), Transient),
    ("timeout", httpx.ReadTimeout("slow"), Transient),
    ("500", httpx.Response(500, text=MARK), Transient),
    ("503", httpx.Response(503, text=MARK), Transient),
    ("429", httpx.Response(429, text=MARK), Transient),
    ("400", httpx.Response(400, text=MARK), Permanent),
    ("404", httpx.Response(404, text=MARK), Permanent),
    ("badjson", httpx.Response(200, text=MARK), Permanent),
    ("nocontent", httpx.Response(200, json={"message": {}}), Permanent),
]


@respx.mock
@pytest.mark.parametrize(("name", "effect", "kind"), CASES, ids=[c[0] for c in CASES])
def test_ollama_error_mapping(wav: Path, name: str, effect: object, kind: type[ModelError]) -> None:
    respx.post(f"{OLLAMA}/api/chat").mock(side_effect=effect)
    c = OllamaClient(settings())
    msgs = [ChatMessage("user", MARK)]
    for call, trans, perm in (
        (
            lambda: c.transcribe(wav),
            ReasonCode.TRANSCRIBE_TRANSIENT,
            ReasonCode.TRANSCRIBE_PERMANENT,
        ),
        (lambda: c.chat_json(msgs, {}), ReasonCode.WRITER_TRANSIENT, ReasonCode.WRITER_PERMANENT),
        (
            lambda: c.chat_json(msgs, {}, role="planner"),
            ReasonCode.PLANNER_TRANSIENT,
            ReasonCode.PLANNER_PERMANENT,
        ),
    ):
        with pytest.raises(ModelError) as ei:
            call()
        e = ei.value
        assert type(e) is kind
        assert e.code == (trans if kind is Transient else perm)
        assert MARK not in str(e) and MARK not in repr(e) and e.__cause__ is None


# ------------------------------------------------------------------ llama-server


@respx.mock
def test_complete_request_shape_and_stats() -> None:
    route = respx.post(f"{LLAMA}/completion").respond(
        json={
            "content": '{"x":1}',
            "stop_type": "limit",
            "timings": {"prompt_n": 321, "predicted_n": 400},
        }
    )
    c = LlamaServerClient(settings())
    text, st = c.complete("PROMPT", grammar="root ::= x", max_tokens=400)
    assert text == '{"x":1}'
    b = json.loads(route.calls.last.request.content)
    assert b["prompt"] == "PROMPT" and b["grammar"] == "root ::= x"
    assert b["n_predict"] == 400 and b["temperature"] == 0 and b["cache_prompt"] is True
    assert b["stream"] is False and b["seed"] == 0
    assert (st.prompt_tokens, st.completion_tokens, st.hit_limit) == (321, 400, True)


@respx.mock
def test_complete_without_grammar_omits_key() -> None:
    route = respx.post(f"{LLAMA}/completion").respond(json={"content": "ok"})
    LlamaServerClient(settings()).complete("p", grammar=None, max_tokens=5)
    assert "grammar" not in json.loads(route.calls.last.request.content)


@respx.mock
def test_complete_chat_uses_pluggable_renderer() -> None:
    route = respx.post(f"{LLAMA}/completion").respond(json={"content": "ok"})
    c = LlamaServerClient(settings())
    c.complete_chat(
        [ChatMessage("user", "hi")], max_tokens=5, renderer=lambda m: f"R:{m[0].content}"
    )
    assert json.loads(route.calls.last.request.content)["prompt"] == "R:hi"
    c.complete_chat([ChatMessage("user", "hi")], max_tokens=5)
    assert json.loads(route.calls.last.request.content)["prompt"].startswith("<|im_start|>user\nhi")


def test_render_qwen_chat_matches_probe_template() -> None:
    msgs = [ChatMessage("system", "SYS"), ChatMessage("user", "USR")]
    assert render_qwen_chat(msgs) == (
        "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nUSR<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert render_qwen_chat(msgs, thinking_off=False).endswith("<|im_start|>assistant\n<think>\n")


LLAMA_CASES: list[tuple[str, object, type[ModelError]]] = [
    ("connect", httpx.ConnectError("refused"), Transient),
    ("timeout", httpx.ReadTimeout("slow"), Transient),
    ("loading503", httpx.Response(503, json={"error": {"message": MARK}}), Transient),
    ("500", httpx.Response(500, text=MARK), Transient),
    ("400", httpx.Response(400, text=MARK), Permanent),
    ("nocontent", httpx.Response(200, json={"nope": MARK}), Permanent),
]


@respx.mock
@pytest.mark.parametrize(("name", "effect", "kind"), LLAMA_CASES, ids=[c[0] for c in LLAMA_CASES])
def test_llama_error_mapping(name: str, effect: object, kind: type[ModelError]) -> None:
    respx.post(f"{LLAMA}/completion").mock(side_effect=effect)
    with pytest.raises(ModelError) as ei:
        LlamaServerClient(settings()).complete(MARK, grammar=MARK, max_tokens=5)
    e = ei.value
    assert type(e) is kind
    assert e.code == (
        ReasonCode.WRITER_TRANSIENT if kind is Transient else ReasonCode.WRITER_PERMANENT
    )
    assert MARK not in str(e) and MARK not in repr(e) and e.__cause__ is None


@respx.mock
def test_llama_health() -> None:
    respx.get(f"{LLAMA}/health").respond(json={"status": "ok"})
    assert LlamaServerClient(settings()).health() is True
    respx.get(f"{LLAMA}/health").respond(503)
    assert LlamaServerClient(settings()).health() is False


# ------------------------------------------------------------------ gpu (real servers)


@pytest.mark.gpu
def test_real_transcribe_en_20s(wav: Path) -> None:
    c = OllamaClient(Settings(_env_file=None))  # type: ignore[call-arg]
    if not c.health():
        pytest.skip("ollama not running")
    text, st = c.transcribe_with_stats(wav)
    ref = (FX / "en_20s.txt").read_text(encoding="utf-8")
    words = {w.strip(".,").lower() for w in ref.split()}
    got = {w.strip(".,").lower() for w in text.split()}
    assert len(words & got) / len(words) > 0.8
    assert st.completion_tokens and st.latency_s > 0


@pytest.mark.gpu
def test_real_llama_complete() -> None:
    c = LlamaServerClient(Settings(_env_file=None))  # type: ignore[call-arg]
    if not c.health():
        pytest.skip("llama-server not running")
    grammar = 'root ::= "{\\"ok\\": " ("true" | "false") "}"'
    text, st = c.complete_chat(
        [ChatMessage("user", "Reply with a JSON object.")], grammar=grammar, max_tokens=20
    )
    assert json.loads(text)["ok"] in (True, False)
    assert st.completion_tokens
