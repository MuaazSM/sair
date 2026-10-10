"""llama.cpp ``llama-server`` client: raw ``/completion`` with an optional GBNF grammar.

The prompt is rendered by us (not the server's chat endpoint) so inference matches training.
``render_qwen_chat`` is the default :class:`Renderer`; the training renderer can be plugged in
by passing ``renderer=`` (P6: "the same prompt renderer as training").
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from hub.models._http import CallStats, post_json
from hub.models.errors import Permanent
from hub.prompts.builders import ChatMessage
from hub.schemas import ReasonCode
from hub.settings import Settings, get_settings

__all__ = [
    "COMPLETE_TIMEOUT_S",
    "LlamaServerClient",
    "Renderer",
    "render_qwen_chat",
]

COMPLETE_TIMEOUT_S = 120.0  # proposed setting: WRITER_TIMEOUT_S
HEALTH_TIMEOUT_S = 3.0
SEED = 0


class Renderer(Protocol):
    def __call__(self, messages: Sequence[ChatMessage]) -> str: ...


def render_qwen_chat(messages: Sequence[ChatMessage], *, thinking_off: bool = True) -> str:
    """Qwen ChatML string ending with an open assistant turn.

    With ``thinking_off`` the template's empty think block is prefilled
    (``<think>\\n\\n</think>\\n\\n``), the P0-proven way to disable thinking.
    """
    parts = [f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n" for m in messages]
    parts.append("<|im_start|>assistant\n")
    parts.append("<think>\n\n</think>\n\n" if thinking_off else "<think>\n")
    return "".join(parts)


class LlamaServerClient:
    def __init__(
        self, settings: Settings | None = None, client: httpx.Client | None = None
    ) -> None:
        self._s = settings or get_settings()
        self._client = client or httpx.Client()

    def complete(
        self, prompt: str, *, grammar: str | None = None, max_tokens: int | None = None
    ) -> tuple[str, CallStats]:
        body: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_tokens if max_tokens is not None else self._s.writer_max_tokens,
            "temperature": 0,
            "seed": SEED,
            "cache_prompt": True,
            "stream": False,
        }
        if grammar:
            body["grammar"] = grammar
        t0 = time.perf_counter()
        j = post_json(
            self._client, f"{self._s.writer_url}/completion", body,
            timeout=COMPLETE_TIMEOUT_S,
            transient=ReasonCode.WRITER_TRANSIENT,
            permanent=ReasonCode.WRITER_PERMANENT,
            component="llama_server",
        )  # fmt: skip
        content = j.get("content")
        if not isinstance(content, str):
            raise Permanent(ReasonCode.WRITER_PERMANENT, "llama_server")
        tm = j.get("timings")
        tm = tm if isinstance(tm, dict) else {}
        pn, gn = tm.get("prompt_n"), tm.get("predicted_n")
        stats = CallStats(
            latency_s=time.perf_counter() - t0,
            prompt_tokens=pn if isinstance(pn, int) else None,
            completion_tokens=gn if isinstance(gn, int) else None,
            hit_limit=j.get("stop_type") == "limit",
        )
        return content, stats

    def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        grammar: str | None = None,
        max_tokens: int | None = None,
        renderer: Renderer = render_qwen_chat,
    ) -> tuple[str, CallStats]:
        return self.complete(renderer(messages), grammar=grammar, max_tokens=max_tokens)

    def health(self) -> bool:
        try:
            return self._client.get(
                f"{self._s.writer_url}/health", timeout=HEALTH_TIMEOUT_S
            ).is_success
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._client.close()
