"""Ollama client (gemma4:e2b): transcription, JSON-schema chat, warm-up, health.

Synchronous (graph nodes call it directly or via ``asyncio.to_thread``). Loopback is enforced
by ``hub.settings``; nothing here adds a host.

Transcription API proven in P0: ``POST /api/chat`` with a 16 kHz mono WAV, base64, in the
message's ``images`` array, ``think: false``, ``options.temperature: 0``.

Sampling choices: the writer backend runs at temperature 0 (grounding wants determinism);
the mission planner defaults to 0.7 (variety across days, safety is checked in code and a
fixed fallback exists). Callers may override with ``temperature=``.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import httpx

from hub.models import audio
from hub.models._http import CallStats, post_json
from hub.models.errors import Permanent
from hub.prompts.builders import ChatMessage
from hub.schemas import ReasonCode
from hub.settings import Settings, get_settings

__all__ = [
    "CHAT_TIMEOUT_S",
    "PLANNER_TEMPERATURE",
    "TRANSCRIBE_PROMPT",
    "TRANSCRIBE_TIMEOUT_S",
    "OllamaClient",
    "get_client",
]

# Proposed settings (see report): OLLAMA_TRANSCRIBE_TIMEOUT_S, OLLAMA_CHAT_TIMEOUT_S.
TRANSCRIBE_TIMEOUT_S = 180.0
CHAT_TIMEOUT_S = 120.0
WARM_TIMEOUT_S = 120.0
HEALTH_TIMEOUT_S = 3.0
TRANSCRIBE_NUM_PREDICT = 1024
PLANNER_TEMPERATURE = 0.7
WRITER_TEMPERATURE = 0.0

TRANSCRIBE_PROMPT = (
    "Transcribe this audio verbatim, in the original language and script. "
    "Output only the transcript, with no commentary."
)

Role = Literal["writer", "planner"]


def _stats(j: dict[str, Any], latency: float) -> CallStats:
    def ns(key: str) -> float | None:
        v = j.get(key)
        return v / 1e9 if isinstance(v, (int, float)) else None

    def count(key: str) -> int | None:
        v = j.get(key)
        return v if isinstance(v, int) else None

    return CallStats(
        latency_s=latency,
        prompt_tokens=count("prompt_eval_count"),
        completion_tokens=count("eval_count"),
        load_s=ns("load_duration"),
        hit_limit=j.get("done_reason") == "length",
    )


def join_transcripts(parts: Sequence[str]) -> str:
    """Deterministic join: trim each, drop empties, single space."""
    return " ".join(p.strip() for p in parts if p.strip())


class OllamaClient:
    def __init__(
        self, settings: Settings | None = None, client: httpx.Client | None = None
    ) -> None:
        self._s = settings or get_settings()
        self._client = client or httpx.Client()
        self.last_stats: CallStats | None = None

    # ------------------------------------------------------------------ internals

    def _chat(
        self,
        body: dict[str, Any],
        *,
        timeout: float,
        transient: ReasonCode,
        permanent: ReasonCode,
    ) -> tuple[str, CallStats]:
        body = {"model": self._s.ollama_model, "stream": False, "think": False,
                "keep_alive": self._s.ollama_keep_alive, **body}  # fmt: skip
        t0 = time.perf_counter()
        j = post_json(
            self._client, f"{self._s.ollama_host}/api/chat", body,
            timeout=timeout, transient=transient, permanent=permanent, component="ollama",
        )  # fmt: skip
        msg = j.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, str):
            raise Permanent(permanent, "ollama")
        stats = _stats(j, time.perf_counter() - t0)
        self.last_stats = stats
        return content, stats

    # ------------------------------------------------------------------ public

    def _transcribe_one(self, wav: Path) -> tuple[str, CallStats]:
        b64 = base64.b64encode(wav.read_bytes()).decode("ascii")
        return self._chat(
            {
                "options": {"temperature": 0, "num_predict": TRANSCRIBE_NUM_PREDICT},
                "messages": [{"role": "user", "content": TRANSCRIBE_PROMPT, "images": [b64]}],
            },
            timeout=TRANSCRIBE_TIMEOUT_S,
            transient=ReasonCode.TRANSCRIBE_TRANSIENT,
            permanent=ReasonCode.TRANSCRIBE_PERMANENT,
        )

    def transcribe(self, wav: Path) -> str:
        """Transcribe a 16 kHz mono WAV (chunked and joined if TRANSCRIBE_CHUNK_S > 0)."""
        return self.transcribe_with_stats(wav)[0]

    def transcribe_with_stats(self, wav: Path) -> tuple[str, CallStats]:
        chunk_s = self._s.transcribe_chunk_s
        t0 = time.perf_counter()
        if chunk_s <= 0 or audio.duration_s(wav) <= chunk_s:
            text, st = self._transcribe_one(wav)
            return text.strip(), st
        texts: list[str] = []
        pt = ct = 0
        with audio.chunk_files(wav, chunk_s) as chunks:
            for c in chunks:
                t, st = self._transcribe_one(c)
                texts.append(t)
                pt += st.prompt_tokens or 0
                ct += st.completion_tokens or 0
        return join_transcripts(texts), CallStats(time.perf_counter() - t0, pt, ct)

    def chat_json(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        *,
        role: Role = "writer",
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> tuple[str, CallStats]:
        """Chat with ``format`` = JSON schema. Returns raw text; grounding judges validity."""
        temp = temperature if temperature is not None else (
            WRITER_TEMPERATURE if role == "writer" else PLANNER_TEMPERATURE
        )  # fmt: skip
        options: dict[str, Any] = {"temperature": temp}
        options["num_predict"] = num_predict or self._s.writer_max_tokens
        trans, perm = (
            (ReasonCode.WRITER_TRANSIENT, ReasonCode.WRITER_PERMANENT)
            if role == "writer"
            else (ReasonCode.PLANNER_TRANSIENT, ReasonCode.PLANNER_PERMANENT)
        )
        return self._chat(
            {
                "format": schema,
                "options": options,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
            },
            timeout=CHAT_TIMEOUT_S,
            transient=trans,
            permanent=perm,
        )

    def warm(self) -> bool:
        """FR-M9: load the model into memory (empty chat + keep_alive). Never raises."""
        try:
            self._client.post(
                f"{self._s.ollama_host}/api/chat",
                json={
                    "model": self._s.ollama_model,
                    "messages": [],
                    "stream": False,
                    "keep_alive": self._s.ollama_keep_alive,
                },  # fmt: skip
                timeout=WARM_TIMEOUT_S,
            ).raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    def health(self) -> bool:
        try:
            return self._client.get(
                f"{self._s.ollama_host}/api/tags", timeout=HEALTH_TIMEOUT_S
            ).is_success
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._client.close()


_default: OllamaClient | None = None


def get_client() -> OllamaClient:
    global _default
    if _default is None:
        _default = OllamaClient()
    return _default
