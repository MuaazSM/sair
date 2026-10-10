"""Shared helpers for the Tinker scripts (teacher sampling, training). Synthetic data only.

Nothing here reads real family data or the Tinker key: the key reaches the SDK only through
the environment (``(set -a; . ./.env.training; set +a; uv run python ...)``) and is never
read, printed or logged by our code. Printed output is counts, ids, codes and costs only.

Renderer
--------
``tinker_cookbook`` 0.1.0 (installed) predates Qwen3.5 and has no renderer for it; its
``qwen3_disable_thinking`` renderer also builds SFT examples without the empty think block
(``<think>\\n`` only), which does not match the Qwen3.5 chat template. So this module ships
``Qwen35NoThinkRenderer``, a ``tinker_cookbook.renderers.Renderer`` subclass whose strings are
exactly what the Qwen3.5 ``chat_template.jinja`` emits with ``enable_thinking=False``
(verified against ``tokenizer.apply_chat_template`` in ``tests/unit/test_train_config.py``):

    <|im_start|>system\\n{system.strip()}<|im_end|>\\n
    <|im_start|>user\\n{user.strip()}<|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{assistant}<|im_end|>

The same renderer is used for the teacher (Qwen3.5 family, same template) and the student,
and its config is written to ``training/renderer.json`` for serving (raw ``/completion``).
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hub.prompts.builders import ChatMessage, MissionText, Transcript, WalkerGroup
from hub.prompts.builders import build_writer_prompt as _build_writer_prompt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "training" / "data"
TEACHER_CACHE = ROOT / "training" / "cache" / "teacher"
RUNS_DIR = ROOT / "training" / "runs"
RENDERER_JSON = ROOT / "training" / "renderer.json"
LEXICON_HASH_PATH = ROOT / "training" / "LEXICON_HASH"

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3.5-397B-A17B"

# --------------------------------------------------------------------------- prices
# USD per million tokens, from the Tinker models/pricing page, fetched 2026-10-10:
# https://tinker-docs.thinkingmachines.ai/tinker/models/ (64K-context rows). Cached prefill
# is billed at an 80% discount. For num_samples > 1 the prompt is billed once and the other
# num_samples - 1 prefills are billed as cache hits (tinker SampleResponse docstring).
PRICES_SOURCE = "https://tinker-docs.thinkingmachines.ai/tinker/models/ (fetched 2026-10-10)"
PRICES: dict[str, dict[str, float]] = {
    "Qwen/Qwen3.5-4B": {"prefill": 0.33, "prefill_cached": 0.066, "sample": 1.005, "train": 0.737},
    "Qwen/Qwen3.5-9B": {"prefill": 0.30, "prefill_cached": 0.06, "sample": 0.807, "train": 1.463},
    "Qwen/Qwen3.5-397B-A17B": {
        "prefill": 1.65,
        "prefill_cached": 0.33,
        "sample": 7.50,
        "train": 6.60,
    },
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": {
        "prefill": 3.60,
        "prefill_cached": 0.36,
        "sample": 12.45,
        "train": 10.956,
    },
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": {
        "prefill": 1.14,
        "prefill_cached": 0.114,
        "sample": 2.88,
        "train": 2.552,
    },
}
# Unknown model: assume the most expensive listed rates (conservative), and say so.
FALLBACK_PRICE = {"prefill": 3.60, "prefill_cached": 0.36, "sample": 12.45, "train": 10.956}


def price_for(model: str) -> tuple[dict[str, float], bool]:
    """(rates, known). Unknown models get the conservative fallback rates."""
    base = model.split(":")[0]
    if base in PRICES:
        return PRICES[base], True
    return FALLBACK_PRICE, False


@dataclass
class Usage:
    """Token counters for one model; cost is an estimate from the published rate card."""

    model: str
    requests: int = 0
    prefill_tokens: int = 0
    prefill_cached_tokens: int = 0
    sample_tokens: int = 0
    train_tokens: int = 0

    def add_sample(self, prompt_tokens: int, cached: int, num_samples: int, out: int) -> None:
        self.requests += 1
        cached = max(0, min(cached, prompt_tokens))
        self.prefill_tokens += prompt_tokens - cached
        self.prefill_cached_tokens += cached + prompt_tokens * max(0, num_samples - 1)
        self.sample_tokens += out

    def cost(self) -> float:
        p, _ = price_for(self.model)
        return (
            self.prefill_tokens * p["prefill"]
            + self.prefill_cached_tokens * p["prefill_cached"]
            + self.sample_tokens * p["sample"]
            + self.train_tokens * p["train"]
        ) / 1e6

    def to_dict(self) -> dict[str, Any]:
        p, known = price_for(self.model)
        return {
            "model": self.model,
            "requests": self.requests,
            "prefill_tokens": self.prefill_tokens,
            "prefill_cached_tokens": self.prefill_cached_tokens,
            "sample_tokens": self.sample_tokens,
            "train_tokens": self.train_tokens,
            "rates_usd_per_mtok": p,
            "rates_known": known,
            "rates_source": PRICES_SOURCE if known else "unknown: conservative fallback",
            "estimated_cost_usd": round(self.cost(), 6),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Usage:
        return cls(
            model=str(d["model"]),
            requests=int(d.get("requests", 0)),
            prefill_tokens=int(d.get("prefill_tokens", 0)),
            prefill_cached_tokens=int(d.get("prefill_cached_tokens", 0)),
            sample_tokens=int(d.get("sample_tokens", 0)),
            train_tokens=int(d.get("train_tokens", 0)),
        )


# --------------------------------------------------------------------------- io


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_id(item_id: str) -> str:
    """Item ids become file names; refuse anything that could escape the cache dir."""
    if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,128}", item_id) or item_id.startswith("."):
        raise ValueError("unsafe item id")
    return item_id


# --------------------------------------------------------------------------- prompts


def writer_messages(item: dict[str, Any]) -> list[ChatMessage]:
    """The PRODUCTION writer prompt for one synthetic item (FR-J3), via hub builders."""
    mission = item.get("mission_text")
    return _build_writer_prompt(
        WalkerGroup(str(item["group"])),
        MissionText(str(mission)) if mission else None,
        Transcript(str(item["transcript"])),
    )


def as_dicts(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def gold_json(output: Any) -> str:
    """Compact gold JSON, key order facts, mission_result, mission_span, entry (§9.1)."""
    facts = []
    for f in output.facts:
        d: dict[str, Any] = {
            "span": f.span,
            "kind": f.kind.value,
            "subject": f.subject,
            "certainty": f.certainty.value,
        }
        if f.place is not None and f.place.strip():
            d["place"] = f.place
        facts.append(d)
    obj = {
        "facts": facts,
        "mission_result": output.mission_result.value,
        "mission_span": output.mission_span,
        "entry": output.entry,
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- renderer

RENDERER_NAME = "qwen3_5_no_think"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
THINK_OFF = "<think>\n\n</think>\n\n"


def render_prompt_text(messages: Sequence[dict[str, str]]) -> str:
    """Prompt string up to (and including) the empty think block of the assistant turn."""
    out: list[str] = []
    for i, m in enumerate(messages):
        role, content = m["role"], m["content"]
        if role == "system" and i != 0:
            raise ValueError("system message must be first")
        if role not in ("system", "user"):
            raise ValueError("prompt messages must be system/user")
        out.append(f"{IM_START}{role}\n{content.strip()}{IM_END}\n")
    out.append(f"{IM_START}assistant\n{THINK_OFF}")
    return "".join(out)


def render_completion_text(content: str) -> str:
    """The trained (loss-bearing) assistant text; generation stops at <|im_end|>."""
    return content + IM_END


def renderer_config(
    tokenizer: Any | None = None, tokenizer_id: str = DEFAULT_BASE_MODEL
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "renderer": RENDERER_NAME,
        "tokenizer": tokenizer_id,
        "thinking": "off",
        "bos": None,
        "message_template": IM_START + "{role}\n{content_stripped}" + IM_END + "\n",
        "generation_prefix": IM_START + "assistant\n" + THINK_OFF,
        "assistant_suffix": IM_END,
        "stop_strings": [IM_END],
        "train_on": "last assistant message only (prompt and think block weight 0)",
        "prompt_example": render_prompt_text(
            [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]
        ),
        "hf_equivalent": "tokenizer.apply_chat_template(messages, add_generation_prompt=True, "
        "enable_thinking=False)",
    }
    if tokenizer is not None:
        cfg["stop_token_ids"] = tokenizer.encode(IM_END, add_special_tokens=False)
        tmpl = getattr(tokenizer, "chat_template", None)
        if isinstance(tmpl, str):
            import hashlib

            cfg["hf_chat_template_sha256"] = hashlib.sha256(tmpl.encode("utf-8")).hexdigest()
    return cfg


def make_renderer(tokenizer: Any) -> Any:
    """A tinker_cookbook Renderer for Qwen3.5 with thinking off (see module docstring)."""
    import tinker
    import torch
    from tinker_cookbook.renderers import (  # type: ignore[import-untyped]
        Message,
        Renderer,
        TrainOnWhat,
    )

    class Qwen35NoThinkRenderer(Renderer):  # type: ignore[misc]
        name = RENDERER_NAME

        def prompt_tokens(self, messages: Sequence[dict[str, str]]) -> list[int]:
            ids: list[int] = self.tokenizer.encode(
                render_prompt_text(messages), add_special_tokens=False
            )
            return ids

        def build_generation_prompt(
            self, messages: list[Message], role: str = "assistant", prefill: str | None = None
        ) -> tinker.ModelInput:
            if role != "assistant":
                raise ValueError("only assistant generation is supported")
            ids = self.prompt_tokens([dict(m) for m in messages])
            if prefill:
                ids += self.tokenizer.encode(prefill, add_special_tokens=False)
            return tinker.ModelInput.from_ints(ids)

        def build_supervised_example(
            self,
            messages: list[Message],
            train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if train_on_what not in (
                TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            ):
                raise ValueError("assistant-only loss is required (PRD §12.3)")
            if not messages or messages[-1]["role"] != "assistant":
                raise ValueError("last message must be the assistant reply")
            if any(m["role"] == "assistant" for m in messages[:-1]):
                raise ValueError("single-turn examples only")
            prompt = self.prompt_tokens([dict(m) for m in messages[:-1]])
            completion = self.tokenizer.encode(
                render_completion_text(messages[-1]["content"]), add_special_tokens=False
            )
            tokens = torch.tensor(prompt + completion, dtype=torch.int64)
            weights = torch.tensor([0.0] * len(prompt) + [1.0] * len(completion))
            return tokens, weights

        def get_stop_sequences(self) -> list[int]:
            ids: list[int] = self.tokenizer.encode(IM_END, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("expected a single <|im_end|> token")
            return ids

        def parse_response(self, response: list[int]) -> tuple[Message, bool]:
            from tinker_cookbook.renderers import (
                parse_response_for_stop_token,
            )

            msg, ok = parse_response_for_stop_token(
                response, self.tokenizer, self.get_stop_sequences()[0]
            )
            return msg, ok

    return Qwen35NoThinkRenderer(tokenizer)


def load_tokenizer(model: str, *, offline_first: bool = True) -> Any:
    """HF tokenizer; tries the local cache first, then (training machine only) the Hub."""
    from transformers import AutoTokenizer

    name = model.split(":")[0]
    if offline_first:
        old = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            return AutoTokenizer.from_pretrained(name, use_fast=True)
        except Exception:  # noqa: BLE001 - fall through to the Hub
            pass
        finally:
            if old is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old
    return AutoTokenizer.from_pretrained(name, use_fast=True)


# --------------------------------------------------------------------------- text stats

_DEVA = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
# Common romanised-Hindi function words (for the Hinglish script check; not exhaustive).
# English collisions (the, me, main, do, na) are left out on purpose.
HINDI_ROMAN = frozenset(
    " ".join(
        (
            "hai hain tha thi  ho hoga hogi ka ki ke ko se  mein mai  par pe bhi aur ya",
            "nahi nahin  kya kyun kaise kitna kitne bahut bohot kaafi thoda thodi aaj kal abhi",
            "phir wahan yahan woh wo yeh ye hum humne maine mujhe tum aap apna apni raha rahi rahe",
            "gaya gayi gaye dekha dekhi suna laga lagta lagti shayad jaise jaisa kuch koi sab ek",
            " teen accha achha acha haan ji wala wali wale tak liye",
        )
    ).split()
)


def script_stats(text: str) -> dict[str, float]:
    """Letter-script ratios and the share of romanised-Hindi tokens among Latin tokens."""
    nfc = unicodedata.normalize("NFC", text)
    deva = len(_DEVA.findall(nfc))
    latin = len(_LATIN.findall(nfc))
    letters = deva + latin
    toks = [t.casefold() for t in re.findall(r"[A-Za-z]+", nfc)]
    hindi = sum(1 for t in toks if t in HINDI_ROMAN)
    return {
        "deva_ratio": deva / letters if letters else 0.0,
        "latin_ratio": latin / letters if letters else 0.0,
        "roman_hindi_ratio": hindi / len(toks) if toks else 0.0,
    }


def word_count(text: str) -> int:
    return len(text.split())
