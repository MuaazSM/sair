"""Runtime prompt renderer: the raw ``/completion`` string, exactly as training rendered it.

PRD §12.4: serve "with the same prompt renderer as training via raw /completion". The
format is loaded from ``training/renderer.json`` (written by ``training/train.py``), so the
serving string cannot drift from the training string without the config changing too:

    <|im_start|>system\\n{system.strip()}<|im_end|>\\n
    <|im_start|>user\\n{user.strip()}<|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

Two renderers, both usable as

* a hub :class:`hub.models.llama_server.Renderer` (``renderer(messages) -> str``), and
* an eval ``PromptRenderer`` (``renderer.render(messages) -> str``; ``--renderer module:attr``,
  eval instantiates a class with no arguments):

``training.renderer_runtime:renderer``           Sair-Writer (system D), zero-shot.
``training.renderer_runtime:few_shot_renderer``  Qwen3.5-4B base few-shot (system C).

Few-shot (C): two FIXED synthetic examples from the TRAIN split are inserted as earlier
user/assistant turns between the system message and the real user message. Historical
assistant turns are rendered the way the Qwen3.5 chat template renders them with thinking
off (``<|im_start|>assistant\\n{content}<|im_end|>\\n``, no think block; verified against
``apply_chat_template`` in the unit test). The examples are frozen into
``training/data/few_shot.json`` by ``freeze-few-shot`` and carry a sha256 fingerprint; put it
into the eval backend id (``EVAL_QWEN_BASE_ID=...-fs<sha8>``) so cached C outputs can never be
replayed for a different few-shot block.

    uv run python training/renderer_runtime.py freeze-few-shot     # writes few_shot.json
    uv run python training/renderer_runtime.py fingerprint         # renderer + few-shot sha

Nothing here prints message content; the CLI prints item ids, counts and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hub.prompts.builders import ChatMessage

ROOT = Path(__file__).resolve().parent.parent
RENDERER_JSON = ROOT / "training" / "renderer.json"
DATA_DIR = ROOT / "training" / "data"
FEW_SHOT_JSON = DATA_DIR / "few_shot.json"
EXPECTED_RENDERER = "qwen3_5_no_think"
N_FEW_SHOT = 2
MAX_EXAMPLE_WORDS = 70

__all__ = [
    "FewShotExample",
    "FewShotRenderer",
    "RendererConfigError",
    "RuntimeRenderer",
    "few_shot_renderer",
    "freeze_few_shot",
    "load_config",
    "renderer",
    "select_few_shot",
]


class RendererConfigError(ValueError):
    """Fixed-message renderer configuration error (no content)."""


def load_config(path: Path = RENDERER_JSON) -> dict[str, Any]:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RendererConfigError("renderer.json missing or unreadable") from exc
    if not isinstance(cfg, dict) or cfg.get("renderer") != EXPECTED_RENDERER:
        raise RendererConfigError("renderer.json is not the qwen3_5_no_think renderer")
    tmpl = cfg.get("message_template")
    if (
        not isinstance(tmpl, str)
        or tmpl.count("{role}") != 1
        or tmpl.count("{content_stripped}") != 1
        or not isinstance(cfg.get("generation_prefix"), str)
        or not isinstance(cfg.get("assistant_suffix"), str)
    ):
        raise RendererConfigError("renderer.json lacks message_template/generation_prefix")
    if cfg.get("bos"):
        raise RendererConfigError("renderer.json: a BOS string is not supported")
    return cfg


def _role_of(m: ChatMessage | dict[str, str]) -> tuple[str, str]:
    if isinstance(m, dict):
        return m["role"], m["content"]
    return m.role, m.content


class RuntimeRenderer:
    """Zero-shot renderer (system D, production writer). Same rules as the training render:
    an optional system message first, then user messages; no assistant turns."""

    def __init__(self, config_path: Path = RENDERER_JSON) -> None:
        cfg = load_config(config_path)
        self.config = cfg
        self._tmpl: str = cfg["message_template"]
        self.generation_prefix: str = cfg["generation_prefix"]
        self.assistant_suffix: str = cfg["assistant_suffix"]
        self.stop_strings: list[str] = list(cfg.get("stop_strings") or [self.assistant_suffix])

    # the "{role}" / "{content_stripped}" slots are filled with replace(), never format():
    # message text containing braces must not be interpreted
    def _msg(self, role: str, content: str) -> str:
        head, _, tail = self._tmpl.partition("{content_stripped}")
        return head.replace("{role}", role) + content.strip() + tail.replace("{role}", role)

    def _history_assistant(self, content: str) -> str:
        # earlier assistant turns: no think block (Qwen3.5 template, thinking off)
        return self._msg("assistant", content)

    def _prompt_parts(
        self, messages: Sequence[ChatMessage | dict[str, str]]
    ) -> list[tuple[str, str]]:
        parts = [_role_of(m) for m in messages]
        for i, (role, _) in enumerate(parts):
            if role == "system" and i != 0:
                raise ValueError("system message must be first")
            if role not in ("system", "user"):
                raise ValueError("prompt messages must be system/user")
        if not parts or parts[-1][0] != "user":
            raise ValueError("the last prompt message must be the user turn")
        return parts

    def render(self, messages: Sequence[ChatMessage | dict[str, str]]) -> str:
        out = [self._msg(r, c) for r, c in self._prompt_parts(messages)]
        out.append(self.generation_prefix)
        return "".join(out)

    def __call__(self, messages: Sequence[ChatMessage | dict[str, str]]) -> str:
        return self.render(messages)

    def fingerprint(self) -> str:
        blob = json.dumps(
            {"t": self._tmpl, "g": self.generation_prefix, "s": self.assistant_suffix},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FewShotExample:
    item_id: str
    user: str
    assistant: str


def _examples_sha(examples: Sequence[FewShotExample]) -> str:
    blob = json.dumps(
        [[e.item_id, e.user, e.assistant] for e in examples], ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_few_shot(path: Path | None = None) -> list[FewShotExample]:
    path = path or FEW_SHOT_JSON  # module global read at call time
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RendererConfigError(
            "few_shot.json missing: run training/renderer_runtime.py freeze-few-shot"
        ) from exc
    ex = [
        FewShotExample(str(e["item_id"]), str(e["user"]), str(e["assistant"]))
        for e in data.get("examples", [])
    ]
    if len(ex) != N_FEW_SHOT:
        raise RendererConfigError(f"few_shot.json must hold {N_FEW_SHOT} examples")
    if data.get("sha256") != _examples_sha(ex):
        raise RendererConfigError("few_shot.json sha256 mismatch (edited by hand?)")
    return ex


class FewShotRenderer(RuntimeRenderer):
    """Base few-shot renderer (system C): system, then the fixed examples as earlier
    user/assistant turns, then the real user turn, then the generation prefix."""

    def __init__(
        self,
        config_path: Path = RENDERER_JSON,
        few_shot_path: Path | None = None,
        examples: Sequence[FewShotExample] | None = None,
    ) -> None:
        super().__init__(config_path)
        self.examples = list(examples) if examples is not None else load_few_shot(few_shot_path)
        self.few_shot_sha = _examples_sha(self.examples)

    def render(self, messages: Sequence[ChatMessage | dict[str, str]]) -> str:
        parts = self._prompt_parts(messages)
        out: list[str] = []
        rest = parts
        if parts[0][0] == "system":
            out.append(self._msg(*parts[0]))
            rest = parts[1:]
        for e in self.examples:
            out.append(self._msg("user", e.user))
            out.append(self._history_assistant(e.assistant))
        out.extend(self._msg(r, c) for r, c in rest)
        out.append(self.generation_prefix)
        return "".join(out)

    def fingerprint(self) -> str:
        return hashlib.sha256(
            (super().fingerprint() + self.few_shot_sha).encode("utf-8")
        ).hexdigest()


# eval loads ``module:attr`` and instantiates classes with no arguments
renderer = RuntimeRenderer
few_shot_renderer = FewShotRenderer


# --------------------------------------------------------------------------- few-shot choice


def _example_from_item(item: dict[str, Any], sft_assistant: str | None) -> FewShotExample | None:
    """User text = the production writer prompt's user turn for the item; assistant = the
    SFT gold JSON when present, else the code-built template output from the gold facts.
    Kept only if it passes the §8 grounding check (examples must be faithful)."""
    from hub.grounding import build_template_entry, check_entry
    from hub.schemas import EntryOrigin, Fact, MissionResult, WriterOutput, parse_writer_output
    from training.tinker_common import gold_json, writer_messages

    user = writer_messages(item)[-1].content
    gold = item.get("gold") or {}
    if sft_assistant is not None:
        text, origin = sft_assistant, EntryOrigin.MODEL
    else:
        facts = [Fact.model_validate(f) for f in gold.get("facts") or []]
        mr = MissionResult(gold.get("mission_result", "not_mentioned"))
        out = WriterOutput(
            facts=facts,
            mission_result=mr,
            mission_span=gold.get("mission_span")
            if mr is not MissionResult.NOT_MENTIONED
            else None,
            entry=build_template_entry(str(item["group"]), facts, mr is MissionResult.DONE),
        )
        text, origin = gold_json(out), EntryOrigin.TEMPLATE
    parsed = parse_writer_output(text)
    if parsed.value is None:
        return None
    chk = check_entry(parsed.value, str(item["transcript"]), str(item["group"]), origin=origin)
    if not chk.passed or chk.unsupported_details:
        return None
    return FewShotExample(str(item["item_id"]), user, text)


def select_few_shot(
    train_items: Sequence[dict[str, Any]],
    sft_rows: Sequence[dict[str, Any]] = (),
    n: int = N_FEW_SHOT,
) -> list[FewShotExample]:
    """Deterministic: sorted by item_id; prefer non-hard items with a mission, 1-3 gold facts
    and <= 70 transcript words, from distinct seeds, first one per language mix seen."""
    sft = {}
    for row in sft_rows:
        iid = (row.get("meta") or {}).get("item_id")
        msgs = row.get("messages") or []
        if iid and msgs and msgs[-1].get("role") == "assistant":
            sft[str(iid)] = str(msgs[-1]["content"])

    def ok(it: dict[str, Any]) -> bool:
        nf = len((it.get("gold") or {}).get("facts") or [])
        return (
            bool(it.get("mission_text"))
            and 1 <= nf <= 3
            and not (it.get("meta") or {}).get("hard", False)
            and len(str(it["transcript"]).split()) <= MAX_EXAMPLE_WORDS
        )

    items = sorted(train_items, key=lambda it: str(it["item_id"]))
    chosen: list[FewShotExample] = []
    seeds: set[str] = set()
    mixes: set[str] = set()
    for strict in (True, False):
        for it in items:
            if len(chosen) == n:
                break
            seed = str(it.get("seed_id", it["item_id"]))
            mix = str((it.get("meta") or {}).get("language_mix", "?"))
            if seed in seeds or (strict and (not ok(it) or mix in mixes)):
                continue
            ex = _example_from_item(it, sft.get(str(it["item_id"])))
            if ex is None:
                continue
            chosen.append(ex)
            seeds.add(seed)
            mixes.add(mix)
    if len(chosen) < n:
        raise RendererConfigError(f"could not find {n} grounded train examples from distinct seeds")
    return chosen


def freeze_few_shot(
    train_path: Path = DATA_DIR / "train.jsonl",
    sft_path: Path = DATA_DIR / "sft_train.jsonl",
    out: Path = FEW_SHOT_JSON,
) -> list[FewShotExample]:
    from training.tinker_common import atomic_write_json, read_jsonl

    items = read_jsonl(train_path)
    sft = read_jsonl(sft_path) if sft_path.is_file() else []
    ex = select_few_shot(items, sft)
    atomic_write_json(
        out,
        {
            "version": 1,
            "source": {"train": train_path.name, "sft": sft_path.name if sft else None},
            "examples": [e.__dict__ for e in ex],
            "sha256": _examples_sha(ex),
        },
    )
    return ex


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="runtime renderer tools (no content printed)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze-few-shot")
    f.add_argument("--force", action="store_true")
    sub.add_parser("fingerprint")
    a = ap.parse_args(argv)
    if a.cmd == "freeze-few-shot":
        if FEW_SHOT_JSON.is_file() and not a.force:
            ex = load_few_shot()
            print(
                f"few_shot: kept existing ids={[e.item_id for e in ex]} "
                f"sha256={_examples_sha(ex)} (--force to re-pick)"
            )
            return 0
        ex = freeze_few_shot()
        print(f"few_shot: wrote ids={[e.item_id for e in ex]} sha256={_examples_sha(ex)}")
        return 0
    r = RuntimeRenderer()
    print(f"renderer: {r.config['renderer']} fingerprint={r.fingerprint()}")
    if FEW_SHOT_JSON.is_file():
        fs = FewShotRenderer()
        print(
            f"few_shot: ids={[e.item_id for e in fs.examples]} sha8={fs.few_shot_sha[:8]} "
            f"fingerprint={fs.fingerprint()}"
        )
    else:
        print("few_shot: not frozen yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
