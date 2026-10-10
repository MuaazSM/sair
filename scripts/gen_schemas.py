"""Generate schemas/entry.schema.json and schemas/mission.schema.json from hub.schemas.

Definitions are inlined (no ``$ref``/``$defs``) so the entry schema can be passed directly
to Ollama's ``format`` field. Output is deterministic: keys sorted, 2-space indent,
trailing newline. The one exception: the children of ``properties`` keep model field
order (facts -> mission_result -> mission_span -> entry), because grammar-constrained
decoding emits properties in schema order and the entry must come after the facts.

``schemas/mission.schema.json`` is the *planner* schema (the Ollama ``format`` for the
mission planner), derived from ``MissionPlan`` with two restrictions that mirror
``grammars/mission.gbnf``: ``type`` excludes ``seed`` (seed missions are hand-written,
never planned; FR-M2) and ``references`` is required. ``MissionPlan`` itself keeps
``MissionType.SEED`` and the ``references=[]`` default, because the same model also
validates stored and hand-written missions.

    uv run python scripts/gen_schemas.py          # write files
    uv run python scripts/gen_schemas.py --check  # exit 1 if files are stale
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hub.schemas import MissionPlan, MissionType, WriterOutput  # noqa: E402


def _planner_only(schema: dict[str, Any]) -> dict[str, Any]:
    """Restrict the MissionPlan schema to what the planner may emit (FR-M2, §9.2)."""
    schema["properties"]["type"]["enum"] = [
        t.value for t in MissionType if t is not MissionType.SEED
    ]
    schema["required"] = list(schema["properties"])
    return schema


class Target(NamedTuple):
    model: type[BaseModel]
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None


TARGETS: dict[str, Target] = {
    "schemas/entry.schema.json": Target(WriterOutput),
    "schemas/mission.schema.json": Target(MissionPlan, _planner_only),
}


def _inline(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            merged = dict(_inline(defs[name], defs))
            for key, value in node.items():
                if key != "$ref":
                    merged[key] = _inline(value, defs)
            return merged
        return {k: _inline(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline(v, defs) for v in node]
    return node


def build(model: type[BaseModel]) -> dict[str, Any]:
    raw = model.model_json_schema(mode="validation")
    inlined: dict[str, Any] = _inline(raw, raw.get("$defs", {}))
    return inlined


def _sorted(node: Any, keep_order: bool = False) -> Any:
    if isinstance(node, dict):
        keys = list(node) if keep_order else sorted(node)
        child_keeps = {k: k == "properties" and not keep_order for k in keys}
        return {k: _sorted(node[k], keep_order=child_keeps[k]) for k in keys}
    if isinstance(node, list):
        return [_sorted(v) for v in node]
    return node


def render(target: Target | type[BaseModel]) -> str:
    if not isinstance(target, Target):
        target = Target(target)
    schema = build(target.model)
    if target.transform is not None:
        schema = target.transform(schema)
    return json.dumps(_sorted(schema), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    check = "--check" in argv
    stale = []
    for rel, target in TARGETS.items():
        path = ROOT / rel
        text = render(target)
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(rel)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {rel}")
    if stale:
        print("stale: " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
