"""Shared helpers for the synthetic-data scripts (synthetic data only; no real family data)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
SEEDS_PATH = ROOT / "data" / "seeds.yaml"
CACHE_DIR = ROOT / "training" / "cache" / "gen"
DATA_DIR = ROOT / "training" / "data"
ITEMS_PATH = DATA_DIR / "items.jsonl"
EVAL_SYNTH_DIR = ROOT / "eval" / "data" / "synthetic"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")

Seed = dict[str, Any]


def load_seeds(path: Path = SEEDS_PATH) -> list[Seed]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    seeds: list[Seed] = list(data["seeds"])
    return seeds


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
