"""Freeze the detail lexicon before training (PRD §12.1.6).

    uv run python training/freeze_lexicon.py            # writes training/LEXICON_HASH
    uv run python training/freeze_lexicon.py --check    # exit 1 if the lexicon changed

Writes ``hub.grounding.lexicon_hash()`` (sha256 of ``data/lexicon.yaml`` bytes) to
``training/LEXICON_HASH`` and prints it. ``training/train.py`` refuses to start when the
current hash differs from the frozen one.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hub.grounding import lexicon_hash
from training.tinker_common import LEXICON_HASH_PATH


def frozen_hash(path: Path = LEXICON_HASH_PATH) -> str | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def check_frozen(path: Path = LEXICON_HASH_PATH, current: str | None = None) -> tuple[bool, str]:
    """(ok, code): ok iff a frozen hash exists and equals the current lexicon hash."""
    frozen = frozen_hash(path)
    if frozen is None:
        return False, "LEXICON_NOT_FROZEN"
    if frozen != (current or lexicon_hash()):
        return False, "LEXICON_HASH_MISMATCH"
    return True, "OK"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="freeze data/lexicon.yaml (write its sha256)")
    ap.add_argument("--check", action="store_true", help="only compare; exit 1 on mismatch")
    ap.add_argument("--out", type=Path, default=LEXICON_HASH_PATH)
    a = ap.parse_args(argv)
    current = lexicon_hash()
    if a.check:
        ok, code = check_frozen(a.out, current)
        print(f"{code} current={current} frozen={frozen_hash(a.out)}")
        return 0 if ok else 1
    a.out.write_text(current + "\n", encoding="utf-8")
    print(current)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
