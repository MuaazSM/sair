"""Kokoro worker process. Speaks JSON lines on stdin/stdout; spawned lazily by ``hub.models.tts``.

Protocol (one JSON object per line):
  request  {"text": str, "voice": str, "lang": "a"|"b"|"h", "out": "<wav path>"}  or {"cmd": "quit"}
  response {"ok": true} | {"ok": false, "code": "<fixed code>"}   (also {"ready": true} at start)

Privacy (NFR-P4): text is read from the pipe and never printed. fd 1 is redirected to fd 2
(which the parent sends to /dev/null) so library chatter can never corrupt or leak via the
protocol channel; protocol lines go through a private dup of the original stdout.
Offline flags are set by the parent in the child env and re-asserted here.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

SR = 24000


def main() -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    warnings.filterwarnings("ignore")

    def reply(obj: dict[str, Any]) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    model_dir = Path(sys.argv[1])
    model: Any = None
    pipes: dict[str, Any] = {}
    reply({"ready": True})
    for line in sys.stdin:
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                return 0
            if model is None:
                from kokoro import KModel

                model = KModel(
                    repo_id="hexgrad/Kokoro-82M",
                    config=str(model_dir / "config.json"),
                    model=str(model_dir / "kokoro-v1_0.pth"),
                ).eval()
            lang = req["lang"]
            if lang not in pipes:
                from kokoro import KPipeline

                pipes[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=model)
            import numpy as np
            import soundfile as sf

            voice = str(model_dir / "voices" / f"{req['voice']}.pt")
            parts = [
                r.audio.numpy()
                for r in pipes[lang](req["text"], voice=voice)
                if r.audio is not None
            ]
            if not parts:
                reply({"ok": False, "code": "empty"})
                continue
            sf.write(req["out"], np.concatenate(parts), SR, subtype="PCM_16")
            reply({"ok": True})
        except Exception:  # noqa: BLE001 - never leak exception text (may echo input)
            reply({"ok": False, "code": "render_failed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
