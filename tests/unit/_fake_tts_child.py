"""Stand-in for hub.models._tts_child: same JSON-lines protocol, writes 1 s of tone."""

import json
import os
import sys

import numpy as np
import soundfile as sf

pid_file = os.environ.get("FAKE_TTS_PIDFILE")
if pid_file:
    with open(pid_file, "a") as f:
        f.write(f"{os.getpid()}\n")
print("noise line on stdout that is not json", flush=True)
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req.get("cmd") == "quit":
        break
    if os.environ.get("FAKE_TTS_FAIL"):
        print(json.dumps({"ok": False, "code": "render_failed"}), flush=True)
        continue
    if os.environ.get("FAKE_TTS_DIE"):
        sys.exit(1)
    t = np.arange(24000) / 24000
    sf.write(req["out"], (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 24000)
    with open(os.environ["FAKE_TTS_LOG"], "a") as f:
        f.write(
            json.dumps({"voice": req["voice"], "lang": req["lang"], "text": req["text"]}) + "\n"
        )
    print(json.dumps({"ok": True}), flush=True)
