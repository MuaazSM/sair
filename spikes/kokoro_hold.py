"""Hold a loaded Kokoro model in its own process (memory spike). Renders once, then sleeps."""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import tts  # noqa: E402

pcm = tts.render(tts.MISSION_EN, "hf_alpha", "a")
print(f"kokoro_hold: rendered {len(pcm) / tts.SR:.2f}s, sleeping", flush=True)
time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 600)
