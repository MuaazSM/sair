"""P0 writer spike (PRD 18.2): stock Qwen3.5-4B Q4_K_M on llama-server, GBNF, raw /completion.

    uv run python spikes/writer.py [--grammar spikes/entry.draft.gbnf] [--quiet]

Prompt = Qwen chat template built by hand, thinking OFF via the template's own empty
think block ("<think>\\n\\n</think>\\n\\n" after "<|im_start|>assistant\\n").
Inputs are three synthetic transcripts (generic walks). Prints JSON validity and tokens/s.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

SYSTEM = (
    "You write one short, faithful family diary entry from a walk voice note. "
    "Return only JSON with keys facts, mission_result, mission_span, entry. "
    "facts: 0-6 items {span, kind, subject, certainty, place?}; span must be copied exactly "
    "from the transcript; kind is one of sight|sound|smell|person|animal|plant|weather|other; "
    "certainty is unsure if the walker hedged (maybe, I think, shayad, lagta), else sure. "
    "mission_result is done|partly|not_done|not_mentioned; mission_span is an exact transcript "
    "substring that shows it, or null when not_mentioned. entry: 1-4 English sentences, at most "
    "70 words, mentions the walkers by their group name, adds nothing that is not in the transcript, "
    "keeps hedges, no questions."
)

SAMPLES = [
    (
        "Mom and Dad",
        "Look for a gulmohar that is flowering on your way.",
        "We walked down the lane behind the market. The gulmohar near the corner has started "
        "flowering again, bright red on top. I think we saw a kingfisher by the pond.",
    ),
    (
        "Mom and Dad",
        "Listen for the koel near the temple. Is it there again?",
        "Aaj hum market ke peeche wali gali se gaye. Mandir ke paas koyal bol rahi thi, pichhli baar "
        "jaisi. Shayad talaab ke paas kingfisher dekha, par woh jaldi ud gaya.",
    ),
    (
        "Mom and Dad",
        "Count the parrots you can spot near the bus stop.",
        "पार्क में बहुत बच्चे थे। बस स्टॉप के पास तार पर तीन तोते बैठे थे। गीली मिट्टी की खुशबू आ रही थी।",
    ),
]


def build_prompt(group: str, mission: str, transcript: str) -> str:
    user = f"Group: {group}\nMission: {mission}\nTranscript:\n{transcript}"
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def complete(prompt: str, grammar: str, client: httpx.Client, seed: int = 0) -> dict:
    body = {
        "prompt": prompt,
        "grammar": grammar,
        "n_predict": 400,
        "temperature": 0,
        "seed": seed,
        "cache_prompt": True,
        "stream": False,
    }
    r = client.post(f"{c.WRITER_URL}/completion", json=body, timeout=120)
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", default=str(Path(__file__).with_name("entry.draft.gbnf")))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    g = Path(a.grammar).read_text()
    ok = 0
    with httpx.Client() as cl:
        for i, (grp, mis, tr) in enumerate(SAMPLES, 1):
            t0 = time.perf_counter()
            j = complete(build_prompt(grp, mis, tr), g, cl)
            wall = time.perf_counter() - t0
            tm = j.get("timings", {})
            try:
                obj = json.loads(j["content"])
                valid, nf = True, len(obj["facts"])
                spans_ok = sum(c.norm_text(f["span"]) in c.norm_text(tr) for f in obj["facts"])
            except (json.JSONDecodeError, KeyError) as e:
                valid, nf, spans_ok, obj = False, -1, -1, {"error": type(e).__name__}
            ok += valid
            print(
                f"sample {i}: valid_json={valid} facts={nf} spans_substring={spans_ok}/{nf} "
                f"stop={j.get('stop_type', j.get('stopped_eos'))} truncated={j.get('truncated')} "
                f"prompt_n={tm.get('prompt_n')} gen_n={tm.get('predicted_n')} "
                f"gen_tok_s={tm.get('predicted_per_second', 0):.1f} "
                f"prompt_tok_s={tm.get('prompt_per_second', 0):.1f} wall={wall:.2f}s"
            )
            if not a.quiet:
                print("   ", json.dumps(obj, ensure_ascii=False))
    print(f"valid {ok}/{len(SAMPLES)}")


if __name__ == "__main__":
    main()
