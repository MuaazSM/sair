# Sair (सैर)

A screenless evening-walk companion for two parents. Before the walk, Sair speaks one small *noticing* mission at the door, twice. The phone stays home. Afterwards, a voice note of up to a minute becomes a short family diary entry, and the next mission builds only on what was actually said.

> "On Tuesday you heard a koel near the temple. Is it there again?"

Everything runs on a MacBook at home. No inference, and no content or timing of any real walk, leaves the house.

## Why it's built this way

Hallucinations compound. If a diary invents a bird, tomorrow's mission sends two people looking for a bird that was never there. Sair keeps the diary faithful with three pieces:

- a small open model fine-tuned to write grounded entries (Qwen3.5-4B + LoRA, served by `llama-server` with a JSON grammar);
- a grounding check in code: every fact must quote the transcript exactly, hedges must survive, and any concrete detail with no verified fact behind it fails the entry, which falls back to a template built from verified facts;
- a memory made only of verified facts. The mission planner never sees entry prose.

## How it runs

| Part | What |
|---|---|
| Transcription, mission planning | Gemma 4 E2B via Ollama, temperature 0 for transcripts |
| Diary writer | Sair-Writer (Qwen3.5-4B LoRA, Q4_K_M) on `llama-server`; Gemma fallback |
| Speech | Kokoro-82M in its own local process |
| Hub | FastAPI + LangGraph (no persistent checkpointer) + SQLite (WAL), bound to `127.0.0.1` |
| Phones | A React PWA (Android) and Apple Shortcuts (iPhone), reaching the hub only through `tailscale serve` |
| Owner pings | self-hosted ntfy, metadata only |

## Setup

```bash
make setup           # uv sync, npm ci in web/
make models          # pre-fetch every model and voice (nothing downloads at runtime)
make test            # unit, integration and privacy tests with stubbed models
make up              # start ollama, llama-server, ntfy and the hub on loopback
make install-agents  # optional: launchd agents, incl. keep-awake
```

Copy `.env.example` to `.env` and fill in the device tokens. iPhone Shortcuts are described in [`shortcuts/README.md`](shortcuts/README.md), and the launchd agents in [`ops/README.md`](ops/README.md).

## Privacy model

- **Loopback only.** The hub refuses to bind anything but `127.0.0.1`; phones reach it through `tailscale serve`, Funnel off.
- **Prod outbound = loopback model ports only.** Models are pre-fetched; `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
- **No telemetry from real walks.** Sentry tracing exists only for dev and eval runs on synthetic data, with allowlisted attributes and scrubbers. Prod records latency and counts to local SQLite.
- **No text in any sink.** Transcripts, entries, missions and facts never reach logs, notifications, telemetry or exception messages. A canary test pushes a marker through every input path and fails if it shows up anywhere it shouldn't.
- **Raw audio is ephemeral.** It's deleted right after transcription, or on a permanent failure; transient failures retry for up to 10 minutes, then the audio is deleted.
- **No geolocation, no streaks, no gamification.**
- **Training was not fully local.** Synthetic transcripts went to Tinker for teacher sampling and LoRA training. No family voice note, transcript or entry was ever part of training.

## Credits

The hub topology, exact-substring grounding checks, the Sentry scrubber, the privacy canary and the GGUF export path are patterns from [Rakshak](https://github.com/MuaazSM/rakshak).

## Commits after the deadline

None yet. Any commit made after the challenge deadline (11 Oct 2026, 11:59 PM PDT) will be listed here.
