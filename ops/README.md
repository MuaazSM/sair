# ops: launchd agents

Five user LaunchAgents keep the local stack up. All bind to `127.0.0.1`, all use
`RunAtLoad`, `KeepAlive` and `ThrottleInterval 10`, and all write logs to `logs/` in the repo.
Templates live in `ops/launchd/` with `__REPO__`, `__DATA__` (default `~/.sair`) and
`__WRITER__` placeholders that `scripts/install_agents.sh` fills in.

| Label | What it runs |
|---|---|
| `com.sair.ollama` | `ollama serve` on 127.0.0.1:11434 (keep-alive 30m, debug off, HF offline) |
| `com.sair.llama-server` | `llama-server` on 127.0.0.1:8091 with the writer GGUF, `--log-disable` |
| `com.sair.ntfy` | self-hosted `ntfy serve` on 127.0.0.1:8090, cache under `~/.sair/ntfy/`, no upstream |
| `com.sair.hub` | `uvicorn hub.app:app` on 127.0.0.1:8787, `SAIR_ENV=prod` |
| `com.sair.keepawake` | `caffeinate -s`, no system sleep while on mains power (NFR-R2) |

## Install

```bash
make install-agents                        # render, copy to ~/Library/LaunchAgents, bootstrap
bash scripts/install_agents.sh --dry-run   # print the rendered plists, install nothing
bash scripts/install_agents.sh --uninstall
```

## Check

```bash
launchctl print gui/$(id -u)/com.sair.hub
launchctl list | grep com.sair
```

## Kill and restart check (NFR-R2)

```bash
pid=$(launchctl print gui/$(id -u)/com.sair.hub | awk '/^[[:space:]]*pid =/ {print $3}')
kill "$pid"; sleep 12
launchctl print gui/$(id -u)/com.sair.hub | grep -E 'pid =|state ='   # new pid, running
```

The agent must relaunch within about 10 seconds.

## Power

```bash
pmset -g            # sleep should read 0 while on AC
pmset -g assertions # shows the caffeinate PreventSystemSleep assertion
```

## Backups

```bash
tmutil addexclusion ~/.sair
```

## ntfy upstream

Off by default. If `NTFY_IOS_UPSTREAM=on` is used, the self-hosted server is configured
with an upstream base URL so iOS can be woken. The server then forwards only a poll
request with the message ID, never the message body. Disclose this when enabling it.
