SHELL := /bin/bash
.PHONY: setup models install-agents up down test canary visual eval eval-live forget lint typecheck

setup:
	uv sync --all-extras
	@if [ -d web ] && [ -f web/package-lock.json ]; then cd web && npm ci; else echo "web/ not present yet; skipping npm ci"; fi
	git config core.hooksPath .githooks

models:
	bash scripts/models.sh

install-agents:
	@echo "TODO(P2): launchd plists live in ops/launchd/ (not created yet)."; exit 1

up:
	bash scripts/up.sh

down:
	bash scripts/down.sh

test:
	uv run pytest tests/unit tests/integration tests/privacy -m "not gpu" || [ $$? -eq 5 ]

canary:
	@echo "TODO(P8): privacy canary not implemented yet."; exit 1

visual:
	cd web && npx playwright test

eval:
	@test -f eval/run.py || { echo "eval/run.py does not exist yet (P7)."; exit 1; }
	uv run python eval/run.py --cached

eval-live:
	@test -f eval/run.py || { echo "eval/run.py does not exist yet (P7)."; exit 1; }
	uv run python eval/run.py --live

forget:
	bash scripts/forget.sh

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy --strict hub
