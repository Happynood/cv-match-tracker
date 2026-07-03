.PHONY: sync verify lint format typecheck test smoke run eval calibrate clean

sync:
	uv sync --all-extras

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

typecheck:
	uv run pyright

test:
	uv run pytest -q -m "not smoke"

smoke:
	uv run pytest -q -m smoke

verify: lint format typecheck test smoke
	@echo "verify: OK"

run:
	uv run matchtracker run --config configs/config.yaml

eval:
	uv run matchtracker eval --config configs/config.yaml

calibrate:
	uv run matchtracker calibrate --config configs/config.yaml

clean:
	rm -rf results/ .pytest_cache .ruff_cache .mypy_cache dist build
