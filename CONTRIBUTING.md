# Contributing

Thanks for considering a contribution to matchtracker.

## Setup

```bash
uv sync --all-extras
uv run pytest -q
```

## Workflow

1. Fork and branch from `main`.
2. Keep changes focused; one logical change per PR.
3. Run `make verify` before opening a PR — it must be green.
4. Follow the existing code style (`ruff format`, `ruff check`).
5. Add or update tests for any behavior change.
6. Update `CHANGELOG.md` under `Unreleased`.

## Design principle

This project wraps mature open-source components (RF-DETR, supervision's
ByteTrack, roboflow/sports team classifier and homography tools). New stages
should follow the same pattern: a thin adapter over an existing library,
configured through Hydra, rather than a reimplementation. Geometry and
statistics that are genuinely project-specific live in `metrics.py`.

## Reporting bugs

Open a GitHub issue with: input video properties (resolution/fps), the config
used, the command run, and the full traceback.
