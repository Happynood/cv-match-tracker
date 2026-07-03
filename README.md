# matchtracker

Offline player tracking and post-match statistics for football, from a single **fixed**
tactical camera. Detect → track → assign team → project onto a 2D pitch → compute
geometry-based statistics (distance, sprints, speed, heatmaps, formation) — batch/offline,
tuned to run on a 4GB-VRAM laptop GPU.

**Try it:** [huggingface.co/spaces/happynood/cv-match-tracker-demo](https://huggingface.co/spaces/happynood/cv-match-tracker-demo)
— upload a short clip, get tracking + team split (CPU demo; see the Space's README for
what's simplified vs. the full CLI below).

This project is assembly, not invention: detection (RF-DETR), tracking
(`supervision`'s ByteTrack), team clustering and pitch homography
(`roboflow/sports`) are reused as-is. The code here is the Hydra config, thin
adapters wiring those libraries together, the pitch-geometry statistics, and
the export/report layer.

## Stat tiers — read this before trusting a number

| Tier | Examples | Status | `method` in `stats.json` |
|---|---|---|---|
| **A — geometry** | distance covered, sprint distance/count, top/avg speed, heatmap, average position, minutes on screen | **Implemented, core of this repo** | `geometry` |
| **B — possession heuristics** | team possession %, naive pass/turnover detection | **Implemented, optional, off by default** (`ball.enabled: false`) | `heuristic` |
| **C — event spotting** | goals, shots, tackles, fouls, offsides, assists | **Not implemented.** Would require action-recognition models this repo does not ship. | — |

Every number in `stats.json` carries its `method` and a `confidence`. Nothing in Tier C is
fabricated or estimated — it is simply absent.

## Quickstart

```bash
uv sync --all-extras
uv run matchtracker run --config configs/config.yaml --video data/sample_clip.mp4 --output results/
```

Outputs land in `results/`:

- `players.parquet` — per-frame projected track positions (see data contract below)
- `stats.json` — per-track aggregates + a reproducibility manifest
- `report/index.html` — heatmaps, formation snapshot, stats table
- `annotated.mp4` — debug overlay (tracks + team colors)
- `run_manifest.json` — git SHA, config hash, model revisions, GPU fingerprint

`data/sample_clip.mp4` is a short real broadcast clip of a fixed tactical camera, included
as a working example. `data/smoke/` holds a tiny (2s, 640x360) crop used by the test suite
and CI — see [`data/smoke/README.md`](data/smoke/README.md).

### Static calibration

The camera is fixed, so the homography `H` is computed **once**, not per frame. By default
this repo uses **manual correspondences** (a JSON file of pixel → pitch-meter points) —
see `configs/config.yaml`'s `calibration` block and `data/smoke/manual_keypoints.json` for
the file format. An automatic pitch-keypoint detector is pluggable
(`calibrate.PitchKeypointDetector`) but none ships with this repo; see
[docs/DESIGN.md](docs/DESIGN.md) for why.

```bash
uv run matchtracker calibrate --config configs/config.yaml \
    --correspondences data/smoke/manual_keypoints.json --output results/homography.json
```

### Detector: bring your own football checkpoint

No football-fine-tuned detector weights ship with this repository (see
[docs/DESIGN.md](docs/DESIGN.md)). Without one, `detect.py` falls back to RF-DETR's
COCO-pretrained weights with `person → player` / `sports ball → ball` remapping — enough to
exercise the pipeline, not enough to hit the Tier A accuracy targets. To use a real
checkpoint fine-tuned on a football dataset, set in your config:

```yaml
detector:
  hf_repo_id: <your-org>/<your-football-rfdetr-checkpoint>
  hf_revision: <pinned-commit-or-tag>
```

## Metrics (Tier A geometry, spec §8)

- **Projection:** `[X,Y,W]ᵀ = H·[u,v,1]ᵀ`, `(x,y) = (X/W, Y/W)`, feet-point `(u,v) = ((x1+x2)/2, y2)`.
- **Smoothing:** Savitzky–Golay on `x(t), y(t)` before differencing.
- **Speed / outlier rejection:** `vₜ = ‖pₜ − pₜ₋₁‖ / Δt`; reject `vₜ > v_max` (default 12 m/s).
- **Distance covered:** sum of per-step displacement over valid (non-rejected) steps.
- **Sprints:** maximal segments with `vₜ > v_sprint` (default 7 m/s); summed distance + count.
- **Heatmap:** 2D histogram of `(x, y)` per track.
- **Minutes on screen / coverage:** `tracked_frames / fps / 60`, and fraction of the
  track's span actually observed.

## Config (Hydra)

All thresholds live in `configs/`, never hardcoded:

```
configs/
├── config.yaml              # top-level: video, team, calibration, metrics, pitch, ball, output
├── sport/football.yaml
├── detector/rfdetr_nano.yaml, rfdetr_small.yaml
└── tracker/bytetrack.yaml
```

Override anything from the CLI: `matchtracker run --config configs/config.yaml --override detector.conf=0.5 tracker.track_buffer=60`.

## Verification

```bash
make verify   # ruff check + ruff format --check + pyright + pytest (unit + CPU smoke e2e)
```

`make smoke` runs only the CPU end-to-end smoke test against `data/smoke/`. See
[docs/RUN_LOCAL.md](docs/RUN_LOCAL.md) for full-match GPU runs.

## Reproducing a run

Every `stats.json` embeds a `manifest`: git SHA (+ dirty flag), a SHA-256 of the resolved
config, the detector/team-classifier revisions used, and the GPU/driver/CUDA fingerprint.
Clustering (team assignment) is seeded (`seed` in config) for repeatability.

## Data contract (`players.parquet`)

`frame, t_s, track_id, team, cls, u, v, x_m, y_m, speed_ms, calib_valid` — validated by
`schemas.py` (also the single validation point for `stats.json`).

## Validation approach

- Tracking: MOTA/IDF1 via `matchtracker eval --predictions players.parquet --ground-truth <mot.csv>`
  (`motmetrics`; install with `uv sync --extra eval`).
- Team accuracy / calibration reprojection error / distance error: see
  [docs/DESIGN.md](docs/DESIGN.md) for the intended validation protocol (spec §12)
  — this repository ships the tooling; running it against a specific labeled clip is
  left to the user, since no such labeled clip is bundled.

## Hardware

Target: i5-11400H / 16GB RAM / RTX 3050 Laptop 4GB. RF-DETR Nano/Small in FP16; models load
sequentially; SAM2 is out of scope. See [docs/RUN_LOCAL.md](docs/RUN_LOCAL.md).

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE). Default tracker (`supervision`'s ByteTrack) is Apache-2.0
(MIT-compatible). The optional `boxmot` extra is AGPL-3.0 — kept out of core dependencies
for this reason; only install it if that license is acceptable for your use.
