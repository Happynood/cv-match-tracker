# Running a full match locally (GPU)

Target hardware: i5-11400H, 16GB RAM, RTX 3050 Laptop 4GB. A 90-minute match is expected
to take **hours**, not minutes, at these specs — that's an accepted tradeoff for running
on a 4GB card at all.

## 1. Environment

```bash
uv sync --all-extras
nvidia-smi   # confirm the GPU is visible; RF-DETR/team classifier auto-detect CUDA
```

## 2. Calibrate once

Pick >= 4 pixel points you can identify against known pitch features (penalty spot,
box corners, center circle, halfway line) in a representative frame, and write them to
a JSON file (see `data/smoke/manual_keypoints.json` for the format: `pixel` in image
coordinates, `pitch_m` in meters on a 105x68 pitch, or `pitch_vertex` referencing
`SoccerPitchConfiguration().vertices`).

```bash
uv run matchtracker calibrate \
    --config configs/config.yaml \
    --correspondences path/to/your_match_keypoints.json \
    --output results/homography.json
```

Check `reprojection_error_m` before proceeding — it should be well under
`calibration.reproj_error_max_m` (default 2.0m). If it isn't, your points likely don't
correspond to the same physical locations; re-pick them.

## 3. Iterate on a short clip first

```bash
uv run matchtracker run \
    --config configs/config.yaml \
    --video /path/to/full_match.mp4 \
    --output results/clip_test/ \
    --override video.clip=[60,180] calibration.correspondences_path=path/to/your_match_keypoints.json
```

`video.clip=[60,180]` restricts decoding to seconds 60-180 — use this to sanity-check
detection/tracking/team assignment quality before committing to a multi-hour full run.

## 4. Full match

Once the short clip looks right, drop `video.clip` and let it run:

```bash
uv run matchtracker run \
    --config configs/config.yaml \
    --video /path/to/full_match.mp4 \
    --output results/full_match/ \
    --override calibration.correspondences_path=path/to/your_match_keypoints.json
```

Recommended overrides for full 90-minute runs on 4GB VRAM:

- `detector.name=rfdetr_nano` (Small only if you have headroom to spare)
- `video.target_fps=10` (or lower; positional metrics tolerate downsampling per spec)
- `output.save_annotated_video=false` if you don't need the debug video (saves time + disk)

## 5. Ball / possession (Tier B, optional)

```bash
uv run matchtracker run --config configs/config.yaml --video /path/to/full_match.mp4 \
    --override ball.enabled=true ball.sahi=true
```

This is a heuristic (nearest player within `ball.possession_radius_m`, debounced by
`ball.hold_frames`) and is tagged `method="heuristic"` in `stats.json` — expect it to be
noisier than the Tier A geometry metrics, especially with low ball-detection recall.

## 6. Evaluate tracking quality

If you have MOTChallenge-format ground truth for a segment:

```bash
uv sync --extra eval
uv run matchtracker eval --predictions results/full_match/players.parquet \
    --ground-truth path/to/ground_truth.csv --output results/eval.json
```
