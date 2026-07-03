# Design

## Scope (locked)

Football, a single **fixed** tactical camera (wide, full-pitch), offline/batch
processing, anonymous `track_id` + team (no jersey-number OCR), Tier A statistics with
optional Tier B. Tier C (event spotting) is out of scope for v1.

The fixed-camera assumption is what makes this tractable on a 4GB-VRAM laptop GPU: the
homography `H` is one constant matrix, computed once, so there is no per-frame
calibration and no ego-motion compensation. A plain motion tracker (ByteTrack) is
adequate; the residual error moves to far-touchline perspective distortion and
small-player recall at distance.

## Reuse vs. authored

| Stage | Reused library | Authored |
|---|---|---|
| Detection | `rfdetr` (RFDETRNano/Small) | Config + adapter (`detect.py`): model loading, per-class confidence thresholds, NMS |
| Tracking | `supervision`'s `sv.ByteTrack` | Adapter (`track.py`): config mapping, gap-fill, min-track-length filter |
| Team split | `roboflow/sports` `TeamClassifier` (SigLIP → UMAP → KMeans) | Crop extraction, per-track majority vote, stable A/B labeling (`team.py`) |
| Pitch + homography | `roboflow/sports` `SoccerPitchConfiguration`, `ViewTransformer`; `cv2.findHomography` (RANSAC) | Correspondence loading, fit/validate/wrap (`calibrate.py`) |
| Annotation / minimap | `supervision` annotators, `sports.annotators.soccer` | Report glue (`report.py`) |
| Metrics (Tier A) | — (this is the project) | `metrics.py`: projection, smoothing, speed, outlier rejection, distance, sprints, heatmap |
| Ball / possession (Tier B) | Same RF-DETR adapter at a lower confidence | Nearest-player heuristic, hold-frame debouncing (`ball.py`) |

Nothing here reimplements a detector, tracker, clustering algorithm, or homography
solver from scratch.

## Two-pass pipeline

```
Pass 1 (sampled, ~team.sample_fps): decode -> detect (no tracking needed) -> collect
        player crops -> fit TeamClassifier once.

Pass 2 (full video.target_fps): decode -> detect -> track (ByteTrack) -> at the same
        sample rate, crop + predict with the already-fitted classifier -> gap-fill /
        drop-short-tracks -> project through the static H -> Tier A metrics per track.
```

Pass 1 exists so team labels are decided from a stable, once-fit clustering rather than
re-fit per chunk, and so the (comparatively expensive) SigLIP embedding step only ever
processes sampled frames, not every frame of the match.

## What's *not* bundled, and why

**No football-fine-tuned detector checkpoint.** Fine-tuning RF-DETR on a labeled
football dataset (e.g. a Roboflow Universe football-players set) requires GPU-hours and
a licensed dataset; neither is something this repository can respectably fabricate. As
shipped, `detect.py` falls back to RF-DETR's COCO-pretrained weights with a
`person -> player` / `sports ball -> ball` label remap — sufficient to exercise every
stage of the pipeline (see the CPU smoke test), **not** sufficient to hit the Tier A
accuracy targets in the spec. `DetectorConfig.hf_repo_id` / `hf_revision` let you point
at your own fine-tuned checkpoint, pinned by revision.

**No bundled pitch-keypoint detector.** Roboflow's own pitch-keypoint model is served
through their hosted Inference API, a paid external dependency outside this project's
reuse list (RF-DETR, supervision, sports — all runnable locally). Rather than fake an
integration or silently degrade accuracy, `calibrate.py` ships a **manual
correspondences** path (a JSON file of pixel → pitch-meter points, `>= 4`, RANSAC-fit)
as the working default, and a `PitchKeypointDetector` protocol so a real keypoint model
can be dropped in without touching the rest of the pipeline.

**No labeled validation clip.** The spec's validation protocol (MOTA/IDF1/HOTA vs. a
SportsMOT-style clip, team-accuracy on hand-labeled frames, calibration reprojection
error, distance error on a hand-tracked segment) needs ground truth this repository does
not manufacture. `matchtracker eval` implements the MOTA/IDF1 computation
(`motmetrics`) against any MOTChallenge-format ground truth CSV the user supplies.

## Data contract

`players.parquet`: one row per `(track_id, frame)` —
`frame, t_s, track_id, team, cls, u, v, x_m, y_m, speed_ms, calib_valid`.
`stats.json`: per-track aggregates, each wrapped in a `{value, method, confidence}`
envelope, plus a `manifest` (git SHA/dirty, config SHA-256, model revisions, GPU
fingerprint, seed). `schemas.py` (pydantic) is the single validation point for both.

## Risks

- **Far-field perspective / small-object recall** is the main residual error under a
  fixed wide camera — worse at the far touchline. `calib_valid` and per-metric
  `confidence` are the escape hatches; there is no per-track distance-dependent
  confidence model yet.
- **Ball detection recall is low** at typical broadcast resolution; Tier B possession
  is a heuristic (nearest player within a radius, debounced) and is explicitly tagged
  `method="heuristic"`, never conflated with Tier A geometry.
- **4GB VRAM** forces sequential model loading (detector, then team classifier) and
  rules out SAM2-class segmentation models.
- **License:** `sv.ByteTrack` is Apache-2.0 (MIT-compatible) and is the default. The
  optional `boxmot` extra (OC-SORT etc.) is AGPL-3.0 and is kept an opt-in extra, never
  a core dependency, to avoid copyleft on an MIT repo.
