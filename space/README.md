---
title: Match Tracker Demo
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
python_version: "3.11"
---

# matchtracker demo

Upload a short clip (a few seconds) of a **fixed, wide tactical football camera** and
this Space will detect players/referees, track them across frames, split them into two
teams, and return an annotated video + a position heatmap per team.

**No football-fine-tuned detector ships with this repo.** Both detector options below use
RF-DETR's COCO-pretrained weights with a `person -> player` / `sports ball -> ball` label
remap — good enough to demo tracking and team-splitting, not tuned for football-specific
accuracy. See the [GitHub repo](https://github.com/happynood/cv-match-tracker) for the full
CLI, which supports bringing your own fine-tuned checkpoint.

## Options

- **Detector model** — RF-DETR Nano (fast, default) or Small (more accurate, slower on CPU).
- **Compute device** — defaults to CPU everywhere. A **GPU (CUDA)** option appears
  automatically when this app is running somewhere with CUDA available — it won't show up
  on this hosted Space (CPU-only hardware), but will if you run it locally with an NVIDIA
  GPU (see below).
- **Calibration keypoints (optional)** — without them, positions are reported in **pixel
  space** (no real distance/speed, since those need real units). Provide >= 4
  `{"pixel": [u, v], "pitch_m": [x, y]}` correspondences and this demo fits a one-time
  homography and reports real distance/top-speed/avg-speed/sprint-count in pitch meters,
  same as the full CLI's `matchtracker calibrate`. After uploading a clip, expand
  "Calibration keypoints" to preview its first frame, click on visible pitch markings
  (penalty box / six-yard box corners work well) to read off pixel coordinates, and pair
  each with its real position in meters (pitch is 105m x 68m) in the JSON box.

Keep uploads short (a few seconds) — the hosted Space runs on shared CPU.

## Run locally on a GPU

This Space's code is a normal Gradio app — clone it and run it anywhere with Python:

```bash
git clone https://huggingface.co/spaces/happynood/cv-match-tracker-demo
cd cv-match-tracker-demo
pip install -r requirements.txt
python app.py
```

If a CUDA-capable NVIDIA GPU and driver are present, standard PyTorch wheels already include
CUDA support, so no extra step is needed — `torch.cuda.is_available()` is checked at
startup, and the **Compute device** dropdown will offer **GPU (CUDA)** alongside CPU.
Selecting it runs both the detector and the team classifier on the GPU, which is
meaningfully faster than CPU, especially with the Small detector.
