---
title: Match Tracker Demo
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
python_version: "3.11"
---

# matchtracker demo

Upload a short clip (a few seconds) of a **fixed, wide tactical football camera** and
this Space will detect players/referees, track them across frames, split them into two
teams, and return an annotated video + a pixel-space position heatmap per team.

**This is a CPU demo, not the full pipeline.** Two things are simplified relative to the
full `matchtracker` CLI (see the [GitHub repo](https://github.com/happynood/cv-match-tracker)):

1. **No pitch calibration.** The full pipeline projects pixel positions onto real pitch
   meters via a one-time homography fit, which needs hand-picked correspondences for
   *your specific camera*. An anonymous upload has none, so this demo reports positions
   in **pixel space** — no distance/speed/sprint statistics, since those require real
   units. Run the CLI locally with `matchtracker calibrate` for those.
2. **No football-fine-tuned detector.** This Space uses RF-DETR Nano's COCO-pretrained
   weights with a `person -> player` / `sports ball -> ball` label remap — good enough
   to demo tracking and team-splitting, not tuned for football-specific accuracy.

Keep uploads short (a few seconds) — this runs on shared CPU.
