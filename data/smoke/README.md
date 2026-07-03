# Smoke fixtures

- `smoke_clip.mp4` — a 2s, 640x360, 10fps crop of `data/sample_clip.mp4`, committed so
  CI and local `make smoke` can run the full pipeline without network access to a real
  match video.
- `manual_keypoints.json` — 4 pixel-to-pitch-meter correspondences, hand-picked by eye
  against the penalty box / six-yard box lines visible in `smoke_clip.mp4`'s first
  frame. These are **approximate** (eyeballed from a low-resolution frame) and exist
  only to exercise `calibrate.py`'s manual-correspondences code path in tests — they
  are not a validated calibration and should not be used to make accuracy claims.
