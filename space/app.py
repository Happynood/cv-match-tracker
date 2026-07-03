"""Gradio demo: upload a short fixed-camera football clip, get tracking + team split.

Reuses matchtracker's adapters directly (detect/track/team/ingest/metrics/calibrate/report)
- see README.md in this Space for what's simplified relative to the full CLI. Supports an
optional calibration-keypoints JSON (real distance/speed instead of pixel space), a detector
choice, and a compute-device choice (GPU only offered when CUDA is actually available - e.g.
when this Space is cloned and run locally on a machine with an NVIDIA GPU).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from matchtracker.calibrate import Correspondence, fit_homography, project_feet_point
from matchtracker.detect import DetectorConfig, RFDETRAdapter
from matchtracker.ingest import iter_frames, probe_video
from matchtracker.metrics import (
    MetricsConfig,
    TrackTrajectory,
    avg_speed_ms,
    compute_speeds,
    coverage,
    distance_covered_m,
    heatmap_2d,
    minutes_on_screen,
    smooth_trajectory,
    sprint_summary,
    top_speed_ms,
)
from matchtracker.report import annotate_video, render_heatmap_image
from matchtracker.team import TeamClassifierAdapter, TeamConfig, crop_box, resolve_team_labels
from matchtracker.track import (
    ByteTrackAdapter,
    TrackerConfig,
    TrackRecord,
    detections_to_records,
    filter_short_tracks,
    gap_fill,
)

MAX_SECONDS = 8
TARGET_FPS = 5
MAX_WIDTH = 960
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
REPROJ_ERROR_MAX_M = 2.0

DETECTOR_CHOICES: dict[str, tuple[str, int]] = {
    "RF-DETR Nano (fast, default)": ("rfdetr_nano", 384),
    "RF-DETR Small (more accurate, slower on CPU)": ("rfdetr_small", 512),
}
DEVICE_MAP: dict[str, str] = {"CPU": "cpu", "GPU (CUDA)": "cuda"}
DEVICE_CHOICES: list[str] = ["CPU"] + (["GPU (CUDA)"] if torch.cuda.is_available() else [])

KEYPOINTS_PLACEHOLDER = """[
  {"pixel": [712, 486], "pitch_m": [88.5, 13.84]},
  {"pixel": [730, 900], "pitch_m": [88.5, 54.16]},
  {"pixel": [1400, 520], "pitch_m": [99.5, 24.84]},
  {"pixel": [1490, 780], "pitch_m": [99.5, 43.16]}
]"""

_detector_cache: dict[tuple[str, str], RFDETRAdapter] = {}


def _get_detector(model_choice: str, device_choice: str) -> RFDETRAdapter:
    device = DEVICE_MAP[device_choice]
    key = (model_choice, device)
    if key not in _detector_cache:
        name, resolution = DETECTOR_CHOICES[model_choice]
        _detector_cache[key] = RFDETRAdapter(
            DetectorConfig(name=name, resolution=resolution, device=device)
        )
    return _detector_cache[key]


def _trim_and_resize(video_path: str, out_path: str) -> None:
    # Reject anything that isn't a plain local file before it reaches ffmpeg: ffmpeg's
    # own protocol handling (http/data/concat/subprocess/...) can otherwise be abused
    # for SSRF or local-file-read tricks via a crafted -i argument.
    validated_path = Path(video_path).resolve(strict=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe,fd",
            "-i",
            str(validated_path),
            "-t",
            str(MAX_SECONDS),
            "-vf",
            f"scale='min({MAX_WIDTH},iw)':-2",
            "-an",
            out_path,
        ],
        check=True,
    )


def _extract_preview_frame(video_path: str | None) -> np.ndarray | None:
    """First frame of the *trimmed/resized* clip, so clicked pixel coords match what
    the pipeline itself sees (the raw upload may be a different resolution)."""
    if not video_path:
        return None
    with tempfile.TemporaryDirectory() as tmp_dir:
        preview_path = str(Path(tmp_dir) / "preview.mp4")
        try:
            _trim_and_resize(video_path, preview_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        cap = cv2.VideoCapture(preview_path)
        ok, frame = cap.read()
        cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _on_frame_click(evt: gr.SelectData) -> str:
    u, v = evt.index
    return f'Clicked pixel: u={u}, v={v}  ->  {{"pixel": [{u}, {v}], "pitch_m": [X, Y]}}'


def _parse_keypoints(text: str) -> list[Correspondence] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        entries = json.loads(text)
    except json.JSONDecodeError as e:
        raise gr.Error(f"Calibration keypoints JSON is invalid: {e}") from e
    if not isinstance(entries, list) or len(entries) < 4:
        raise gr.Error(
            'Provide at least 4 correspondences: [{"pixel": [u, v], "pitch_m": [x, y]}, ...]'
        )
    correspondences = []
    for entry in entries:
        try:
            u, v = entry["pixel"]
            x_m, y_m = entry["pitch_m"]
        except (KeyError, TypeError, ValueError) as e:
            raise gr.Error(
                f"Bad keypoint entry {entry!r}: expected pixel:[u,v], pitch_m:[x,y]"
            ) from e
        correspondences.append(
            Correspondence(pixel=(float(u), float(v)), pitch_m=(float(x_m), float(y_m)))
        )
    return correspondences


def _pixel_heatmap(
    positions: list[tuple[float, float]], width: int, height: int, title: str
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if positions:
        xs, ys = zip(*positions, strict=True)
        ax.hist2d(xs, ys, bins=30, range=[[0, width], [0, height]], cmap="hot")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("pixel u")
    ax.set_ylabel("pixel v")
    fig.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(*fig.canvas.get_width_height()[::-1], 4)[:, :, :3]
    plt.close(fig)
    return image


def _persist_temp_file(src_path: Path, suffix: str) -> str:
    """Gradio needs output files to outlive the temp dir they were built in."""
    fd, out_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    Path(out_path).write_bytes(src_path.read_bytes())
    return out_path


def _project_trajectories(records: list[TrackRecord], transformer) -> dict[int, TrackTrajectory]:
    by_track: dict[int, list[TrackRecord]] = {}
    for r in records:
        by_track.setdefault(r.track_id, []).append(r)

    trajectories = {}
    for track_id, track_records in by_track.items():
        track_records.sort(key=lambda r: r.frame_idx)
        points = np.array([project_feet_point(r.xyxy) for r in track_records], dtype=np.float32)
        projected = transformer.transform_points(points)
        trajectories[track_id] = TrackTrajectory(
            track_id=track_id,
            frame_idx=np.array([r.frame_idx for r in track_records]),
            t_s=np.array([r.t_s for r in track_records]),
            x_m=projected[:, 0],
            y_m=projected[:, 1],
        )
    return trajectories


def _calibrated_report(
    all_records: list[TrackRecord],
    team_by_track: dict[int, str],
    tmp_dir: str,
    correspondences: list[Correspondence],
) -> tuple[str, str, str, pd.DataFrame]:
    calibration = fit_homography(correspondences, reproj_error_max_m=REPROJ_ERROR_MAX_M)
    trajectories = _project_trajectories(all_records, calibration.transformer)

    metrics_cfg = MetricsConfig()
    smoothed = {tid: smooth_trajectory(traj, metrics_cfg) for tid, traj in trajectories.items()}
    speeds = {tid: compute_speeds(s, metrics_cfg.v_max_ms) for tid, s in smoothed.items()}

    records_by_track: dict[int, list[TrackRecord]] = {}
    for r in all_records:
        records_by_track.setdefault(r.track_id, []).append(r)

    rows = []
    for tid in trajectories:
        s = smoothed[tid]
        v = speeds[tid]
        sprint = sprint_summary(s, v, metrics_cfg.sprint_thresh_ms)
        recs = records_by_track[tid]
        span = max(r.frame_idx for r in recs) - min(r.frame_idx for r in recs) + 1
        rows.append(
            {
                "track_id": tid,
                "team": team_by_track.get(tid, "unknown"),
                "class": recs[0].cls,
                "distance_m": round(distance_covered_m(s, v), 1),
                "top_speed_ms": round(top_speed_ms(v), 2),
                "avg_speed_ms": round(avg_speed_ms(v), 2),
                "sprint_count": sprint.count,
                "minutes_on_screen": round(minutes_on_screen(len(recs), TARGET_FPS), 3),
                "coverage": round(coverage(len(recs), span), 2),
            }
        )
    stats_df = pd.DataFrame(rows).sort_values("track_id")

    heatmap_paths = {}
    for team in ("A", "B"):
        xs = [x for tid, s in smoothed.items() if team_by_track.get(tid) == team for x in s.x_m]
        ys = [y for tid, s in smoothed.items() if team_by_track.get(tid) == team for y in s.y_m]
        combined = TrackTrajectory(
            track_id=-1,
            frame_idx=np.arange(len(xs)),
            t_s=np.arange(len(xs), dtype=np.float64),
            x_m=np.array(xs, dtype=np.float64),
            y_m=np.array(ys, dtype=np.float64),
        )
        hist = heatmap_2d(combined, PITCH_LENGTH_M, PITCH_WIDTH_M)
        img_path = render_heatmap_image(
            hist, Path(tmp_dir) / f"heatmap_{team}.png", title=f"Team {team} (pitch meters)"
        )
        heatmap_paths[team] = _persist_temp_file(img_path, ".png")

    valid = "yes" if calibration.valid else "NO (exceeds threshold - treat metrics as unreliable)"
    status = (
        f"**Calibrated.** Reprojection error: {calibration.reprojection_error_m:.2f}m "
        f"(valid: {valid}). Distance/speed below are in real pitch meters."
    )
    return status, heatmap_paths["A"], heatmap_paths["B"], stats_df


def _pixel_space_report(
    all_records: list[TrackRecord],
    team_by_track: dict[int, str],
    by_track: dict[int, list[TrackRecord]],
    width: int,
    height: int,
) -> tuple[str, np.ndarray, np.ndarray, pd.DataFrame]:
    positions_by_team: dict[str, list[tuple[float, float]]] = {"A": [], "B": []}
    for r in all_records:
        team = team_by_track.get(r.track_id, "unknown")
        if team in positions_by_team:
            u = (r.xyxy[0] + r.xyxy[2]) / 2
            v = r.xyxy[3]
            positions_by_team[team].append((u, v))

    heatmap_a = _pixel_heatmap(positions_by_team["A"], width, height, "Team A (pixel space)")
    heatmap_b = _pixel_heatmap(positions_by_team["B"], width, height, "Team B (pixel space)")

    rows = []
    for track_id, records in by_track.items():
        span = max(r.frame_idx for r in records) - min(r.frame_idx for r in records) + 1
        rows.append(
            {
                "track_id": track_id,
                "team": team_by_track.get(track_id, "unknown"),
                "class": records[0].cls,
                "minutes_on_screen": round(minutes_on_screen(len(records), TARGET_FPS), 3),
                "coverage": round(coverage(len(records), span), 2),
            }
        )
    stats_df = pd.DataFrame(rows).sort_values("track_id")

    status = (
        "No calibration keypoints provided - positions shown in **pixel space** "
        "(no real distance/speed). Add keypoints below for real-world metrics."
    )
    return status, heatmap_a, heatmap_b, stats_df


def process(
    video_path: str,
    model_choice: str,
    device_choice: str,
    keypoints_json: str,
    progress=gr.Progress(),  # noqa: B008 (Gradio's documented pattern)
):
    if video_path is None:
        raise gr.Error("Upload a video first.")

    correspondences = _parse_keypoints(keypoints_json)
    device = DEVICE_MAP[device_choice]

    with tempfile.TemporaryDirectory() as tmp_dir:
        trimmed_path = str(Path(tmp_dir) / "trimmed.mp4")
        progress(0.05, desc="Trimming clip")
        _trim_and_resize(video_path, trimmed_path)
        info = probe_video(trimmed_path)

        detector = _get_detector(model_choice, device_choice)
        tracker = ByteTrackAdapter(
            TrackerConfig(min_track_len=3, frame_rate=TARGET_FPS, track_buffer=15)
        )
        team_adapter = TeamClassifierAdapter(TeamConfig(n_clusters=2, device=device))

        all_records = []
        fit_crops: list[tuple[int, np.ndarray]] = []

        progress(0.15, desc="Detecting + tracking")
        for frame in iter_frames(trimmed_path, target_fps=TARGET_FPS):
            image_rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
            detections = detector.detect(image_rgb)
            tracked = tracker.update(detections)
            records = detections_to_records(tracked, frame.frame_idx, frame.t_s)
            all_records.extend(r for r in records if r.cls != "ball")
            for r in records:
                if r.cls in ("referee", "ball"):
                    continue
                crop = crop_box(frame.image, r.xyxy)
                if crop.size:
                    fit_crops.append((r.track_id, crop))

        if len(fit_crops) < 2:
            raise gr.Error(
                "No players detected in this clip — try a clip with a clearer, wider pitch view."
            )

        progress(0.6, desc="Splitting into teams")
        track_ids_for_fit = [tid for tid, _ in fit_crops]
        crops_only = [c for _, c in fit_crops]
        team_adapter.fit(crops_only)
        predictions = team_adapter.predict(crops_only)
        crop_predictions = list(zip(track_ids_for_fit, (int(p) for p in predictions), strict=True))

        all_records = gap_fill(all_records, max_gap=15)
        all_records = filter_short_tracks(all_records, min_track_len=3)
        team_by_track = resolve_team_labels(crop_predictions, all_records)

        progress(0.75, desc="Rendering annotated video")
        records_by_frame: dict[int, list] = {}
        by_track: dict[int, list] = {}
        for r in all_records:
            records_by_frame.setdefault(r.frame_idx, []).append(r)
            by_track.setdefault(r.track_id, []).append(r)

        annotated_path = str(Path(tmp_dir) / "annotated.mp4")
        annotate_video(
            trimmed_path,
            records_by_frame,
            team_by_track,
            annotated_path,
            TARGET_FPS,
            (info.width, info.height),
        )
        final_video_path = _persist_temp_file(Path(annotated_path), ".mp4")

        progress(0.9, desc="Building heatmaps + stats")
        if correspondences:
            status, heatmap_a, heatmap_b, stats_df = _calibrated_report(
                all_records, team_by_track, tmp_dir, correspondences
            )
        else:
            status, heatmap_a, heatmap_b, stats_df = _pixel_space_report(
                all_records, team_by_track, by_track, info.width, info.height
            )

        progress(1.0, desc="Done")
        return final_video_path, heatmap_a, heatmap_b, stats_df, status


with gr.Blocks(title="matchtracker demo") as demo:
    gr.Markdown(
        "# matchtracker demo\n"
        "Upload a short (few-second) clip of a **fixed, wide tactical football camera**. "
        "This demo detects/tracks players and splits them into two teams. Add calibration "
        "keypoints below for real distance/speed stats, or leave them empty for a pixel-space "
        "demo. See this Space's README for what's simplified vs. the full "
        "[CLI](https://github.com/happynood/cv-match-tracker)."
    )
    with gr.Row():
        video_in = gr.Video(label="Upload clip")
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=list(DETECTOR_CHOICES),
                value=next(iter(DETECTOR_CHOICES)),
                label="Detector model",
            )
            device_dropdown = gr.Dropdown(
                choices=DEVICE_CHOICES,
                value="CPU",
                label="Compute device",
                info=(
                    "Defaults to CPU. A GPU option appears automatically when this app runs "
                    "somewhere with CUDA available (e.g. cloned and run locally with an "
                    "NVIDIA GPU) — see this Space's README for 'Run locally on a GPU'."
                ),
            )

    with gr.Accordion(
        "Calibration keypoints (optional, for real-world distance/speed)", open=False
    ):
        gr.Markdown(
            "Pixel positions are only meaningful for the exact camera that produced them, so "
            "there's no universal default. Upload a clip below to preview its first frame "
            "(after this demo's own trim/resize, so coordinates line up), click on visible "
            "pitch markings (penalty box / six-yard box corners work well) to read off pixel "
            "coordinates, then pair each with its real pitch position in meters "
            "(0-105 x 0-68) in the JSON box. Need >= 4 points."
        )
        frame_preview = gr.Image(
            label="First frame — click to read pixel coordinates", interactive=False
        )
        click_readout = gr.Textbox(label="Last clicked pixel", interactive=False)
        keypoints_box = gr.Textbox(
            value="",
            lines=6,
            label="Calibration keypoints (JSON)",
            placeholder=KEYPOINTS_PLACEHOLDER,
        )

    video_in.change(fn=_extract_preview_frame, inputs=[video_in], outputs=[frame_preview])
    frame_preview.select(fn=_on_frame_click, outputs=[click_readout])

    run_btn = gr.Button("Run", variant="primary")
    status_out = gr.Markdown()
    with gr.Row():
        video_out = gr.Video(label="Annotated (tracks + team colors)")
    with gr.Row():
        heatmap_a_out = gr.Image(label="Team A position heatmap")
        heatmap_b_out = gr.Image(label="Team B position heatmap")
    stats_out = gr.Dataframe(label="Per-track stats")
    run_btn.click(
        process,
        inputs=[video_in, model_dropdown, device_dropdown, keypoints_box],
        outputs=[video_out, heatmap_a_out, heatmap_b_out, stats_out, status_out],
    )

if __name__ == "__main__":
    demo.queue().launch()
