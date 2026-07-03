"""Gradio demo: upload a short fixed-camera football clip, get tracking + team split.

Reuses matchtracker's adapters directly (detect/track/team/ingest/metrics/report) —
see README.md in this Space for what's simplified relative to the full CLI (no
calibration, no football-fine-tuned detector).
"""

from __future__ import annotations

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

from matchtracker.detect import DetectorConfig, RFDETRAdapter
from matchtracker.ingest import iter_frames, probe_video
from matchtracker.metrics import coverage, minutes_on_screen
from matchtracker.report import annotate_video
from matchtracker.team import TeamClassifierAdapter, TeamConfig, crop_box, resolve_team_labels
from matchtracker.track import (
    ByteTrackAdapter,
    TrackerConfig,
    detections_to_records,
    filter_short_tracks,
    gap_fill,
)

MAX_SECONDS = 8
TARGET_FPS = 5
MAX_WIDTH = 960

_detector: RFDETRAdapter | None = None


def _get_detector() -> RFDETRAdapter:
    global _detector
    if _detector is None:
        _detector = RFDETRAdapter(DetectorConfig(name="rfdetr_nano", resolution=384))
    return _detector


def _trim_and_resize(video_path: str, out_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            video_path,
            "-t",
            str(MAX_SECONDS),
            "-vf",
            f"scale='min({MAX_WIDTH},iw)':-2",
            "-an",
            out_path,
        ],
        check=True,
    )


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


def process(video_path: str, progress=gr.Progress()):  # noqa: B008 (Gradio's documented pattern)
    if video_path is None:
        raise gr.Error("Upload a video first.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        trimmed_path = str(Path(tmp_dir) / "trimmed.mp4")
        progress(0.05, desc="Trimming clip")
        _trim_and_resize(video_path, trimmed_path)
        info = probe_video(trimmed_path)

        detector = _get_detector()
        tracker = ByteTrackAdapter(
            TrackerConfig(min_track_len=3, frame_rate=TARGET_FPS, track_buffer=15)
        )
        team_adapter = TeamClassifierAdapter(TeamConfig(n_clusters=2, device="cpu"))

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
        # Gradio needs the file to outlive the temp dir context.
        final_video_fd, final_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(final_video_fd)
        Path(final_video_path).write_bytes(Path(annotated_path).read_bytes())

        progress(0.9, desc="Building heatmaps + stats")
        positions_by_team: dict[str, list[tuple[float, float]]] = {"A": [], "B": []}
        for r in all_records:
            team = team_by_track.get(r.track_id, "unknown")
            if team in positions_by_team:
                u = (r.xyxy[0] + r.xyxy[2]) / 2
                v = r.xyxy[3]
                positions_by_team[team].append((u, v))

        heatmap_a = _pixel_heatmap(positions_by_team["A"], info.width, info.height, "Team A")
        heatmap_b = _pixel_heatmap(positions_by_team["B"], info.width, info.height, "Team B")

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

        progress(1.0, desc="Done")
        return final_video_path, heatmap_a, heatmap_b, stats_df


with gr.Blocks(title="matchtracker demo") as demo:
    gr.Markdown(
        "# matchtracker demo\n"
        "Upload a short (few-second) clip of a **fixed, wide tactical football camera**. "
        "This demo detects/tracks players and splits them into two teams, in **pixel space** "
        "(no per-clip pitch calibration — see this Space's README for what's simplified vs. "
        "the full [CLI](https://github.com/happynood/cv-match-tracker))."
    )
    with gr.Row():
        video_in = gr.Video(label="Upload clip")
        video_out = gr.Video(label="Annotated (tracks + team colors)")
    with gr.Row():
        heatmap_a_out = gr.Image(label="Team A position heatmap (pixel space)")
        heatmap_b_out = gr.Image(label="Team B position heatmap (pixel space)")
    stats_out = gr.Dataframe(label="Per-track stats (no calibration -> no distance/speed)")
    run_btn = gr.Button("Run", variant="primary")
    run_btn.click(
        process, inputs=[video_in], outputs=[video_out, heatmap_a_out, heatmap_b_out, stats_out]
    )

if __name__ == "__main__":
    demo.queue().launch()
