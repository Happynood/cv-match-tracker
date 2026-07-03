"""Export + report adapter (spec §6.8, §9): parquet/json export, annotated
video, minimap/heatmaps, FIFA-style HTML sheet.

Drawing is delegated to ``supervision`` annotators (video overlay) and
``sports.annotators.soccer`` (pitch/minimap); this module only wires them
together and writes files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import supervision as sv
from jinja2 import Template
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration

from matchtracker.schemas import MatchStats, PlayerFrameRow

TEAM_COLORS = {
    "A": sv.Color(230, 57, 70),
    "B": sv.Color(29, 53, 87),
    "unknown": sv.Color(150, 150, 150),
}


def write_players_parquet(rows: list[PlayerFrameRow], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row.model_dump() for row in rows])
    df.to_parquet(path, index=False)
    return path


def write_stats_json(stats: MatchStats, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stats.model_dump_json(indent=2))
    return path


def render_heatmap_image(heatmap: np.ndarray, out_path: str | Path, title: str = "") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(heatmap.T, origin="lower", cmap="hot", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("x (pitch length)")
    ax.set_ylabel("y (pitch width)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_formation_image(
    positions_m: dict[int, tuple[float, float]],
    team_by_track: dict[int, str],
    pitch: SoccerPitchConfiguration,
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = draw_pitch(pitch)
    for team, color in (("A", sv.Color(230, 57, 70)), ("B", sv.Color(29, 53, 87))):
        xy_cm = np.array(
            [
                (x * 100, y * 100)
                for tid, (x, y) in positions_m.items()
                if team_by_track.get(tid) == team
            ],
            dtype=np.float32,
        )
        if xy_cm.size == 0:
            continue
        canvas = draw_points_on_pitch(pitch, xy_cm, face_color=color, pitch=canvas)
    import cv2

    cv2.imwrite(str(out_path), canvas)
    return out_path


def annotate_video(
    video_path: str | Path,
    records_by_frame: dict[int, list],
    team_by_track: dict[int, str],
    output_path: str | Path,
    fps: float,
    resolution_wh: tuple[int, int],
) -> Path:
    """Draw ellipse + track-id labels per frame using ``supervision`` annotators."""
    from matchtracker.ingest import iter_frames

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette(list(TEAM_COLORS.values())))
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette(list(TEAM_COLORS.values())))

    with sv.VideoSink(
        str(output_path), sv.VideoInfo(resolution_wh[0], resolution_wh[1], int(fps))
    ) as sink:
        for frame in iter_frames(video_path, target_fps=fps):
            records = records_by_frame.get(frame.frame_idx, [])
            if not records:
                sink.write_frame(frame.image)
                continue

            xyxy = np.array([r.xyxy for r in records], dtype=np.float32)
            track_ids = np.array([r.track_id for r in records])
            teams = [team_by_track.get(r.track_id, "unknown") for r in records]
            color_idx = np.array([list(TEAM_COLORS.keys()).index(t) for t in teams])

            detections = sv.Detections(xyxy=xyxy, tracker_id=track_ids, class_id=color_idx)
            annotated = ellipse_annotator.annotate(frame.image.copy(), detections)  # type: ignore[arg-type]
            annotated = label_annotator.annotate(
                annotated,  # type: ignore[arg-type]
                detections,
                labels=[str(tid) for tid in track_ids],
            )
            sink.write_frame(np.asarray(annotated))

    return output_path


_REPORT_TEMPLATE = Template(
    """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Match Report</title>
<style>
body { font-family: sans-serif; margin: 2rem; background: #0b1220; color: #eee; }
table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
th, td { border: 1px solid #333; padding: 0.4rem 0.6rem; text-align: right; }
th { background: #1d3557; }
td:first-child, th:first-child { text-align: left; }
img { max-width: 100%; margin: 0.5rem 0; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.caveat { color: #f4a261; font-size: 0.9rem; }
</style>
</head>
<body>
<h1>Match Report</h1>
<p class="caveat">Tier A metrics are geometric (method=geometry). Tier B (possession) is
heuristic. No Tier C (events) metrics are reported.</p>

<div class="grid">
  {% for image in images %}
  <div><img src="{{ image }}"></div>
  {% endfor %}
</div>

<h2>Per-track statistics</h2>
<table>
<tr>
  <th>track_id</th><th>team</th><th>class</th>
  <th>distance (m)</th><th>sprint dist (m)</th><th>sprints</th>
  <th>top speed (m/s)</th><th>avg speed (m/s)</th>
  <th>minutes</th><th>coverage</th>
</tr>
{% for t in tracks %}
<tr>
  <td>{{ t.track_id }}</td><td>{{ t.team }}</td><td>{{ t.cls }}</td>
  <td>{{ "%.1f"|format(t.distance_m.value) }}</td>
  <td>{{ "%.1f"|format(t.sprint_distance_m.value) }}</td>
  <td>{{ t.sprint_count.value }}</td>
  <td>{{ "%.1f"|format(t.top_speed_ms.value) }}</td>
  <td>{{ "%.1f"|format(t.avg_speed_ms.value) }}</td>
  <td>{{ "%.1f"|format(t.minutes_on_screen.value) }}</td>
  <td>{{ "%.2f"|format(t.coverage.value) }}</td>
</tr>
{% endfor %}
</table>

<p>git SHA: {{ manifest.git_sha }} (dirty={{ manifest.git_dirty }}) &middot;
config SHA-256: {{ manifest.config_sha256[:12] }} &middot;
GPU: {{ manifest.gpu_name }}</p>
</body>
</html>
"""
)


def render_report_html(stats: MatchStats, image_paths: list[str], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = _REPORT_TEMPLATE.render(images=image_paths, tracks=stats.tracks, manifest=stats.manifest)
    out_path.write_text(html)
    return out_path


__all__ = [
    "write_players_parquet",
    "write_stats_json",
    "render_heatmap_image",
    "render_formation_image",
    "annotate_video",
    "render_report_html",
]
