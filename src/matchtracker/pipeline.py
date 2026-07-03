"""Two-pass pipeline orchestration (spec §5).

Pass 1 (sampled, cheap): decode at ``team.sample_fps``, run detection only
(no tracking needed for fitting), fit ``TeamClassifier`` once.
Pass 2 (full ``video.target_fps``): decode + detect + track every sampled
frame, sampling player crops for team majority-vote along the way, then
gap-fill/filter tracks, project through the static homography, and compute
Tier A metrics per track.

This module is glue: each stage is delegated to its adapter module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf

from matchtracker import ball as ball_mod
from matchtracker import metrics as metrics_mod
from matchtracker import report as report_mod
from matchtracker.calibrate import (
    CalibrationConfig,
    calibrate_static,
    pitch_config,
    project_feet_point,
)
from matchtracker.detect import DetectorConfig, RFDETRAdapter
from matchtracker.ingest import build_run_manifest, iter_frames, probe_video, write_run_manifest
from matchtracker.schemas import (
    MatchStats,
    MetricValue,
    PlayerFrameRow,
    Team,
    TrackClass,
    TrackStats,
)
from matchtracker.team import TeamClassifierAdapter, TeamConfig, crop_box, resolve_team_labels
from matchtracker.track import (
    ByteTrackAdapter,
    TrackerConfig,
    TrackRecord,
    detections_to_records,
    filter_short_tracks,
    gap_fill,
)

log = logging.getLogger(__name__)

REFEREE_CLASS = "referee"
BALL_CLASS = "ball"


@dataclass
class PipelineResult:
    stats: MatchStats
    players_parquet_path: Path
    stats_json_path: Path
    report_html_path: Path | None
    annotated_video_path: Path | None


def _sample_stride(target_fps: float, sample_fps: float) -> int:
    return max(1, round(target_fps / sample_fps))


def _pass1_fit_team_classifier(
    video_path: str,
    team_cfg: TeamConfig,
    detector: RFDETRAdapter,
) -> TeamClassifierAdapter:
    """Cheap sampled decode: collect crops from raw detections, fit once."""
    adapter = TeamClassifierAdapter(team_cfg)
    crops: list[np.ndarray] = []

    for frame in iter_frames(video_path, target_fps=team_cfg.sample_fps):
        image_rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        detections = detector.detect(image_rgb)
        class_names = detections.data.get("class_name", [])
        for i in range(len(detections)):
            cls = str(class_names[i]) if len(class_names) > i else "player"
            if cls in (REFEREE_CLASS, BALL_CLASS):
                continue
            x1, y1, x2, y2 = detections.xyxy[i]
            crop = crop_box(frame.image, (float(x1), float(y1), float(x2), float(y2)))
            if crop.size:
                crops.append(crop)

    adapter.fit(crops)
    return adapter


def _pass2_detect_track(
    video_path: str,
    video_cfg: Any,
    detector: RFDETRAdapter,
    tracker_cfg: TrackerConfig,
    team_cfg: TeamConfig,
    team_adapter: TeamClassifierAdapter,
) -> tuple[list[TrackRecord], dict[int, str], list[TrackRecord]]:
    """Full-fps decode + detect + track; sample crops for team voting inline."""
    tracker = ByteTrackAdapter(tracker_cfg)
    target_fps = video_cfg.get("target_fps") or 10
    resize = tuple(video_cfg["resize"]) if video_cfg.get("resize") else None
    clip = tuple(video_cfg["clip"]) if video_cfg.get("clip") else None
    team_stride = _sample_stride(target_fps, team_cfg.sample_fps)

    all_records: list[TrackRecord] = []
    ball_records: list[TrackRecord] = []
    crop_predictions: list[tuple[int, int]] = []

    for frame in iter_frames(video_path, target_fps=target_fps, resize=resize, clip=clip):
        image_rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        detections = detector.detect(image_rgb)
        tracked = tracker.update(detections)
        records = detections_to_records(tracked, frame.frame_idx, frame.t_s)
        all_records.extend(r for r in records if r.cls != BALL_CLASS)
        ball_records.extend(r for r in records if r.cls == BALL_CLASS)

        if frame.sample_idx % team_stride == 0:
            for r in records:
                if r.cls in (REFEREE_CLASS, BALL_CLASS):
                    continue
                crop = crop_box(frame.image, r.xyxy)
                if crop.size == 0:
                    continue
                cluster = int(team_adapter.predict([crop])[0])
                crop_predictions.append((r.track_id, cluster))

    all_records = gap_fill(all_records, max_gap=tracker_cfg.track_buffer)
    all_records = filter_short_tracks(all_records, tracker_cfg.min_track_len)
    team_by_track = resolve_team_labels(crop_predictions, all_records)

    return all_records, team_by_track, ball_records


def _project_records(
    records: list[TrackRecord], transformer
) -> dict[int, metrics_mod.TrackTrajectory]:
    by_track: dict[int, list[TrackRecord]] = {}
    for r in records:
        by_track.setdefault(r.track_id, []).append(r)

    trajectories = {}
    for track_id, track_records in by_track.items():
        track_records.sort(key=lambda r: r.frame_idx)
        points = np.array([project_feet_point(r.xyxy) for r in track_records], dtype=np.float32)
        projected = transformer.transform_points(points)
        trajectories[track_id] = metrics_mod.TrackTrajectory(
            track_id=track_id,
            frame_idx=np.array([r.frame_idx for r in track_records]),
            t_s=np.array([r.t_s for r in track_records]),
            x_m=projected[:, 0],
            y_m=projected[:, 1],
        )
    return trajectories


def _track_stats(
    traj: metrics_mod.TrackTrajectory,
    smoothed: metrics_mod.TrackTrajectory,
    speeds: np.ndarray,
    records: list[TrackRecord],
    team: str,
    cls: str,
    metrics_cfg: metrics_mod.MetricsConfig,
    fps: float,
    calib_valid: bool,
) -> TrackStats:
    distance = metrics_mod.distance_covered_m(smoothed, speeds)
    sprint = metrics_mod.sprint_summary(smoothed, speeds, metrics_cfg.sprint_thresh_ms)
    n_frames = len(records)
    span = int(traj.frame_idx.max() - traj.frame_idx.min() + 1) if n_frames else 1
    avg_pos = metrics_mod.average_position_m(smoothed)
    confidence = 1.0 if calib_valid else 0.5

    return TrackStats(
        track_id=traj.track_id,
        team=Team(team),
        cls=TrackClass(cls),
        distance_m=MetricValue(value=distance, method="geometry", confidence=confidence),
        sprint_distance_m=MetricValue(
            value=sprint.distance_m, method="geometry", confidence=confidence
        ),
        sprint_count=MetricValue(value=sprint.count, method="geometry", confidence=confidence),
        top_speed_ms=MetricValue(
            value=metrics_mod.top_speed_ms(speeds), method="geometry", confidence=confidence
        ),
        avg_speed_ms=MetricValue(
            value=metrics_mod.avg_speed_ms(speeds), method="geometry", confidence=confidence
        ),
        minutes_on_screen=MetricValue(
            value=metrics_mod.minutes_on_screen(n_frames, fps), method="geometry", confidence=1.0
        ),
        coverage=MetricValue(
            value=metrics_mod.coverage(n_frames, span), method="geometry", confidence=1.0
        ),
        avg_position_m=MetricValue(value=list(avg_pos), method="geometry", confidence=confidence),
    )


def run_pipeline(cfg: DictConfig, output_dir: str | Path | None = None) -> PipelineResult:
    output_dir = Path(output_dir or cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = cfg.video.path
    video_info = probe_video(video_path)
    target_fps = cfg.video.target_fps or video_info.fps

    detector_cfg = DetectorConfig.from_mapping(OmegaConf.to_container(cfg.detector, resolve=True))
    tracker_cfg = TrackerConfig.from_mapping(OmegaConf.to_container(cfg.tracker, resolve=True))
    team_cfg = TeamConfig.from_mapping(OmegaConf.to_container(cfg.team, resolve=True))
    calib_cfg = CalibrationConfig.from_mapping(
        OmegaConf.to_container(cfg.calibration, resolve=True)
    )
    metrics_cfg = metrics_mod.MetricsConfig.from_mapping(
        OmegaConf.to_container(cfg.metrics, resolve=True)
    )

    log.info("Loading detector: %s", detector_cfg.name)
    detector = RFDETRAdapter(detector_cfg)

    log.info("Pass 1: fitting team classifier on sampled crops")
    team_adapter = _pass1_fit_team_classifier(video_path, team_cfg, detector)

    log.info("Calibrating static homography (method=%s)", calib_cfg.method)
    pitch = pitch_config(cfg.pitch.length_m, cfg.pitch.width_m)
    calibration = calibrate_static(calib_cfg, pitch)
    if not calibration.valid:
        log.warning(
            "Calibration reprojection error %.2fm exceeds threshold %.2fm",
            calibration.reprojection_error_m,
            calib_cfg.reproj_error_max_m,
        )

    log.info("Pass 2: detect + track at target_fps=%s", target_fps)
    records, team_by_track, ball_records = _pass2_detect_track(
        video_path,
        OmegaConf.to_container(cfg.video, resolve=True),
        detector,
        tracker_cfg,
        team_cfg,
        team_adapter,
    )

    trajectories = _project_records(records, calibration.transformer)
    records_by_track: dict[int, list[TrackRecord]] = {}
    for r in records:
        records_by_track.setdefault(r.track_id, []).append(r)

    cls_by_track = {tid: recs[0].cls for tid, recs in records_by_track.items()}

    smoothed_by_track = {
        tid: metrics_mod.smooth_trajectory(traj, metrics_cfg) for tid, traj in trajectories.items()
    }
    speeds_by_track = {
        tid: metrics_mod.compute_speeds(smoothed, metrics_cfg.v_max_ms)
        for tid, smoothed in smoothed_by_track.items()
    }

    track_stats = [
        _track_stats(
            trajectories[tid],
            smoothed_by_track[tid],
            speeds_by_track[tid],
            records_by_track[tid],
            team_by_track.get(tid, "unknown"),
            cls_by_track[tid],
            metrics_cfg,
            target_fps,
            calibration.valid,
        )
        for tid in trajectories
    ]

    ball_config = ball_mod.BallConfig.from_mapping(OmegaConf.to_container(cfg.ball, resolve=True))
    team_possession: MetricValue | None = None
    if ball_config.enabled and ball_records:
        ball_positions = ball_mod.ball_track_to_positions_m(ball_records, calibration.transformer)
        player_positions_by_frame: dict[int, dict[int, tuple[float, float]]] = {}
        for tid, traj in trajectories.items():
            for frame_idx, x, y in zip(traj.frame_idx, traj.x_m, traj.y_m, strict=True):
                player_positions_by_frame.setdefault(int(frame_idx), {})[tid] = (float(x), float(y))

        frame_holders = []
        for r in sorted(ball_records, key=lambda r: r.frame_idx):
            ball_xy = ball_positions.get(r.frame_idx)
            players_here = player_positions_by_frame.get(r.frame_idx, {})
            holder = (
                ball_mod.nearest_player_to_ball(
                    ball_xy, players_here, ball_config.possession_radius_m
                )
                if ball_xy and players_here
                else None
            )
            frame_holders.append((r.frame_idx, r.t_s, holder))

        events = ball_mod.detect_possession_changes(
            frame_holders, ball_config.hold_frames, team_by_track
        )
        pct = ball_mod.team_possession_pct(events, frame_holders)
        if pct:
            team_possession = MetricValue(value=pct, method="heuristic", confidence=0.6)

    model_revisions = {
        "detector": detector.model_revision,
        "team_classifier": "google/siglip-base-patch16-224",
    }
    manifest = build_run_manifest(
        OmegaConf.to_container(cfg, resolve=True), model_revisions, seed=cfg.get("seed", 0)
    )

    stats = MatchStats(
        manifest=manifest,
        tracks=track_stats,
        team_possession_pct=team_possession,
        calibration_reproj_error_m=calibration.reprojection_error_m,
    )

    row_index_by_track = {
        tid: {int(f): i for i, f in enumerate(traj.frame_idx)} for tid, traj in trajectories.items()
    }

    player_rows = []
    for r in records:
        i = row_index_by_track[r.track_id][r.frame_idx]
        smoothed = smoothed_by_track[r.track_id]
        speeds = speeds_by_track[r.track_id]
        speed = speeds[i]
        player_rows.append(
            PlayerFrameRow(
                frame=r.frame_idx,
                t_s=r.t_s,
                track_id=r.track_id,
                team=Team(team_by_track.get(r.track_id, "unknown")),
                cls=TrackClass(r.cls),
                u=project_feet_point(r.xyxy)[0],
                v=project_feet_point(r.xyxy)[1],
                x_m=float(smoothed.x_m[i]),
                y_m=float(smoothed.y_m[i]),
                speed_ms=float(speed) if not np.isnan(speed) else 0.0,
                calib_valid=calibration.valid,
            )
        )

    players_path = report_mod.write_players_parquet(player_rows, output_dir / "players.parquet")
    stats_path = report_mod.write_stats_json(stats, output_dir / "stats.json")
    write_run_manifest(manifest, output_dir)

    report_html_path = None
    image_paths: list[str] = []
    for track_id, traj in trajectories.items():
        if cls_by_track[track_id] not in ("player", "goalkeeper"):
            continue
        heatmap = metrics_mod.heatmap_2d(traj, cfg.pitch.length_m, cfg.pitch.width_m)
        img_path = report_mod.render_heatmap_image(
            heatmap, output_dir / "report" / f"heatmap_{track_id}.png", title=f"Track {track_id}"
        )
        image_paths.append(img_path.name)

    avg_positions = {
        tid: metrics_mod.average_position_m(traj) for tid, traj in trajectories.items()
    }
    formation_path = report_mod.render_formation_image(
        avg_positions, team_by_track, pitch, output_dir / "report" / "formation.png"
    )
    image_paths.append(formation_path.name)

    report_html_path = report_mod.render_report_html(
        stats, image_paths, output_dir / "report" / "index.html"
    )

    annotated_video_path = None
    if cfg.output.get("save_annotated_video", True):
        records_by_frame: dict[int, list[TrackRecord]] = {}
        for r in records:
            records_by_frame.setdefault(r.frame_idx, []).append(r)
        annotated_video_path = report_mod.annotate_video(
            video_path,
            records_by_frame,
            team_by_track,
            output_dir / "annotated.mp4",
            target_fps,
            (video_info.width, video_info.height)
            if not cfg.video.get("resize")
            else tuple(cfg.video.resize),
        )

    return PipelineResult(
        stats=stats,
        players_parquet_path=players_path,
        stats_json_path=stats_path,
        report_html_path=report_html_path,
        annotated_video_path=annotated_video_path,
    )


__all__ = ["PipelineResult", "run_pipeline"]
