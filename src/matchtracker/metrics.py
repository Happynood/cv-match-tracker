"""Tier A geometry metrics — the genuinely project-specific core (spec §8).

Everything here operates on a single track's already-projected ``(x_m, y_m)``
trajectory. Smoothing uses ``scipy.signal.savgol_filter`` (reuse, not
reimplementation); the metric definitions themselves (distance, sprints,
outlier rejection, heatmap, coverage) are what this project actually
contributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class MetricsConfig:
    v_max_ms: float = 12.0
    sprint_thresh_ms: float = 7.0
    smooth: str = "savgol"
    savgol_window: int = 7
    savgol_polyorder: int = 2

    @classmethod
    def from_mapping(cls, mapping: Any) -> MetricsConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in mapping.items() if k in known})


@dataclass(frozen=True)
class TrackTrajectory:
    track_id: int
    frame_idx: np.ndarray
    t_s: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray


def smooth_trajectory(traj: TrackTrajectory, config: MetricsConfig) -> TrackTrajectory:
    """Savitzky-Golay smoothing on x(t), y(t) — raw projected coords jitter."""
    n = len(traj.x_m)
    window = min(config.savgol_window, n if n % 2 == 1 else n - 1)
    if config.smooth != "savgol" or window < config.savgol_polyorder + 2:
        return traj
    x = np.asarray(savgol_filter(traj.x_m, window, config.savgol_polyorder))
    y = np.asarray(savgol_filter(traj.y_m, window, config.savgol_polyorder))
    return TrackTrajectory(traj.track_id, traj.frame_idx, traj.t_s, x, y)


def compute_speeds(traj: TrackTrajectory, v_max_ms: float) -> np.ndarray:
    """Per-step speed with outlier rejection (spec §8).

    ``v[i]`` is the speed between sample ``i-1`` and ``i``. Speeds exceeding
    ``v_max_ms`` or spanning a frame-index gap (an ID gap unfilled by
    ``track.gap_fill``) are set to ``nan`` so they are excluded from
    distance/sprint accumulation rather than silently treated as zero.
    """
    n = len(traj.x_m)
    speeds = np.full(n, np.nan)
    for i in range(1, n):
        dt = traj.t_s[i] - traj.t_s[i - 1]
        if dt <= 0:
            continue
        dist = np.hypot(traj.x_m[i] - traj.x_m[i - 1], traj.y_m[i] - traj.y_m[i - 1])
        v = dist / dt
        if v <= v_max_ms:
            speeds[i] = v
    return speeds


def distance_covered_m(traj: TrackTrajectory, speeds: np.ndarray) -> float:
    """``D = sum(||p_t - p_{t-1}||)`` over valid (non-nan, post-filter) steps."""
    total = 0.0
    for i in range(1, len(traj.t_s)):
        if np.isnan(speeds[i]):
            continue
        dt = traj.t_s[i] - traj.t_s[i - 1]
        total += speeds[i] * dt
    return float(total)


@dataclass(frozen=True)
class SprintSummary:
    distance_m: float
    count: int


def sprint_summary(
    traj: TrackTrajectory, speeds: np.ndarray, sprint_thresh_ms: float
) -> SprintSummary:
    """Maximal segments with ``v > sprint_thresh_ms``: summed distance + count."""
    total_distance = 0.0
    count = 0
    in_sprint = False
    for i in range(1, len(traj.t_s)):
        v = speeds[i]
        is_sprinting = not np.isnan(v) and v > sprint_thresh_ms
        if is_sprinting:
            dt = traj.t_s[i] - traj.t_s[i - 1]
            total_distance += v * dt
            if not in_sprint:
                count += 1
            in_sprint = True
        else:
            in_sprint = False
    return SprintSummary(distance_m=total_distance, count=count)


def top_speed_ms(speeds: np.ndarray) -> float:
    valid = speeds[~np.isnan(speeds)]
    return float(np.max(valid)) if len(valid) else 0.0


def avg_speed_ms(speeds: np.ndarray) -> float:
    valid = speeds[~np.isnan(speeds)]
    return float(np.mean(valid)) if len(valid) else 0.0


def heatmap_2d(
    traj: TrackTrajectory, pitch_length_m: float, pitch_width_m: float, bins: int = 50
) -> np.ndarray:
    """2-D histogram of ``(x, y)`` occupancy, normalized to sum to 1."""
    hist, _, _ = np.histogram2d(
        traj.x_m,
        traj.y_m,
        bins=bins,
        range=[[0, pitch_length_m], [0, pitch_width_m]],
    )
    total = hist.sum()
    return hist / total if total > 0 else hist


def minutes_on_screen(n_tracked_frames: int, fps: float) -> float:
    return n_tracked_frames / fps / 60.0


def coverage(n_tracked_frames: int, n_expected_frames: int) -> float:
    """Fraction of the track's expected span actually present (expected ~= 1)."""
    if n_expected_frames <= 0:
        return 0.0
    return min(1.0, n_tracked_frames / n_expected_frames)


def average_position_m(traj: TrackTrajectory) -> tuple[float, float]:
    return float(np.mean(traj.x_m)), float(np.mean(traj.y_m))


__all__ = [
    "MetricsConfig",
    "TrackTrajectory",
    "SprintSummary",
    "smooth_trajectory",
    "compute_speeds",
    "distance_covered_m",
    "sprint_summary",
    "top_speed_ms",
    "avg_speed_ms",
    "heatmap_2d",
    "minutes_on_screen",
    "coverage",
    "average_position_m",
]
