import numpy as np

from matchtracker.metrics import (
    MetricsConfig,
    TrackTrajectory,
    average_position_m,
    compute_speeds,
    coverage,
    distance_covered_m,
    heatmap_2d,
    minutes_on_screen,
    sprint_summary,
)


def _traj(x_m, y_m, dt=1.0):
    n = len(x_m)
    return TrackTrajectory(
        track_id=1,
        frame_idx=np.arange(n),
        t_s=np.arange(n) * dt,
        x_m=np.array(x_m, dtype=float),
        y_m=np.array(y_m, dtype=float),
    )


def test_compute_speeds_rejects_outlier_above_v_max():
    # 100m in 1s is way above v_max -> rejected (nan)
    traj = _traj([0, 100, 101], [0, 0, 0])
    speeds = compute_speeds(traj, v_max_ms=12.0)
    assert np.isnan(speeds[0])  # no previous point
    assert np.isnan(speeds[1])  # 100 m/s > 12 m/s -> rejected
    assert not np.isnan(speeds[2])  # 1 m/s -> kept


def test_distance_covered_ignores_rejected_steps():
    traj = _traj([0, 100, 101, 102], [0, 0, 0, 0])
    speeds = compute_speeds(traj, v_max_ms=12.0)
    distance = distance_covered_m(traj, speeds)
    # only the last two 1m steps should count (100->101, 101->102)
    assert distance == 2.0


def test_sprint_summary_counts_maximal_segments():
    # two separate sprint segments (v=8m/s) separated by a slow segment (v=1m/s)
    x = [0, 8, 16, 17, 25, 33]
    traj = _traj(x, [0] * len(x))
    speeds = compute_speeds(traj, v_max_ms=12.0)
    summary = sprint_summary(traj, speeds, sprint_thresh_ms=7.0)
    assert summary.count == 2
    # segment 1: 0->8->16 (2 steps of 8m); segment 2: 17->25->33 (2 steps of 8m)
    assert summary.distance_m == 32.0


def test_coverage_bounded_at_one():
    assert coverage(n_tracked_frames=10, n_expected_frames=10) == 1.0
    assert coverage(n_tracked_frames=5, n_expected_frames=10) == 0.5
    assert coverage(n_tracked_frames=10, n_expected_frames=5) == 1.0  # never > 1
    assert coverage(n_tracked_frames=0, n_expected_frames=0) == 0.0


def test_minutes_on_screen():
    assert minutes_on_screen(n_tracked_frames=600, fps=10.0) == 1.0


def test_average_position():
    traj = _traj([0, 10, 20], [0, 5, 10])
    assert average_position_m(traj) == (10.0, 5.0)


def test_heatmap_2d_normalizes_to_one():
    traj = _traj([1, 2, 3, 4], [1, 2, 3, 4])
    heatmap = heatmap_2d(traj, pitch_length_m=105, pitch_width_m=68, bins=10)
    assert np.isclose(heatmap.sum(), 1.0)


def test_metrics_config_from_mapping_ignores_unknown_keys():
    cfg = MetricsConfig.from_mapping({"v_max_ms": 10.0, "unknown_key": 123})
    assert cfg.v_max_ms == 10.0
