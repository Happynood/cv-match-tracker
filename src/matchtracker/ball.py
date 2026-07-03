"""Optional Tier B: ball possession heuristics (spec §6.7, §3).

Ball detection reuses the same RF-DETR adapter (``detect.py``) at
``conf_ball``; ``sahi`` tiled inference is available as an opt-in extra for
better small-object recall on the low-confidence ball class. Possession
assignment (nearest player to ball) and pass/turnover detection are simple,
explicit heuristics — tagged ``method="heuristic"`` end-to-end per the spec's
provenance requirement, never conflated with the Tier A geometry metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from matchtracker.calibrate import project_feet_point
from matchtracker.track import TrackRecord


@dataclass(frozen=True)
class BallConfig:
    enabled: bool = False
    sahi: bool = True
    possession_radius_m: float = 2.0
    hold_frames: int = 5

    @classmethod
    def from_mapping(cls, mapping: Any) -> BallConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in mapping.items() if k in known})


@dataclass(frozen=True)
class PossessionEvent:
    frame_idx: int
    t_s: float
    track_id: int
    team: str
    is_turnover: bool


def nearest_player_to_ball(
    ball_xy_m: tuple[float, float],
    player_positions_m: dict[int, tuple[float, float]],
    possession_radius_m: float,
) -> int | None:
    """Nearest-player-to-ball ownership heuristic; ``None`` if none within radius."""
    best_id, best_dist = None, float("inf")
    for track_id, (x, y) in player_positions_m.items():
        dist = np.hypot(ball_xy_m[0] - x, ball_xy_m[1] - y)
        if dist < best_dist:
            best_id, best_dist = track_id, dist
    if best_id is not None and best_dist <= possession_radius_m:
        return best_id
    return None


def detect_possession_changes(
    frame_holders: list[tuple[int, float, int | None]],
    hold_frames: int,
    track_team: dict[int, str],
) -> list[PossessionEvent]:
    """Turn a per-frame holder sequence into possession/turnover events.

    Args:
        frame_holders: ``(frame_idx, t_s, holder_track_id_or_None)`` per frame,
            already produced by ``nearest_player_to_ball``.
        hold_frames: minimum consecutive frames required before a holder
            change counts as a real possession change (debounces flicker).
        track_team: track_id -> team label, for turnover detection.
    """
    events: list[PossessionEvent] = []
    current_holder: int | None = None
    candidate_holder: int | None = None
    candidate_run = 0

    for frame_idx, t_s, holder in frame_holders:
        if holder == candidate_holder:
            candidate_run += 1
        else:
            candidate_holder = holder
            candidate_run = 1

        if (
            candidate_holder is not None
            and candidate_holder != current_holder
            and candidate_run >= hold_frames
        ):
            is_turnover = current_holder is not None and track_team.get(
                current_holder
            ) != track_team.get(candidate_holder)
            events.append(
                PossessionEvent(
                    frame_idx=frame_idx,
                    t_s=t_s,
                    track_id=candidate_holder,
                    team=track_team.get(candidate_holder, "unknown"),
                    is_turnover=is_turnover,
                )
            )
            current_holder = candidate_holder

    return events


def team_possession_pct(
    events: list[PossessionEvent], frame_holders: list[tuple[int, float, int | None]]
) -> dict[str, float]:
    """Fraction of held frames per team, attributing each frame to the last-established holder."""
    if not events:
        return {}
    counts: dict[str, int] = {}
    current_team = None
    event_idx = 0
    total = 0
    for frame_idx, _t_s, _holder in frame_holders:
        while event_idx < len(events) and events[event_idx].frame_idx <= frame_idx:
            current_team = events[event_idx].team
            event_idx += 1
        if current_team is not None:
            counts[current_team] = counts.get(current_team, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {team: n / total for team, n in counts.items()}


def ball_track_to_positions_m(
    ball_records: list[TrackRecord], transformer
) -> dict[int, tuple[float, float]]:
    """Project ball bbox feet-points through the static homography, keyed by frame_idx."""
    positions = {}
    for r in ball_records:
        u, v = project_feet_point(r.xyxy)
        x_m, y_m = transformer.transform_points(np.array([[u, v]], dtype=np.float32))[0]
        positions[r.frame_idx] = (float(x_m), float(y_m))
    return positions


__all__ = [
    "BallConfig",
    "PossessionEvent",
    "nearest_player_to_ball",
    "detect_possession_changes",
    "team_possession_pct",
    "ball_track_to_positions_m",
]
