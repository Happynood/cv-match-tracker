"""Tracking adapter: wraps ``supervision``'s ``sv.ByteTrack``.

We do not reimplement a tracker. This module maps Hydra config names onto
``sv.ByteTrack``'s constructor, and adds the two project-specific
post-processing steps named in the spec (§6.3): dropping short tracks and
gap-filling short occlusion holes within a track's span.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv


@dataclass(frozen=True)
class TrackerConfig:
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    min_track_len: int = 15
    frame_rate: float = 10

    @classmethod
    def from_mapping(cls, mapping: Any) -> TrackerConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in mapping.items() if k in known})


class ByteTrackAdapter:
    """Thin wrapper over ``sv.ByteTrack``."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = sv.ByteTrack(
                track_activation_threshold=config.track_thresh,
                lost_track_buffer=config.track_buffer,
                minimum_matching_threshold=config.match_thresh,
                frame_rate=config.frame_rate,
            )

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self._tracker.update_with_detections(detections)

    def reset(self) -> None:
        self._tracker.reset()


@dataclass(frozen=True)
class TrackRecord:
    frame_idx: int
    t_s: float
    track_id: int
    cls: str
    xyxy: tuple[float, float, float, float]
    confidence: float
    interpolated: bool = False


def filter_short_tracks(records: list[TrackRecord], min_track_len: int) -> list[TrackRecord]:
    """Drop every record belonging to a track with fewer than ``min_track_len`` frames."""
    counts: dict[int, int] = {}
    for r in records:
        counts[r.track_id] = counts.get(r.track_id, 0) + 1
    return [r for r in records if counts[r.track_id] >= min_track_len]


def gap_fill(records: list[TrackRecord], max_gap: int) -> list[TrackRecord]:
    """Linearly interpolate missing frames within each track's observed span.

    Gaps longer than ``max_gap`` frames are left unfilled (the track is
    considered to have genuinely ended and re-started, likely a different
    physical object reusing an id after the tracker's lost-track buffer).
    """
    by_track: dict[int, list[TrackRecord]] = {}
    for r in records:
        by_track.setdefault(r.track_id, []).append(r)

    filled: list[TrackRecord] = []
    for track_id, track_records in by_track.items():
        track_records.sort(key=lambda r: r.frame_idx)
        filled.append(track_records[0])
        for prev, curr in zip(track_records, track_records[1:], strict=False):
            gap = curr.frame_idx - prev.frame_idx
            if 1 < gap <= max_gap + 1:
                for step in range(1, gap):
                    alpha = step / gap
                    px1, py1, px2, py2 = prev.xyxy
                    cx1, cy1, cx2, cy2 = curr.xyxy
                    xyxy = (
                        px1 + alpha * (cx1 - px1),
                        py1 + alpha * (cy1 - py1),
                        px2 + alpha * (cx2 - px2),
                        py2 + alpha * (cy2 - py2),
                    )
                    t_s = prev.t_s + alpha * (curr.t_s - prev.t_s)
                    filled.append(
                        TrackRecord(
                            frame_idx=prev.frame_idx + step,
                            t_s=t_s,
                            track_id=track_id,
                            cls=prev.cls,
                            xyxy=xyxy,
                            confidence=min(prev.confidence, curr.confidence),
                            interpolated=True,
                        )
                    )
            filled.append(curr)

    filled.sort(key=lambda r: (r.track_id, r.frame_idx))
    return filled


def detections_to_records(
    detections: sv.Detections, frame_idx: int, t_s: float
) -> list[TrackRecord]:
    """Convert a single frame's tracked ``sv.Detections`` into ``TrackRecord``s."""
    records = []
    class_names = detections.data.get("class_name", np.array(["player"] * len(detections)))
    tracker_ids = detections.tracker_id
    confidences = detections.confidence
    if tracker_ids is None:
        return records
    for i in range(len(detections)):
        if tracker_ids[i] is None:
            continue
        x1, y1, x2, y2 = detections.xyxy[i]
        records.append(
            TrackRecord(
                frame_idx=frame_idx,
                t_s=t_s,
                track_id=int(tracker_ids[i]),
                cls=str(class_names[i]),
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(confidences[i]) if confidences is not None else 0.0,
            )
        )
    return records


__all__ = [
    "TrackerConfig",
    "ByteTrackAdapter",
    "TrackRecord",
    "filter_short_tracks",
    "gap_fill",
    "detections_to_records",
]
