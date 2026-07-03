"""Team-assignment adapter: wraps roboflow/sports ``TeamClassifier``.

``TeamClassifier`` (SigLIP embeddings -> UMAP -> KMeans, fit once) is imported
as-is; we only author the crop extraction, the fixed A/B label assignment
(clusters are unordered, so we canonicalize by mean pitch x-position after the
first homography pass), the referee exclusion, and the per-track majority
vote across a match.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from matchtracker.track import TrackRecord


@dataclass(frozen=True)
class TeamConfig:
    sample_fps: float = 1.0
    n_clusters: int = 2
    device: str = "cpu"
    batch_size: int = 32

    @classmethod
    def from_mapping(cls, mapping: Any) -> TeamConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in mapping.items() if k in known})


def crop_box(image: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(max(0, v)) for v in xyxy)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


class TeamClassifierAdapter:
    """Fits once on sampled player crops, then labels tracks by majority vote."""

    def __init__(self, config: TeamConfig):
        from sports.common.team import TeamClassifier

        self.config = config
        self._classifier = TeamClassifier(device=config.device, batch_size=config.batch_size)
        self._classifier.cluster_model.n_clusters = config.n_clusters
        self._fitted = False

    def fit(self, crops: list[np.ndarray]) -> None:
        if len(crops) < self.config.n_clusters:
            raise ValueError(
                f"Need at least {self.config.n_clusters} sampled player crops to fit "
                f"TeamClassifier, got {len(crops)}."
            )
        self._classifier.fit(crops)
        self._fitted = True

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TeamClassifierAdapter.fit() must be called before predict().")
        return self._classifier.predict(crops)


def majority_vote_teams(
    track_cluster_votes: dict[int, list[int]],
) -> dict[int, int]:
    """Collapse each track's per-frame cluster predictions into one label."""
    result = {}
    for track_id, votes in track_cluster_votes.items():
        if not votes:
            continue
        result[track_id] = Counter(votes).most_common(1)[0][0]
    return result


def canonicalize_labels(
    cluster_to_track_ids: dict[int, list[int]],
    track_mean_x: dict[int, float],
) -> dict[int, str]:
    """Map raw (unordered) cluster ids to stable team labels "A"/"B".

    Cluster "A" is defined as the cluster whose tracks have the smaller mean
    horizontal position (image pixel ``u``, or pitch ``x_m`` if already
    projected — any consistent horizontal axis works since this is only used
    to make the arbitrary cluster order *stable* across a match, not to
    assert a real-world side).
    """
    cluster_mean_x = {}
    for cluster_id, track_ids in cluster_to_track_ids.items():
        xs = [track_mean_x[tid] for tid in track_ids if tid in track_mean_x]
        if xs:
            cluster_mean_x[cluster_id] = float(np.mean(xs))
    ordered = sorted(cluster_mean_x, key=lambda c: cluster_mean_x[c])
    labels = {}
    for i, cluster_id in enumerate(ordered):
        labels[cluster_id] = "A" if i == 0 else "B"
    return labels


def resolve_team_labels(
    track_crop_predictions: list[tuple[int, int]],
    records: list[TrackRecord],
    referee_class: str = "referee",
    ball_class: str = "ball",
) -> dict[int, str]:
    """Collapse per-crop cluster predictions into stable per-track "A"/"B" labels.

    Args:
        track_crop_predictions: ``(track_id, cluster_id)`` pairs, one per
            sampled crop that was fed through ``TeamClassifierAdapter.predict``.
        records: full track history, used only to compute each track's mean
            horizontal position for canonicalizing cluster order (see
            ``canonicalize_labels``).
    """
    votes: dict[int, list[int]] = {}
    for track_id, cluster in track_crop_predictions:
        votes.setdefault(track_id, []).append(cluster)
    track_cluster = majority_vote_teams(votes)

    cluster_to_tracks: dict[int, list[int]] = {}
    for track_id, cluster in track_cluster.items():
        cluster_to_tracks.setdefault(cluster, []).append(track_id)

    eligible = [r for r in records if r.cls not in (referee_class, ball_class)]
    by_track: dict[int, list[TrackRecord]] = {}
    for r in eligible:
        by_track.setdefault(r.track_id, []).append(r)
    track_mean_x = {
        track_id: float(np.mean([(r.xyxy[0] + r.xyxy[2]) / 2 for r in track_records]))
        for track_id, track_records in by_track.items()
    }

    cluster_labels = canonicalize_labels(cluster_to_tracks, track_mean_x)
    return {tid: cluster_labels.get(cluster, "unknown") for tid, cluster in track_cluster.items()}


__all__ = [
    "TeamConfig",
    "TeamClassifierAdapter",
    "crop_box",
    "majority_vote_teams",
    "canonicalize_labels",
    "resolve_team_labels",
]
