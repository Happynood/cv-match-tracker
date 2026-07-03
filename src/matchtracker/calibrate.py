"""Static homography adapter: ``SoccerPitchConfiguration`` + ``ViewTransformer``.

Per the spec, this camera is fixed, so ``H`` is computed **once** for the
whole match, not per frame.

Two correspondence sources are supported:

- ``manual`` (default, always available): a JSON sidecar of
  ``{"pixel": [u, v], "pitch_vertex": <1-based index into
  SoccerPitchConfiguration().vertices>}`` or ``{"pixel": [u, v], "pitch_m":
  [x, y]}`` entries, hand-picked by the operator (spec §6.5's "manual 4-point
  fallback", generalized to N >= 4 points so we can RANSAC-fit and report a
  reprojection error).
- ``keypoints_ransac``: pluggable — accepts any callable implementing
  ``PitchKeypointDetector`` (frame -> correspondences) so a fine-tuned
  pitch-keypoint model can be dropped in. No such detector is bundled with
  this repository (Roboflow's is served behind their hosted Inference API,
  which is a paid external dependency outside this project's reuse list), so
  by default this method raises a clear error directing the user to either
  supply a detector or use ``manual``.

We do not reimplement homography estimation: fitting delegates to
``cv2.findHomography`` (via a thin RANSAC call) and application delegates to
``sports.common.view.ViewTransformer``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration


@dataclass(frozen=True)
class CalibrationConfig:
    static: bool = True
    method: str = "manual"
    refit_on_bump: bool = False
    min_keypoints: int = 6
    reproj_error_max_m: float = 2.0
    correspondences_path: str | None = None

    @classmethod
    def from_mapping(cls, mapping: Any) -> CalibrationConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in mapping.items() if k in known})


@dataclass(frozen=True)
class Correspondence:
    pixel: tuple[float, float]
    pitch_m: tuple[float, float]


class PitchKeypointDetector(Protocol):
    def __call__(self, image: np.ndarray) -> list[Correspondence]: ...


@dataclass
class CalibrationResult:
    transformer: ViewTransformer
    reprojection_error_m: float
    n_points: int
    valid: bool


def pitch_config(length_m: float, width_m: float) -> SoccerPitchConfiguration:
    return SoccerPitchConfiguration(length=int(length_m * 100), width=int(width_m * 100))


def load_manual_correspondences(
    path: str | Path, pitch: SoccerPitchConfiguration
) -> list[Correspondence]:
    entries = json.loads(Path(path).read_text())
    vertices_m = [(x / 100.0, y / 100.0) for x, y in pitch.vertices]

    correspondences = []
    for entry in entries:
        u, v = entry["pixel"]
        if "pitch_m" in entry:
            x_m, y_m = entry["pitch_m"]
        elif "pitch_vertex" in entry:
            x_m, y_m = vertices_m[entry["pitch_vertex"] - 1]
        else:
            raise ValueError(f"Correspondence entry missing pitch_m/pitch_vertex: {entry}")
        correspondences.append(Correspondence(pixel=(u, v), pitch_m=(x_m, y_m)))
    return correspondences


def fit_homography(
    correspondences: list[Correspondence], reproj_error_max_m: float
) -> CalibrationResult:
    if len(correspondences) < 4:
        raise ValueError(
            f"Need >= 4 correspondences to fit a homography, got {len(correspondences)}."
        )

    source = np.array([c.pixel for c in correspondences], dtype=np.float32)
    target = np.array([c.pitch_m for c in correspondences], dtype=np.float32)

    method = cv2.RANSAC if len(correspondences) > 4 else 0
    matrix, _inlier_mask = cv2.findHomography(
        source, target, method=method, ransacReprojThreshold=reproj_error_max_m
    )
    if matrix is None:
        raise ValueError("Homography could not be estimated from the given correspondences.")

    transformer = _wrap_matrix(matrix)

    projected = transformer.transform_points(source)
    errors = np.linalg.norm(projected - target, axis=1)
    reprojection_error_m = float(np.mean(errors))

    return CalibrationResult(
        transformer=transformer,
        reprojection_error_m=reprojection_error_m,
        n_points=len(correspondences),
        valid=reprojection_error_m <= reproj_error_max_m,
    )


def _wrap_matrix(matrix: np.ndarray) -> ViewTransformer:
    """Build a ``ViewTransformer`` from a precomputed homography matrix.

    ``ViewTransformer.__init__`` only accepts point pairs (it always calls
    ``cv2.findHomography`` itself, which requires >= 4 points), so we build it
    from 4 trivial identical-mapping points and then overwrite ``.m`` with our
    own RANSAC-fit matrix.
    """
    dummy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    transformer = ViewTransformer(source=dummy, target=dummy)
    transformer.m = matrix
    return transformer


def calibrate_static(
    config: CalibrationConfig, pitch: SoccerPitchConfiguration
) -> CalibrationResult:
    if config.method == "manual":
        if not config.correspondences_path:
            raise ValueError("calibration.correspondences_path is required for method='manual'.")
        correspondences = load_manual_correspondences(config.correspondences_path, pitch)
    elif config.method == "keypoints_ransac":
        raise NotImplementedError(
            "method='keypoints_ransac' requires a pitch-keypoint detector to be supplied "
            "programmatically (see calibrate.PitchKeypointDetector); none ships with this "
            "repository. Use method='manual' with a correspondences JSON file, or call "
            "fit_homography() directly with correspondences from your own detector."
        )
    else:
        raise ValueError(f"Unknown calibration method: {config.method}")

    if len(correspondences) < config.min_keypoints:
        raise ValueError(
            f"Only {len(correspondences)} correspondences given, "
            f"calibration.min_keypoints={config.min_keypoints}."
        )
    return fit_homography(correspondences, config.reproj_error_max_m)


def project_feet_point(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    """Feet-point per spec §6.6: ``(u, v) = ((x1+x2)/2, y2)``."""
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, y2)


__all__ = [
    "CalibrationConfig",
    "Correspondence",
    "PitchKeypointDetector",
    "CalibrationResult",
    "pitch_config",
    "load_manual_correspondences",
    "fit_homography",
    "calibrate_static",
    "project_feet_point",
]
