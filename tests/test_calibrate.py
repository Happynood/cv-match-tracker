import json

import numpy as np
import pytest

from matchtracker.calibrate import (
    Correspondence,
    fit_homography,
    load_manual_correspondences,
    pitch_config,
    project_feet_point,
)


def test_project_feet_point_is_bottom_center_of_bbox():
    assert project_feet_point((0.0, 0.0, 10.0, 20.0)) == (5.0, 20.0)


def test_fit_homography_recovers_known_rectangle_mapping():
    # pixel rectangle -> pitch rectangle (0,0)-(105,68), an exact affine mapping
    correspondences = [
        Correspondence(pixel=(100, 50), pitch_m=(0, 0)),
        Correspondence(pixel=(540, 50), pitch_m=(105, 0)),
        Correspondence(pixel=(540, 310), pitch_m=(105, 68)),
        Correspondence(pixel=(100, 310), pitch_m=(0, 68)),
    ]
    result = fit_homography(correspondences, reproj_error_max_m=1.0)
    assert result.valid
    assert result.reprojection_error_m < 1e-3
    assert result.n_points == 4

    # midpoint of the pixel rectangle should map close to pitch center
    center = result.transformer.transform_points(np.array([[320.0, 180.0]], dtype=np.float32))
    assert np.allclose(center[0], [52.5, 34.0], atol=1.0)


def test_fit_homography_requires_at_least_four_points():
    correspondences = [
        Correspondence(pixel=(0, 0), pitch_m=(0, 0)),
        Correspondence(pixel=(1, 1), pitch_m=(1, 1)),
    ]
    with pytest.raises(ValueError, match=">= 4"):
        fit_homography(correspondences, reproj_error_max_m=1.0)


def test_load_manual_correspondences_supports_pitch_m_and_vertex(tmp_path):
    pitch = pitch_config(105, 68)
    payload = [
        {"pixel": [10, 20], "pitch_m": [1.0, 2.0]},
        {"pixel": [30, 40], "pitch_vertex": 1},
    ]
    path = tmp_path / "keypoints.json"
    path.write_text(json.dumps(payload))

    correspondences = load_manual_correspondences(path, pitch)
    assert correspondences[0].pixel == (10, 20)
    assert correspondences[0].pitch_m == (1.0, 2.0)
    # vertex 1 (1-indexed) is pitch corner (0, 0) in meters
    assert correspondences[1].pitch_m == (0.0, 0.0)
