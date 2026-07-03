from typing import Any

import pytest
from pydantic import ValidationError

from matchtracker.schemas import MetricValue, PlayerFrameRow, Team, TrackClass


def _valid_row_kwargs() -> dict[str, Any]:
    return dict(
        frame=0,
        t_s=0.0,
        track_id=1,
        team=Team.A,
        cls=TrackClass.PLAYER,
        u=100.0,
        v=200.0,
        x_m=10.0,
        y_m=20.0,
        speed_ms=3.5,
        calib_valid=True,
    )


def test_player_frame_row_accepts_valid_data():
    row = PlayerFrameRow(**_valid_row_kwargs())
    assert row.team == Team.A
    assert row.cls == TrackClass.PLAYER


def test_player_frame_row_rejects_negative_speed():
    kwargs = _valid_row_kwargs()
    kwargs["speed_ms"] = -1.0
    with pytest.raises(ValidationError):
        PlayerFrameRow(**kwargs)


def test_player_frame_row_rejects_unknown_fields():
    kwargs = _valid_row_kwargs()
    kwargs["extra_field"] = "not allowed"
    with pytest.raises(ValidationError):
        PlayerFrameRow(**kwargs)


def test_player_frame_row_rejects_negative_frame():
    kwargs = _valid_row_kwargs()
    kwargs["frame"] = -1
    with pytest.raises(ValidationError):
        PlayerFrameRow(**kwargs)


def test_metric_value_confidence_must_be_in_unit_range():
    MetricValue(value=1.0, method="geometry", confidence=1.0)
    with pytest.raises(ValidationError):
        MetricValue(value=1.0, method="geometry", confidence=1.5)


def test_metric_value_rejects_unknown_method():
    with pytest.raises(ValidationError):
        MetricValue(value=1.0, method="magic", confidence=0.5)  # type: ignore[arg-type]
