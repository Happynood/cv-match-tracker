"""Single validation point for on-disk data contracts (Appendix A of the spec).

Both ``players.parquet`` rows and ``stats.json`` are validated against the
models defined here before being written to disk.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TrackClass(StrEnum):
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"


class Team(StrEnum):
    A = "A"
    B = "B"
    UNKNOWN = "unknown"


class PlayerFrameRow(BaseModel):
    """One row of ``players.parquet``: a single track's state at a single frame."""

    model_config = ConfigDict(extra="forbid")

    frame: int = Field(ge=0)
    t_s: float = Field(ge=0)
    track_id: int
    team: Team
    cls: TrackClass
    u: float
    v: float
    x_m: float
    y_m: float
    speed_ms: float = Field(ge=0)
    calib_valid: bool


Method = Literal["geometry", "heuristic", "model"]


class MetricValue(BaseModel):
    """A single reported statistic with provenance."""

    model_config = ConfigDict(extra="forbid")

    value: float | int | list | dict
    method: Method
    confidence: float = Field(ge=0, le=1)


class TrackStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: int
    team: Team
    cls: TrackClass
    distance_m: MetricValue
    sprint_distance_m: MetricValue
    sprint_count: MetricValue
    top_speed_ms: MetricValue
    avg_speed_ms: MetricValue
    minutes_on_screen: MetricValue
    coverage: MetricValue
    avg_position_m: MetricValue


class RunManifest(BaseModel):
    """Reproducibility manifest embedded in every ``stats.json``."""

    model_config = ConfigDict(extra="forbid")

    git_sha: str | None
    git_dirty: bool
    config_sha256: str
    model_revisions: dict[str, str]
    gpu_name: str | None
    driver_version: str | None
    cuda_version: str | None
    seed: int


class MatchStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: RunManifest
    tracks: list[TrackStats]
    team_possession_pct: MetricValue | None = None
    calibration_reproj_error_m: float | None = None
