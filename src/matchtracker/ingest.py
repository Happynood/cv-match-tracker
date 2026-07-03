"""Video ingest adapter: decode + sample frames with PyAV, persist the run manifest.

Thin wrapper — all heavy lifting (demuxing/decoding) is done by PyAV (``av``) and
frame resizing by OpenCV. Nothing here reimplements a codec or a sampler beyond
simple stride selection.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from matchtracker.schemas import RunManifest


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    duration_s: float


@dataclass(frozen=True)
class Frame:
    frame_idx: int
    """Index into the original video's frame sequence."""
    sample_idx: int
    """Sequential index among sampled frames (0, 1, 2, ...)."""
    t_s: float
    image: np.ndarray


def probe_video(path: str | Path) -> VideoInfo:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        duration_s = (
            float(stream.duration * stream.time_base)
            if stream.duration and stream.time_base
            else 0.0
        )
        return VideoInfo(
            path=str(path),
            fps=fps,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            duration_s=duration_s,
        )


def iter_frames(
    path: str | Path,
    target_fps: float | None = None,
    resize: tuple[int, int] | None = None,
    clip: tuple[float, float] | None = None,
) -> Iterator[Frame]:
    """Decode ``path`` and yield frames sampled at (approximately) ``target_fps``.

    Args:
        path: video file path.
        target_fps: desired effective sampling rate. ``None`` keeps every frame.
        resize: optional ``(width, height)`` to resize each frame to.
        clip: optional ``(start_s, end_s)`` window to restrict decoding to.
    """
    info = probe_video(path)
    source_fps = info.fps or 25.0
    stride = max(1, round(source_fps / target_fps)) if target_fps else 1

    start_s, end_s = clip if clip else (0.0, None)
    sample_idx = 0

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame_idx, av_frame in enumerate(container.decode(stream)):
            t_s = (
                float(av_frame.pts * stream.time_base)
                if av_frame.pts is not None and stream.time_base is not None
                else (frame_idx / source_fps)
            )
            if t_s < start_s:
                continue
            if end_s is not None and t_s > end_s:
                break
            if frame_idx % stride != 0:
                continue

            image = av_frame.to_ndarray(format="bgr24")
            if resize is not None:
                image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)

            yield Frame(frame_idx=frame_idx, sample_idx=sample_idx, t_s=t_s, image=image)
            sample_idx += 1


def _git_info() -> tuple[str | None, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def _gpu_info() -> tuple[str | None, str | None, str | None]:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            cuda_version = torch.version.cuda
            return name, driver, cuda_version
    except Exception:
        pass
    return None, None, None


def config_sha256(config: Any) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_run_manifest(
    config: Any,
    model_revisions: dict[str, str],
    seed: int = 0,
) -> RunManifest:
    git_sha, git_dirty = _git_info()
    gpu_name, driver_version, cuda_version = _gpu_info()
    return RunManifest(
        git_sha=git_sha,
        git_dirty=git_dirty,
        config_sha256=config_sha256(config),
        model_revisions=model_revisions,
        gpu_name=gpu_name,
        driver_version=driver_version,
        cuda_version=cuda_version,
        seed=seed,
    )


def write_run_manifest(manifest: RunManifest, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "run_manifest.json"
    out_path.write_text(manifest.model_dump_json(indent=2))
    return out_path


__all__ = [
    "Frame",
    "VideoInfo",
    "probe_video",
    "iter_frames",
    "config_sha256",
    "build_run_manifest",
    "write_run_manifest",
]
