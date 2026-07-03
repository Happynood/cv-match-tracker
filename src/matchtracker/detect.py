"""Detection adapter: wraps RF-DETR, parses results with ``supervision``.

No detector is reimplemented here. ``RFDETRNano``/``RFDETRSmall`` (package
``rfdetr``) already return ``sv.Detections`` directly from ``.predict()``; this
module only handles model loading (including an optional pinned Hugging Face
checkpoint), per-class confidence thresholds, and NMS.

Football fine-tuning note: this repository does not ship a football-specific
checkpoint. Configure ``detector.hf_repo_id`` + ``detector.hf_revision`` (or
``detector.checkpoint_path``) to point at a checkpoint fine-tuned on a
football dataset (e.g. Roboflow Universe football players). Without one, this
adapter falls back to the COCO-pretrained weights and remaps COCO
``person``/``sports ball`` onto ``player``/``ball`` — sufficient for smoke
testing, not for the Tier A accuracy targets in the spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv

_COCO_TO_FOOTBALL = {
    "person": "player",
    "sports ball": "ball",
}
FOOTBALL_CLASSES = ("player", "goalkeeper", "referee", "ball")


@dataclass(frozen=True)
class DetectorConfig:
    name: str = "rfdetr_nano"
    resolution: int = 384
    fp16: bool = True
    conf: float = 0.40
    conf_ball: float = 0.15
    iou: float = 0.60
    checkpoint_path: str | None = None
    hf_repo_id: str | None = None
    hf_revision: str | None = None
    hf_filename: str | None = None

    @classmethod
    def from_mapping(cls, mapping: Any) -> DetectorConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in mapping.items() if k in known})


def _resolve_checkpoint(config: DetectorConfig) -> str | None:
    if config.checkpoint_path:
        return config.checkpoint_path
    if config.hf_repo_id:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=config.hf_repo_id,
            filename=config.hf_filename or "checkpoint_best_total.pth",
            revision=config.hf_revision,
        )
    return None


class RFDETRAdapter:
    """Thin adapter over ``rfdetr``'s ``RFDETRNano``/``RFDETRSmall`` models."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._is_finetuned = bool(config.checkpoint_path or config.hf_repo_id)
        self.model = self._load_model()
        self.model_revision = config.hf_revision or "coco-pretrained"

    def _load_model(self):
        from rfdetr import RFDETRNano, RFDETRSmall
        from rfdetr.detr import RFDETR

        checkpoint = _resolve_checkpoint(self.config)
        if checkpoint is not None:
            return RFDETR.from_checkpoint(checkpoint, resolution=self.config.resolution)

        model_cls = RFDETRSmall if "small" in self.config.name else RFDETRNano
        return model_cls(resolution=self.config.resolution)

    def _class_name(self, raw_name: str) -> str:
        if self._is_finetuned:
            return raw_name
        return _COCO_TO_FOOTBALL.get(raw_name, raw_name)

    def detect(self, image_rgb: np.ndarray) -> sv.Detections:
        """Run detection on a single RGB image, returning filtered, NMS'd Detections.

        ``detections.data["class_name"]`` is normalized to the football class
        vocabulary (``player``/``goalkeeper``/``referee``/``ball``) when
        possible.
        """
        min_conf = min(self.config.conf, self.config.conf_ball)
        result = self.model.predict(image_rgb, threshold=min_conf)
        assert isinstance(result, sv.Detections), (
            "detect() expects a single-image Detections result"
        )
        detections = result

        raw_names = detections.data.get("class_name")
        if raw_names is None:
            raw_names = np.array(["player"] * len(detections))
        class_names = np.array([self._class_name(n) for n in raw_names])

        confidence = detections.confidence
        assert confidence is not None, "RF-DETR detections always carry confidence scores"
        confidence = np.asarray(confidence, dtype=np.float64)

        is_ball = class_names == "ball"
        keep = np.where(
            is_ball, confidence >= self.config.conf_ball, confidence >= self.config.conf
        )

        filtered = detections[keep]
        assert isinstance(filtered, sv.Detections)
        detections = filtered
        detections.data["class_name"] = class_names[keep]

        if len(detections) > 0:
            nms_result = detections.with_nms(threshold=self.config.iou, class_agnostic=False)
            assert isinstance(nms_result, sv.Detections)
            detections = nms_result
        return detections


__all__ = ["DetectorConfig", "RFDETRAdapter", "FOOTBALL_CLASSES"]
