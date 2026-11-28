"""Shared model registry for Post-Cutting Inspection.

Reuses already-loaded models from the host Packet-Cut application whenever
supplied, avoids duplicate model loading, supports available ONNX Runtime
execution providers, and degrades gracefully when optional models are absent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .inspection_config import InspectionConfig

logger = logging.getLogger(__name__)


def available_onnx_providers() -> List[str]:
    try:
        import onnxruntime as ort  # type: ignore

        return list(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]


def class_names_dict(model: Any) -> Dict[int, str]:
    names = getattr(model, "names", None)
    if names is None:
        return {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(name) for i, name in enumerate(names)}


def matching_class_ids(model: Any, keywords: Sequence[str]) -> Optional[List[int]]:
    found: List[int] = []
    for class_id, name in class_names_dict(model).items():
        normalized = re.sub(r"[^a-z0-9]", "", str(name).lower())
        if any(k in normalized for k in keywords):
            found.append(int(class_id))
    return found or None


@dataclass
class ModelAvailabilityReport:
    loaded: Dict[str, str] = field(default_factory=dict)
    reused: List[str] = field(default_factory=list)
    missing: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    onnx_providers: List[str] = field(default_factory=available_onnx_providers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loaded": dict(self.loaded),
            "reused": list(self.reused),
            "missing": dict(self.missing),
            "warnings": list(self.warnings),
            "onnx_providers": list(self.onnx_providers),
        }


class InspectionModelRegistry:
    """Caches and exposes models needed by Post-Cutting Inspection.

    Explicitly excludes TopStrip (`strip-seg.onnx`) by design.
    """

    def __init__(
        self,
        config: InspectionConfig,
        shared_models: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self.device = self._resolve_device(config.models.device)
        self.models: Dict[str, Any] = {}
        self.class_ids: Dict[str, Optional[List[int]]] = {
            "packet": None,
            "jewel": None,
            "hand": None,
        }
        self.report = ModelAvailabilityReport()

        if shared_models:
            self._adopt_shared_models(shared_models)

    @staticmethod
    def _resolve_device(requested: str) -> Any:
        if str(requested).lower() == "cpu":
            return "cpu"
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                return "cpu"
            return int(requested) if str(requested).isdigit() else requested
        except Exception:
            return "cpu"

    def _adopt_shared_models(self, shared_models: Dict[str, Any]) -> None:
        # Never adopt TopStrip model for post-cut inspection.
        for key in ("packet", "jewel", "hand", "sam"):
            if key in shared_models and shared_models[key] is not None:
                self.models[key] = shared_models[key]
                self.report.reused.append(key)
                self.report.loaded[key] = "reused_from_host"

        self._refresh_class_ids()

    def _refresh_class_ids(self) -> None:
        if "packet" in self.models and self.models["packet"] is not None:
            self.class_ids["packet"] = matching_class_ids(
                self.models["packet"], ("packet", "bag")
            )
        if "jewel" in self.models and self.models["jewel"] is not None:
            self.class_ids["jewel"] = matching_class_ids(
                self.models["jewel"], ("gold", "jewel", "ornament", "ring", "bangle", "chain")
            )
        if "hand" in self.models and self.models["hand"] is not None:
            self.class_ids["hand"] = matching_class_ids(
                self.models["hand"], ("hand",)
            )

    def ensure_loaded(self, keys: Sequence[str] = ("packet", "jewel", "hand", "sam")) -> ModelAvailabilityReport:
        """Load any requested models not already provided via shared_models."""
        path_map: Dict[str, Optional[Path]] = {
            "packet": self.config.models.resolved_packet_seg(),
            "jewel": self.config.models.resolved_jewel_seg(),
            "hand": self.config.models.resolved_hand_seg(),
            "sam": self.config.models.resolved_sam_model(),
        }

        for key in keys:
            if key in self.models and self.models[key] is not None:
                continue

            model_path = path_map.get(key)
            if model_path is None:
                if key == "sam":
                    self.report.missing["sam"] = (
                        "Optional SAM2.1 / MobileSAM model path not configured; "
                        "using OpenCV distance-transform & watershed separator."
                    )
                continue

            if not model_path.is_file():
                msg = f"Model file not found for '{key}': {model_path}"
                self.report.missing[key] = msg
                logger.warning(msg)
                continue

            try:
                if key == "sam":
                    from ultralytics import SAM  # type: ignore

                    self.models[key] = SAM(str(model_path))
                else:
                    from ultralytics import YOLO  # type: ignore

                    self.models[key] = YOLO(str(model_path), task="segment")
                self.report.loaded[key] = str(model_path)
            except Exception as exc:
                msg = f"Failed to load '{key}' model ({model_path}): {type(exc).__name__}: {exc}"
                self.report.warnings.append(msg)
                self.report.missing[key] = msg
                logger.warning(msg)

        self._refresh_class_ids()
        return self.report

    def get(self, key: str) -> Any:
        return self.models.get(key)

    def get_class_ids(self, key: str) -> Optional[List[int]]:
        return self.class_ids.get(key)

    def has(self, key: str) -> bool:
        return self.models.get(key) is not None

