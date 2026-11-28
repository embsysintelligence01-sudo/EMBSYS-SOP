"""Centralized Model Manager with Stage-Based On-Demand Inference.

Architecture:
    Load All Models Once -> Keep Models Initialized -> Execute Only Required Models Per Stage.

At startup:
- Loads all 8 models (7 vision models + PaddleOCR) ONCE into memory.
- Keeps model instances available throughout the entire application lifecycle.
- When executing a stage, strictly runs ONLY the models mapped to that active stage.
- Inactive-stage models receive ZERO inference calls.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from inspection.utils.model_loader import available_onnx_providers, matching_class_ids
from .config import WorkflowMonitorConfig

logger = logging.getLogger("ModelManager")

STAGE_MODELS: Dict[str, List[str]] = {
    "EMPTY_TESTBED": ["packet_seg"],
    "FRONT_OCR": ["packet_seg", "packet_no_obb", "paddle_ocr"],
    "BACK_STRIP": ["packet_seg", "strip_seg"],
    "CUTTING_ACTIVITY": ["packet_seg", "hand_seg", "scissor_seg"],
    "CUT_VERIFICATION": ["packet_seg", "cut_seg"],
    "MOUTH_INSPECTION": ["packet_seg", "jeweltrack"],
    "JEWELLERY_COUNT": ["packet_seg", "jeweltrack", "hand_seg"],
}


class ModelManager:
    """Centralized Model Manager enforcing single-load startup & stage-wise on-demand inference."""

    ALL_KEYS = ("packet", "hand", "scissor", "topstrip", "number", "cut", "jewel", "sam", "ocr")

    STAGE_MODELS = STAGE_MODELS

    MODEL_ALIASES: Dict[str, str] = {
        "packet_seg": "packet",
        "packet": "packet",
        "packet-seg.onnx": "packet",
        "packet_no_obb": "number",
        "number": "number",
        "packet-no-obb.onnx": "number",
        "strip_seg": "topstrip",
        "topstrip": "topstrip",
        "strip-seg.onnx": "topstrip",
        "hand_seg": "hand",
        "hand": "hand",
        "hand-seg.onnx": "hand",
        "scissor_seg": "scissor",
        "scissor": "scissor",
        "scissor-seg.pt": "scissor",
        "cut_seg": "cut",
        "cut": "cut",
        "cut-seg.pt": "cut",
        "jeweltrack": "jewel",
        "jewel": "jewel",
        "jeweltrack.onnx": "jewel",
        "paddle_ocr": "ocr",
        "ocr": "ocr",
        "PaddleOCR": "ocr",
        "sam": "sam",
    }

    STAGE_KEY_MAP: Dict[str, str] = {
        "EMPTY_TESTBED": "EMPTY_TESTBED",
        "01 · WAIT FOR EMPTY TESTBED": "EMPTY_TESTBED",
        "FRONT_OCR": "FRONT_OCR",
        "02 · WAIT FOR FRONTSIDE & OCR": "FRONT_OCR",
        "BACK_STRIP": "BACK_STRIP",
        "03 · WAIT FOR TURN & BACKSIDE STRIP": "BACK_STRIP",
        "CUTTING_ACTIVITY": "CUTTING_ACTIVITY",
        "04 · WAIT FOR CUTTING ACTIVITY": "CUTTING_ACTIVITY",
        "04 · CUTTING ACTIVITY": "CUTTING_ACTIVITY",
        "CUT_VERIFICATION": "CUT_VERIFICATION",
        "05 · VERIFY PHYSICAL CUT": "CUT_VERIFICATION",
        "05 · VERIFY THE CUT": "CUT_VERIFICATION",
        "MOUTH_INSPECTION": "MOUTH_INSPECTION",
        "06 · WAIT FOR UPRIGHT MOUTH": "MOUTH_INSPECTION",
        "JEWELLERY_COUNT": "JEWELLERY_COUNT",
        "07 · WAIT FOR JEWELLERY & COUNT": "JEWELLERY_COUNT",
        "07A · WAIT FOR GOLD PLACEMENT": "JEWELLERY_COUNT",
        "07B · COUNT JEWELLERY": "JEWELLERY_COUNT",
    }

    def __init__(
        self,
        config: Optional[WorkflowMonitorConfig] = None,
        preloaded_models: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config if config is not None else WorkflowMonitorConfig.load()
        self.device = self._resolve_device(self.config.models.device)
        self.models: Dict[str, Any] = {}
        self.class_ids: Dict[str, Optional[List[int]]] = {}
        self.missing: Dict[str, str] = {}
        self.loaded: Dict[str, str] = {}
        self.onnx_providers: List[str] = available_onnx_providers()

        self._all_loaded: bool = False
        self.init_duration_sec: float = 0.0
        self.inference_counts: Dict[str, int] = {
            "packet_seg": 0,
            "packet_no_obb": 0,
            "strip_seg": 0,
            "hand_seg": 0,
            "scissor_seg": 0,
            "cut_seg": 0,
            "jeweltrack": 0,
            "paddle_ocr": 0,
        }
        self.inference_durations: Dict[str, float] = {}

        if preloaded_models:
            for k, m in preloaded_models.items():
                if m is not None:
                    canonical = self.MODEL_ALIASES.get(k, k)
                    self.models[canonical] = m
                    self.loaded[canonical] = "preloaded"
            self._refresh_class_ids()

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

    # ==================================================================
    # 1. STARTUP MODEL LOADING — LOAD ALL 8 MODELS ONCE
    # ==================================================================
    def load_all_models(self, keys: Sequence[str] = ALL_KEYS) -> Dict[str, Any]:
        """Initialize all 8 models once at startup and keep them in memory."""
        start_mono = time.monotonic()
        logger.info("[ModelManager] Loading all models at startup...")

        # 1. Vision models (ONNX / PyTorch)
        for key in keys:
            if key == "ocr":
                continue
            if key in self.models and self.models[key] is not None:
                continue
            path = self.config.models.path_for(key)
            if path is None:
                if key == "sam":
                    self.missing[key] = "Optional SAM weights not configured."
                continue
            if not path.is_file():
                self.missing[key] = f"Model file not found: {path}"
                logger.warning("[ModelManager] Missing model '%s': %s", key, path)
                continue
            try:
                load_start = time.monotonic()
                if key == "sam":
                    from ultralytics import SAM  # type: ignore

                    self.models[key] = SAM(str(path))
                elif key == "number":
                    from ultralytics import YOLO  # type: ignore

                    self.models[key] = YOLO(str(path))
                else:
                    from ultralytics import YOLO  # type: ignore

                    self.models[key] = YOLO(str(path), task="segment")
                load_sec = time.monotonic() - load_start
                self.loaded[key] = str(path)
                logger.info("[ModelManager] Loaded '%s' from %s in %.2fs", key, path.name, load_sec)
            except Exception as exc:
                self.missing[key] = f"Failed to load {key}: {type(exc).__name__}: {exc}"
                logger.error("[ModelManager] Failed to load model '%s': %s", key, exc)

        # 2. PaddleOCR engine
        if "ocr" not in self.models and ("ocr" in keys or "paddle_ocr" in keys):
            try:
                ocr_start = time.monotonic()
                from packet_cut.desktop.ocr import get_number_recognizer

                ocr_eng = get_number_recognizer()
                if ocr_eng is not None:
                    self.models["ocr"] = ocr_eng
                    self.loaded["ocr"] = "PaddleOCR mobile recognition (number crop)"
                    ocr_sec = time.monotonic() - ocr_start
                    logger.info("[ModelManager] Loaded 'PaddleOCR' in %.2fs", ocr_sec)
            except Exception as exc:
                self.missing["ocr"] = f"Failed to initialize PaddleOCR: {exc}"
                logger.warning("[ModelManager] Failed to initialize PaddleOCR: %s", exc)

        self._refresh_class_ids()
        self.init_duration_sec = time.monotonic() - start_mono
        self._all_loaded = True
        logger.info(
            "[ModelManager] Startup initialization completed in %.2fs (%d models loaded, %d missing).",
            self.init_duration_sec,
            len(self.loaded),
            len(self.missing),
        )
        return self.availability_report()

    def ensure_loaded(self, keys: Sequence[str] = ALL_KEYS) -> Dict[str, Any]:
        """Backward-compatible loader."""
        return self.load_all_models(keys)

    def _refresh_class_ids(self) -> None:
        keyword_map = {
            "packet": ("packet", "bag"),
            "hand": ("hand",),
            "scissor": ("scissor",),
            "topstrip": ("topstrip", "strip", "seal", "top"),
            "jewel": ("gold", "jewel", "ornament", "ring", "bangle", "chain"),
        }
        for key, kws in keyword_map.items():
            model = self.models.get(key)
            self.class_ids[key] = matching_class_ids(model, kws) if model is not None else None

    # ==================================================================
    # 2. MODEL ACCESSORS
    # ==================================================================
    def get_model(self, model_name: str) -> Any:
        """Return the initialized model instance by name or alias."""
        key = self.MODEL_ALIASES.get(model_name, model_name)
        return self.models.get(key)

    def get(self, key: str) -> Any:
        """Alias for get_model (backward compatibility)."""
        return self.get_model(key)

    def has(self, key: str) -> bool:
        """Check if model instance is loaded and available."""
        return self.get_model(key) is not None

    def get_class_ids(self, key: str) -> Optional[List[int]]:
        canonical = self.MODEL_ALIASES.get(key, key)
        return self.class_ids.get(canonical)

    def _resolve_stage_name(self, stage: Union[str, Any]) -> str:
        s = stage.value if hasattr(stage, "value") else str(stage)
        return self.STAGE_KEY_MAP.get(s, s)

    def get_required_models(self, stage: Union[str, Any]) -> List[str]:
        """Return the list of required models for the specified stage."""
        canonical = self._resolve_stage_name(stage)
        return list(self.STAGE_MODELS.get(canonical, []))

    # ==================================================================
    # 3. STAGE-WISE ON-DEMAND INFERENCE DISPATCHER
    # ==================================================================
    def run_stage(
        self,
        stage: Union[str, Any],
        frame: np.ndarray,
        *,
        crop: Optional[np.ndarray] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute ONLY the models required by the specified active stage.

        Guarantees:
        - Never initializes models here (all models are loaded once at startup).
        - Inactive-stage models receive ZERO inference calls.
        - Records execution duration and increments inference count for active models.
        """
        ov = overrides or {}
        canonical = self._resolve_stage_name(stage)
        required_models = self.STAGE_MODELS.get(canonical, [])
        results: Dict[str, Any] = {}

        for model_name in required_models:
            start_mono = time.monotonic()
            key = self.MODEL_ALIASES.get(model_name, model_name)
            model = self.models.get(key)

            self.inference_counts[model_name] = self.inference_counts.get(model_name, 0) + 1

            if model is None:
                results[model_name] = None
                continue

            if model_name == "packet_seg":
                if "packet_mask" in ov:
                    results[model_name] = ov["packet_mask"]
                else:
                    results[model_name] = model.predict(
                        source=frame,
                        imgsz=self.config.models.imgsz,
                        conf=0.12,
                        device=self.device,
                        verbose=False,
                    )[0]

            elif model_name == "packet_no_obb":
                if "ocr_roi" in ov:
                    results[model_name] = ov["ocr_roi"]
                else:
                    results[model_name] = model.predict(
                        source=frame,
                        imgsz=self.config.models.number_imgsz,
                        conf=0.25,
                        device=self.device,
                        verbose=False,
                    )[0]

            elif model_name == "paddle_ocr":
                if "ocr_text" in ov:
                    results[model_name] = ov["ocr_text"]
                elif crop is not None and crop.size > 0:
                    from packet_cut.desktop.ocr import run_ocr

                    results[model_name] = run_ocr(crop)
                else:
                    results[model_name] = (None, 0.0, [])

            elif model_name == "strip_seg":
                if "topstrip_visible" in ov:
                    results[model_name] = ov["topstrip_visible"]
                else:
                    results[model_name] = model.predict(
                        source=frame,
                        imgsz=self.config.models.imgsz,
                        conf=0.25,
                        device=self.device,
                        verbose=False,
                    )[0]

            elif model_name == "hand_seg":
                if "hand_mask" in ov or "hand_present" in ov:
                    results[model_name] = ov.get("hand_mask", ov.get("hand_present"))
                else:
                    results[model_name] = model.predict(
                        source=frame,
                        imgsz=self.config.models.imgsz,
                        conf=0.30,
                        device=self.device,
                        verbose=False,
                    )[0]

            elif model_name == "scissor_seg":
                if "scissor_mask" in ov or "scissor_present" in ov:
                    results[model_name] = ov.get("scissor_mask", ov.get("scissor_present"))
                else:
                    h_f, w_f = frame.shape[:2]
                    f_sc = frame.copy()
                    # Mask out bottom extrusion rail (y >= 83% of height) to prevent spurious scissor detection
                    f_sc[int(h_f * 0.8333):, :] = 0
                    results[model_name] = model.predict(
                        source=f_sc,
                        imgsz=self.config.models.imgsz,
                        conf=0.30,
                        device=self.device,
                        verbose=False,
                    )[0]

            elif model_name == "cut_seg":
                if "cut_eval" in ov:
                    results[model_name] = ov["cut_eval"]
                else:
                    results[model_name] = model.predict(
                        source=frame,
                        imgsz=self.config.models.imgsz,
                        conf=0.70,
                        device=self.device,
                        verbose=False,
                    )[0]

            elif model_name == "jeweltrack":
                if "jewel_instances" in ov or "mouth_jewel_instances" in ov:
                    results[model_name] = ov.get("jewel_instances", ov.get("mouth_jewel_instances"))
                else:
                    results[model_name] = model.predict(
                        source=frame,
                        imgsz=self.config.models.imgsz,
                        conf=0.25,
                        device=self.device,
                        verbose=False,
                    )[0]

            duration_ms = (time.monotonic() - start_mono) * 1000.0
            self.inference_durations[model_name] = duration_ms

        return results

    def run_models_for_stage(
        self,
        stage: Union[str, Any],
        frame: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convenience alias for run_stage."""
        return self.run_stage(stage, frame, **kwargs)

    # ==================================================================
    # 4. LIFECYCLE RESOURCE MANAGEMENT
    # ==================================================================
    def unload_all_models(self) -> None:
        """Release all model resources and free memory during application shutdown."""
        logger.info("[ModelManager] Releasing model resources on shutdown...")
        self.models.clear()
        self.loaded.clear()
        self.class_ids.clear()
        self._all_loaded = False
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[ModelManager] All models unloaded.")

    def post_cut_shared_models(self) -> Dict[str, Any]:
        """Return ONLY models permitted for post-cut inspection (strictly excludes TopStrip)."""
        allowed = ("packet", "jewel", "hand", "sam")
        return {k: self.models[k] for k in allowed if k in self.models and self.models[k] is not None}

    def availability_report(self) -> Dict[str, Any]:
        return {
            "loaded": dict(self.loaded),
            "missing": dict(self.missing),
            "device": str(self.device),
            "onnx_providers": list(self.onnx_providers),
            "init_duration_sec": round(self.init_duration_sec, 2),
            "inference_counts": dict(self.inference_counts),
        }


# Backward-compatible alias
UnifiedModelManager = ModelManager
