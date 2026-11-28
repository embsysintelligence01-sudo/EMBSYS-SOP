"""Configuration dataclasses and JSON/environment loader for Post-Cutting Inspection."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(raw: str, base_dir: Path = PROJECT_ROOT) -> str:
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    # Check base_dir first, then check parent directory (e.g. shared models in parent)
    target = (base_dir / p).resolve()
    if target.exists():
        return str(target)
    for parent in base_dir.parents:
        cand = (parent / p).resolve()
        if cand.exists():
            return str(cand)
    return str(target)


@dataclass
class ModelPathsConfig:
    model_dir: str = "models"
    packet_seg: str = "models/packet-seg.onnx"
    jewel_seg: str = "models/jeweltrack.onnx"
    hand_seg: str = "models/hand-seg.onnx"
    sam_model: str = ""
    imgsz: int = 320
    device: str = "cpu"
    onnx_providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])

    def resolved_packet_seg(self) -> Path:
        return Path(_resolve_path(self.packet_seg))

    def resolved_jewel_seg(self) -> Path:
        return Path(_resolve_path(self.jewel_seg))

    def resolved_hand_seg(self) -> Path:
        return Path(_resolve_path(self.hand_seg))

    def resolved_sam_model(self) -> Optional[Path]:
        if not self.sam_model:
            return None
        return Path(_resolve_path(self.sam_model))


@dataclass
class PacketInspectionConfig:
    packet_conf: float = 0.25
    jewel_conf: float = 0.25
    jewel_borderline_conf: float = 0.14
    hand_conf: float = 0.30
    min_packet_area_px: int = 4000
    min_packet_solidity: float = 0.60
    max_hand_occlusion_ratio: float = 0.30
    max_border_touch_edges: int = 2
    min_aspect_ratio: float = 0.22
    max_aspect_ratio: float = 4.5
    max_tilt_deviation_deg: float = 40.0
    border_inset_ratio: float = 0.12
    min_border_inset_px: int = 10
    min_interior_area_px: int = 1200
    min_interior_area_ratio: float = 0.15
    min_blur_variance: float = 25.0
    min_mean_brightness: float = 38.0
    max_mean_brightness: float = 242.0
    min_gold_area_px: int = 90
    min_gold_interior_overlap: float = 0.45
    max_specular_glare_ratio: float = 0.65
    min_gold_texture_std: float = 8.0
    cv_gold_support_area_px: int = 220
    cv_gold_uncertainty_area_px: int = 140
    temporal_confirm_frames: int = 5
    temporal_window_size: int = 7
    roi_crop_padding_ratio: float = 0.15


@dataclass
class SeparationConfig:
    enable_sam: bool = True
    min_peak_distance_px: int = 14
    peak_rel_threshold: float = 0.42
    clean_neck_ratio_max: float = 0.58
    ambiguous_neck_ratio_max: float = 0.85
    touching_solidity_max: float = 0.86
    deep_defect_depth_px: float = 7.0
    overlap_box_iou_threshold: float = 0.25
    complex_area_multiplier: float = 3.2


@dataclass
class TrackingConfig:
    min_match_iou: float = 0.12
    max_centroid_distance_px: float = 65.0
    min_track_confirm_hits: int = 2
    max_missing_frames: int = 4
    id_prefix: str = "JWL"


@dataclass
class StabilityConfig:
    required_stable_frames: int = 8
    min_stable_seconds: float = 1.5
    max_centroid_shift_px: float = 10.0
    max_area_change_ratio: float = 0.20
    max_uncertain_ratio_in_window: float = 0.25


@dataclass
class JewelleryCountingConfig:
    testbed_roi: Tuple[int, int, int, int] = (160, 120, 480, 400)
    use_normalized_roi: bool = False
    testbed_roi_normalized: Tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.8333)
    auto_detect_green_testbed: bool = False
    jewel_conf: float = 0.25
    hand_conf: float = 0.30
    packet_conf: float = 0.25
    min_roi_overlap_ratio: float = 0.50
    min_article_area_px: int = 120
    empty_testbed_confirm_frames: int = 4
    empty_max_foreground_ratio: float = 0.03
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)


@dataclass
class OutputConfig:
    output_dir: str = "packet_cut_outputs/post_cut_inspection"
    save_evidence_images: bool = True
    save_json_report: bool = True

    def resolved_output_dir(self) -> Path:
        return Path(_resolve_path(self.output_dir))


@dataclass
class InspectionConfig:
    models: ModelPathsConfig = field(default_factory=ModelPathsConfig)
    packet_inspection: PacketInspectionConfig = field(default_factory=PacketInspectionConfig)
    jewellery_counting: JewelleryCountingConfig = field(default_factory=JewelleryCountingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return target

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "InspectionConfig":
        models_raw = dict(raw.get("models", {}))
        pkt_raw = dict(raw.get("packet_inspection", {}))
        jwl_raw = dict(raw.get("jewellery_counting", {}))
        out_raw = dict(raw.get("output", {}))

        sep_raw = dict(jwl_raw.pop("separation", {}))
        trk_raw = dict(jwl_raw.pop("tracking", {}))
        stb_raw = dict(jwl_raw.pop("stability", {}))

        if "testbed_roi" in jwl_raw and isinstance(jwl_raw["testbed_roi"], list):
            jwl_raw["testbed_roi"] = tuple(int(v) for v in jwl_raw["testbed_roi"])
        if "testbed_roi_normalized" in jwl_raw and isinstance(jwl_raw["testbed_roi_normalized"], list):
            jwl_raw["testbed_roi_normalized"] = tuple(float(v) for v in jwl_raw["testbed_roi_normalized"])

        return cls(
            models=ModelPathsConfig(**models_raw),
            packet_inspection=PacketInspectionConfig(**pkt_raw),
            jewellery_counting=JewelleryCountingConfig(
                **jwl_raw,
                separation=SeparationConfig(**sep_raw),
                tracking=TrackingConfig(**trk_raw),
                stability=StabilityConfig(**stb_raw),
            ),
            output=OutputConfig(**out_raw),
        )

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "InspectionConfig":
        cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if cfg_path.is_file():
            with cfg_path.open("r", encoding="utf-8") as fh:
                cfg = cls.from_dict(json.load(fh))
        else:
            cfg = cls()

        # Optional environment overrides so deployment paths stay configurable.
        env_map = {
            "INSPECTION_PACKET_MODEL": ("packet_seg",),
            "INSPECTION_JEWEL_MODEL": ("jewel_seg",),
            "INSPECTION_HAND_MODEL": ("hand_seg",),
            "INSPECTION_SAM_MODEL": ("sam_model",),
            "INSPECTION_DEVICE": ("device",),
        }
        for env_key, (attr_name,) in env_map.items():
            val = os.environ.get(env_key)
            if val:
                setattr(cfg.models, attr_name, val)

        env_roi = os.environ.get("INSPECTION_TESTBED_ROI")
        if env_roi:
            parts = [int(p.strip()) for p in env_roi.split(",") if p.strip()]
            if len(parts) == 4:
                cfg.jewellery_counting.testbed_roi = tuple(parts)  # type: ignore[assignment]

        return cfg

