"""Configuration dataclasses and loader for the Automated Workflow Monitoring System."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_WORKFLOW_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(raw: str, base_dir: Path = PROJECT_ROOT) -> str:
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    target = (base_dir / p).resolve()
    if target.exists():
        return str(target)
    for parent in base_dir.parents:
        cand = (parent / p).resolve()
        if cand.exists():
            return str(cand)
    return str(target)


@dataclass
class WorkflowModelsConfig:
    model_dir: str = "models"
    packet_seg: str = "models/packet-seg.onnx"
    hand_seg: str = "models/hand-seg.onnx"
    scissor_seg: str = "models/scissor-seg.pt"
    topstrip_seg: str = "models/strip-seg.onnx"
    packet_number_obb: str = "models/packet-no-obb.onnx"
    cut_seg: str = "models/cut-seg.pt"
    jewel_seg: str = "models/jeweltrack.onnx"
    sam_model: str = ""
    imgsz: int = 320
    number_imgsz: int = 640
    device: str = "cpu"

    def path_for(self, key: str) -> Optional[Path]:
        mapping = {
            "packet": self.packet_seg,
            "hand": self.hand_seg,
            "scissor": self.scissor_seg,
            "topstrip": self.topstrip_seg,
            "number": self.packet_number_obb,
            "cut": self.cut_seg,
            "jewel": self.jewel_seg,
            "sam": self.sam_model,
        }
        raw = mapping.get(key, "")
        if not raw:
            return None
        return Path(_resolve_path(raw))


@dataclass
class OCRProtectionConfig:
    number_conf: float = 0.05
    min_ocr_confidence: float = 0.65
    min_reliable_reads: int = 2
    preferred_digit_length: int = 6
    min_id_length: int = 4
    ocr_attempt_interval_frames: int = 3
    max_unresolved_backside_frames_before_alert: int = 6


@dataclass
class AutomatedProcessingConfig:
    packet_conf: float = 0.12
    hand_conf: float = 0.30
    scissor_conf: float = 0.30
    topstrip_conf: float = 0.25
    cut_model_conf: float = 0.25
    cut_model_min_conf: float = 0.70
    cut_model_class_id: int = 0
    baseline_stable_frames: int = 4
    orientation_closed_confirm_frames: int = 4
    hand_confirm_frames: int = 2
    scissor_confirm_frames: int = 2
    clear_wait_seconds: float = 1.5
    cut_verify_seconds: float = 6.0
    verify_sample_interval: float = 0.5
    min_verify_samples: int = 4
    min_valid_sample_ratio: float = 0.70
    min_cut_streak: int = 3
    post_cut_unknown_alert_frames: int = 8
    auto_complete_stable_frames: int = 10
    auto_complete_min_seconds: float = 2.0
    outside_testbed_alert_frames: int = 3
    uncertain_count_alert_frames: int = 4


@dataclass
class AlertsConfig:
    suppression_cooldown_seconds: float = 8.0
    max_active_alerts: int = 20


@dataclass
class WorkflowOutputConfig:
    output_dir: str = "packet_cut_outputs/automated_sessions"
    save_evidence_images: bool = True
    save_json_report: bool = True
    save_markdown_report: bool = True

    def resolved_output_dir(self) -> Path:
        return Path(_resolve_path(self.output_dir))


@dataclass
class WorkflowMonitorConfig:
    models: WorkflowModelsConfig = field(default_factory=WorkflowModelsConfig)
    ocr_protection: OCRProtectionConfig = field(default_factory=OCRProtectionConfig)
    automated_processing: AutomatedProcessingConfig = field(default_factory=AutomatedProcessingConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    output: WorkflowOutputConfig = field(default_factory=WorkflowOutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "WorkflowMonitorConfig":
        return cls(
            models=WorkflowModelsConfig(**dict(raw.get("models", {}))),
            ocr_protection=OCRProtectionConfig(**dict(raw.get("ocr_protection", {}))),
            automated_processing=AutomatedProcessingConfig(**dict(raw.get("automated_processing", {}))),
            alerts=AlertsConfig(**dict(raw.get("alerts", {}))),
            output=WorkflowOutputConfig(**dict(raw.get("output", {}))),
        )

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "WorkflowMonitorConfig":
        cfg_path = Path(path) if path is not None else DEFAULT_WORKFLOW_CONFIG_PATH
        if cfg_path.is_file():
            with cfg_path.open("r", encoding="utf-8") as fh:
                cfg = cls.from_dict(json.load(fh))
        else:
            cfg = cls()

        env_device = os.environ.get("WORKFLOW_DEVICE")
        if env_device:
            cfg.models.device = env_device
        return cfg

