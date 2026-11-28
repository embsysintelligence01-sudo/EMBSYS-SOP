"""Packet release inspection with independent observations by default.

Pass-by mode evaluates evidence checks independently, accepts late observations,
retains clear count evidence across removal, and saves incomplete checks honestly.
No check advances or blocks another check. The strict sequential policy remains
available with AutoPipelineConfig(pass_by=False) for compatibility and regression
coverage; its physical verification rules are reused by the release observers.
"""

from __future__ import annotations

import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

import packet_cut_core
from inspection.jewellery_counting.instance_separator import InstanceSeparator
from inspection.jewellery_counting.jewel_detector import JewelDetector
from inspection.jewellery_counting.object_tracker import JewelleryObjectTracker
from inspection.jewellery_counting.testbed_roi import TestbedROIManager
from inspection.utils.image_utils import (
    RawSegInstance,
    bbox_iou,
    draw_mask_overlay,
    evaluate_frame_quality,
    extract_yolo_instances,
    mask_centroid_and_bbox,
    union_masks,
)
from inspection.utils.inspection_config import InspectionConfig
from inspection.utils.model_loader import InspectionModelRegistry
from packet_cut.cut_model import evaluate_cut_model_prediction
from workflow_monitor.activity_log import ActivityEvent, ActivityLogger, ComplianceStatus
from workflow_monitor.alerts import AlertRecord, AlertSeverity, RealTimeAlertManager, utc_now_iso
from workflow_monitor.config import WorkflowMonitorConfig
from workflow_monitor.models import UnifiedModelManager
from workflow_monitor.ocr_protector import OCRCallable
from workflow_monitor.procedure_monitor import AppraiserProcedureMonitor
from workflow_monitor.report_generator import FinalReportGenerator
from workflow_monitor.session_media import SessionMediaRecorder

from . import config as desk_cfg
from .state_machine import AutoStage, EventDrivenStateMachine, STATE_MACHINE_SPECS
from .release_observer import ReleaseObserver


BACKSIDE_CLASSIFIER_LIMITATION = (
    "Project models/ contains no dedicated front-vs-back packet classifier "
    "(only packet-seg.onnx, packet-no-obb.onnx, strip-seg.onnx). Backside verification "
    "therefore requires observed physical turn + hand-free stationary view + absent "
    "number box + surface appearance change from frontside (or positive TopStrip detection); "
    "otherwise backside remains UNKNOWN."
)

MISSING_COMPLETION_EVENT_NOTE = (
    "No observable Stage-7 workflow completion event (e.g. empty-packet confirmation or "
    "testbed clearance signal) is defined in the existing EMBSYS Packet-Cut codebase. "
    "Therefore the session remains ACTIVE with a PROVISIONAL jewellery count across pauses "
    "until STOP is pressed to save the full report and evidence."
)


@dataclass
class AutoPipelineConfig:
    pass_by: bool = True
    pass_by_count_frames: int = 3
    number_check_interval_sec: float = 0.35
    pass_by_ocr_stable_frames: int = 2
    strip_check_interval_sec: float = 0.5
    scissor_check_interval_sec: float = 0.3
    jewel_check_interval_sec: float = 0.25
    mouth_check_interval_sec: float = 0.5
    infer_max_retries: int = desk_cfg.AUTO_INFER_MAX_RETRIES
    infer_backoff_sec: float = desk_cfg.AUTO_INFER_BACKOFF_SEC

    # Stage 1: Empty testbed
    empty_testbed_frames: int = 5
    empty_testbed_timeout_sec: float = 15.0

    # Stage 2: Frontside + OCR
    frontside_min_conf: float = 0.20
    frontside_min_box_area: int = 250
    frontside_min_packet_overlap: float = 0.50
    frontside_stable_frames: int = 3
    ocr_min_crop_sharpness: float = 18.0
    ocr_min_confidence: float = 0.65
    ocr_min_consistent_crops: int = 2
    ocr_min_id_length: int = 4
    ocr_turn_early_frames: int = 3
    frontside_timeout_sec: float = 30.0

    # Stage 3: Turn + Backside Strip
    max_stationary_shift_px: float = 12.0
    max_stationary_area_ratio: float = 0.14
    min_turn_motion_px: float = 14.0
    min_backside_appearance_diff: float = 0.18
    backside_confirm_frames: int = 4
    backside_timeout_sec: float = 30.0

    # Stage 4: Cutting activity
    hand_confirm_frames: int = packet_cut_core.HAND_CONFIRM_FRAMES
    scissor_confirm_frames: int = packet_cut_core.SCISSOR_CONFIRM_FRAMES
    cutting_both_confirm_frames: int = 2
    cutting_proximity_px: float = 240.0
    cutting_timeout_sec: float = 45.0

    # Stage 5: Verify physical cut
    cut_confirm_frames: int = 4
    cut_analysis_frames: int = 10
    cut_sample_interval_sec: float = packet_cut_core.VERIFY_SAMPLE_INTERVAL
    cut_min_conf: float = desk_cfg.CUT_MODEL_MIN_CONF
    cut_timeout_sec: float = 25.0

    # Stage 6: Upright mouth inspection
    vertical_lift_shift_px: float = 28.0
    mouth_confirm_frames: int = 4
    mouth_analysis_frames: int = 10
    mouth_min_focus_var: float = 25.0
    mouth_min_open_width_px: float = 12.0
    mouth_min_recess_contrast: float = 22.0
    mouth_min_cavity_area_px: int = 300
    mouth_max_closed_width_px: float = 6.0
    mouth_timeout_sec: float = 30.0

    # Stage 7: Jewellery counting
    jewel_min_track_hits: int = 2
    jewel_max_missing_frames: int = 4
    jewel_max_move_px: float = 140.0
    outside_testbed_alert_frames: int = 3
    uncertain_count_alert_frames: int = 3

    save_outputs: bool = True
    output_dir: str = str(desk_cfg.OUTPUT_DIR / "automated_sessions")
    visualization: bool = True


@dataclass
class RecoveryStats:
    total_inference_errors: int = 0
    total_retries: int = 0
    recovered_errors: int = 0
    ocr_attempts: int = 0
    ocr_failed_attempts: int = 0
    timeout_alerts_emitted: int = 0
    last_error: str = "None"
    last_recovery: str = "Nominal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_inference_errors": self.total_inference_errors,
            "total_retries": self.total_retries,
            "recovered_errors": self.recovered_errors,
            "ocr_attempts": self.ocr_attempts,
            "ocr_failed_attempts": self.ocr_failed_attempts,
            "timeout_alerts_emitted": self.timeout_alerts_emitted,
            "last_error": self.last_error,
            "last_recovery": self.last_recovery,
        }


@dataclass
class AutoStepSnapshot:
    session_id: str
    stage: AutoStage
    active_model: str
    models_executed_this_frame: List[str]
    instruction: str
    annotated_frame: np.ndarray
    live_monitoring: Dict[str, Any]
    compliance_monitoring: Dict[str, Any]
    system_monitoring: Dict[str, Any]
    transitioned_this_frame: bool = False
    completed: bool = False
    final_report: Optional[Dict[str, Any]] = None
    report_paths: Dict[str, Optional[str]] = field(default_factory=dict)


class AutomatedPacketPipeline:
    """Strict one-state-per-fresh-frame event-driven pipeline for MainWindow."""

    def __init__(
        self,
        model_manager: Optional[UnifiedModelManager] = None,
        pipeline_config: Optional[AutoPipelineConfig] = None,
        workflow_config: Optional[WorkflowMonitorConfig] = None,
        inspection_config: Optional[InspectionConfig] = None,
        ocr_runner: Optional[OCRCallable] = None,
        on_alert: Optional[Callable[[AlertRecord], None]] = None,
        on_alert_resolved: Optional[Callable[[AlertRecord], None]] = None,
        on_activity: Optional[Callable[[ActivityEvent], None]] = None,
    ) -> None:
        self.pipe_cfg = pipeline_config if pipeline_config is not None else AutoPipelineConfig()
        self.wf_cfg = workflow_config if workflow_config is not None else WorkflowMonitorConfig.load()
        self.insp_cfg = inspection_config if inspection_config is not None else InspectionConfig.load()

        # Sync Stage 1 & Stage 7 configs
        self.insp_cfg.jewellery_counting.empty_testbed_confirm_frames = self.pipe_cfg.empty_testbed_frames
        self.insp_cfg.jewellery_counting.tracking.min_track_confirm_hits = self.pipe_cfg.jewel_min_track_hits
        self.insp_cfg.jewellery_counting.tracking.max_missing_frames = self.pipe_cfg.jewel_max_missing_frames
        self.insp_cfg.jewellery_counting.tracking.max_centroid_distance_px = self.pipe_cfg.jewel_max_move_px

        self.model_manager = model_manager if model_manager is not None else UnifiedModelManager(self.wf_cfg)
        self.post_cut_registry = InspectionModelRegistry(
            self.insp_cfg,
            shared_models=self.model_manager.post_cut_shared_models(),
        )

        self._ocr_runner = ocr_runner
        self._lazy_ocr_loaded = False
        self._on_alert = on_alert
        self._on_alert_resolved = on_alert_resolved
        self._on_activity = on_activity

        self.testbed_roi = TestbedROIManager(self.insp_cfg)
        self._has_calibrated_empty_reference = False
        self.jewel_detector = JewelDetector(self.insp_cfg, self.post_cut_registry)
        self.jewel_separator = InstanceSeparator(self.insp_cfg, self.post_cut_registry)
        self.jewel_tracker = JewelleryObjectTracker(self.insp_cfg)

        self.state_machine = EventDrivenStateMachine()
        self.session_id: str = ""
        self._media_recorder = None
        self.started_at: str = ""
        self.started_mono: float = 0.0
        self.stage_entered_mono: float = 0.0
        self.camera_status: str = "IDLE"
        self.recovery = RecoveryStats()

        self.alert_manager: Optional[RealTimeAlertManager] = None
        self.activity_logger: Optional[ActivityLogger] = None
        self.procedure_monitor: Optional[AppraiserProcedureMonitor] = None

        self._reset_session_state()

    @property
    def stage(self) -> AutoStage:
        return getattr(self, "_observation_context", None) or self.state_machine.stage

    def _reset_session_state(self) -> None:
        self._observer = None
        self._observation_context = None
        self._frame_idx: int = 0
        self._instruction: str = STATE_MACHINE_SPECS[AutoStage.IDLE].display_instruction
        self._packet_id: str = "UNRESOLVED"
        self._identity_status: str = "UNRESOLVED"
        self._ocr_confidence: float = 0.0
        self._ocr_evidence: Optional[str] = None
        self._ocr_crop_evidence: Optional[str] = None
        self._ocr_timing = {"number_detector_calls": 0, "number_detector_total_ms": 0.0,
                            "ocr_calls": 0, "ocr_total_ms": 0.0, "last_ocr_ms": 0.0}
        self._packet_det_summary: str = "Waiting for START"
        self._latest_observed_action: str = "Idle"

        self._empty_verified: bool = False
        self._frontside_verified: bool = False
        self._frontside_ref_hist: Optional[np.ndarray] = None
        self._frontside_grid_density: float = 0.0
        self._frontside_centroid: Optional[Tuple[float, float]] = None

        self._backside_verified: bool = False
        self._strip_result: str = "NOT_CHECKED"
        self._intact_baseline_mask: Optional[np.ndarray] = None
        self._intact_baseline_area: float = 0.0

        self._cutting_activity_observed: bool = False
        self._cutting_evidence = None
        self._cut_evidence_path = None
        self._cut_evidence_score = -1
        self._early_cutting_reported = False
        self._cut_model_filename: str = self._resolve_cut_model_filename()
        self._cut_result: Dict[str, Any] = {
            "state": "NOT_STARTED",
            "cut_model_filename": self._cut_model_filename,
            "pieces_count": 0,
            "two_pieces_confirmed": False,
            "min_gap_px": 0.0,
            "cut_model_confidence": 0.0,
            "reason": "Cutting not yet verified.",
        }
        self._postcut_resting_centroids: List[Tuple[float, float]] = []

        # Decoupled mouth status vs. gold visibility (Requirement #6)
        self._mouth_status: str = "UNKNOWN"  # OPEN | CLOSED | UNKNOWN
        self._mouth_reason: str = "Not inspected yet."
        self._mouth_evidence: Optional[str] = None
        self._gold_visibility_in_mouth: Optional[bool] = None
        self._gold_mouth_confidence: float = 0.0

        # Stage 7 provisional jewellery counting (Requirement #7)
        self._provisional_jewel_count: int = 0
        self._confirmed_unambiguous_count: int = 0
        self._jewel_count_status: str = "NOT_STARTED"  # NOT_STARTED | WAITING_PLACEMENT | PROVISIONAL | PROVISIONAL_WITH_UNCERTAINTY
        self._jewel_uncertainty_recorded: bool = False
        self._jewel_uncertainty_reasons: List[str] = []
        self._jewel_evidence: Optional[str] = None
        self._jewel_evidence_count: Optional[int] = None
        self._jewel_evidence_status: Optional[str] = None
        self._count_candidate: int = 0
        self._count_stable_streak: int = 0

        self._observed_mistakes: List[Dict[str, Any]] = []
        self._unverified_actions: List[Dict[str, Any]] = []
        self._evidence_references: List[Dict[str, str]] = []
        self._last_timeout_alert_mono: float = -1e9
        self._final_report: Optional[Dict[str, Any]] = None
        self._report_paths: Dict[str, Optional[str]] = {}
        self._clear_stage_counters()

    def retry_current_stage(self, expected_stage: str) -> None:
        if self.stage.value != expected_stage or self.stage in {AutoStage.IDLE, AutoStage.STOPPED_OR_SAVED}:
            return
        if self._observer is not None:
            self._observer.counters.clear()
            return
        self._clear_stage_counters()
        if self.stage == AutoStage.WAIT_EMPTY_TESTBED:
            self.testbed_roi.reset()
        elif self.stage in {AutoStage.WAIT_GOLD_PLACEMENT, AutoStage.WAIT_JEWELLERY_AND_COUNT}:
            self.jewel_tracker.reset()
            self._provisional_jewel_count = 0
            self._confirmed_unambiguous_count = 0
            self._jewel_count_status = "WAITING_PLACEMENT"
            self._jewel_evidence = None
            self._jewel_evidence_count = None
            if self._media_recorder is not None:
                self._media_recorder.clear_event("jewellery_count")
            self._count_candidate = 0
            self._count_stable_streak = 0

    def _clear_stage_counters(self) -> None:
        """Reset all per-stage evidence & debounce counters on every transition."""
        self._empty_streak: int = 0
        self._frontside_streak: int = 0
        self._frontside_seen_once: bool = False
        self._prev_pkt_centroid: Optional[Tuple[float, float]] = None
        self._prev_pkt_area: float = 0.0
        self._ocr_votes: Counter[str] = Counter()
        self._ocr_best_conf: Dict[str, float] = {}
        self._ocr_missing_streak: int = 0

        self._turn_motion_observed: bool = False
        self._backside_ready_streak: int = 0

        self._hand_streak: int = 0
        self._scissor_streak: int = 0
        self._cutting_both_streak: int = 0

        self._next_cut_sample_mono: float = 0.0
        self._cut_valid_streak: int = 0
        self._cut_analysis_count = 0
        self._cut_valid_samples_in_batch: int = 0
        self._cut_prev_sig: Optional[Any] = None

        self._mouth_open_streak: int = 0
        self._mouth_analysis_count = 0
        self._mouth_open_samples_in_batch: int = 0
        self._mouth_closed_samples_in_batch: int = 0
        self._batch_gold_seen: bool = False
        self._open_streak_gold_confidence = 0.0
        self._mouth_closed_streak: int = 0

        self._outside_testbed_streak: int = 0
        self._uncertain_count_streak: int = 0
        self._count_candidate: int = 0
        self._count_stable_streak: int = 0

    def _resolve_cut_model_filename(self) -> str:
        loaded = self.model_manager.loaded.get("cut")
        if loaded and loaded != "preloaded":
            return Path(loaded).name
        cfg_p = self.wf_cfg.models.path_for("cut")
        if cfg_p is not None:
            return cfg_p.name
        return desk_cfg.CUT_MODEL_PATH.name

    def refresh_shared_models(self) -> None:
        self._cut_model_filename = self._resolve_cut_model_filename()
        self._cut_result["cut_model_filename"] = self._cut_model_filename
        self.post_cut_registry = InspectionModelRegistry(
            self.insp_cfg,
            shared_models=self.model_manager.post_cut_shared_models(),
        )
        self.jewel_detector = JewelDetector(self.insp_cfg, self.post_cut_registry)
        self.jewel_separator = InstanceSeparator(self.insp_cfg, self.post_cut_registry)

    def _get_ocr_runner(self) -> Optional[OCRCallable]:
        if self._ocr_runner is not None:
            return self._ocr_runner
        if not self._lazy_ocr_loaded:
            self._lazy_ocr_loaded = True
            try:
                from packet_cut.desktop.ocr import run_ocr, run_number_ocr  # type: ignore

                self._ocr_runner = (lambda crop: run_number_ocr(crop, allow_detection_fallback=(self.recovery.ocr_failed_attempts % 3 == 2))) if self.pipe_cfg.pass_by else run_ocr
            except Exception:
                self._ocr_runner = None
        return self._ocr_runner

    def start_session(self, now_sec: Optional[float] = None) -> str:
        ts = time.monotonic() if now_sec is None else float(now_sec)
        self.session_id = f"SES-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        self.started_at = utc_now_iso()
        self.started_mono = ts
        self.stage_entered_mono = ts
        self.camera_status = "ACTIVE"
        self.recovery = RecoveryStats()

        self.testbed_roi.reset()
        self.jewel_tracker.reset()
        self._reset_session_state()

        evidence_dir = Path(self.pipe_cfg.output_dir) / self.session_id / "evidence"
        self.alert_manager = RealTimeAlertManager(
            session_id=self.session_id,
            suppression_cooldown_seconds=self.wf_cfg.alerts.suppression_cooldown_seconds,
            on_alert=self._on_alert,
            on_resolved=self._on_alert_resolved,
        )
        self.activity_logger = ActivityLogger(
            session_id=self.session_id,
            evidence_dir=evidence_dir,
            save_evidence_images=self.pipe_cfg.save_outputs,
            on_event=self._on_activity,
        )
        self._media_recorder = SessionMediaRecorder(
            Path(self.pipe_cfg.output_dir) / self.session_id / "clips",
            self.session_id, self.started_at, ts, enabled=self.pipe_cfg.save_outputs,
        )
        self.procedure_monitor = AppraiserProcedureMonitor(
            session_id=self.session_id,
            alert_manager=self.alert_manager,
            activity_logger=self.activity_logger,
        )

        self.state_machine.reset(self.session_id)
        self.state_machine.transition(
            AutoStage.RELEASE_OBSERVATION if self.pipe_cfg.pass_by else AutoStage.WAIT_EMPTY_TESTBED,
            packet_id="UNRESOLVED",
            actual_detected_event="START button pressed",
            model_used="None",
            confidence=1.0,
            result="SESSION_INITIALIZED",
            compliance_status="COMPLIANT",
        )
        self._clear_stage_counters()
        self._instruction = "Clear the testbed."
        self._latest_observed_action = "Session started — verifying empty testbed."
        self.activity_logger.log(
            packet_id="UNRESOLVED",
            stage=self.stage.value,
            procedure_step="SESSION_START",
            expected_action="Observe release checks independently." if self.pipe_cfg.pass_by else "Verify empty testbed before packet placement.",
            actual_action="START clicked; continuous release observation." if self.pipe_cfg.pass_by else "START clicked; showing 'Clear the testbed.'",
            compliance_status=ComplianceStatus.INFO,
        )
        if self.pipe_cfg.pass_by:
            self._observer = ReleaseObserver(self)
            self._instruction = STATE_MACHINE_SPECS[AutoStage.RELEASE_OBSERVATION].display_instruction
            self._latest_observed_action = "Release observation started; no blocking stage order."
        return self.session_id

    def _save_evidence(self, frame: np.ndarray, tag: str) -> Optional[str]:
        if self.activity_logger is None:
            return None
        path = self.activity_logger.save_evidence_frame(frame, tag)
        if path:
            self._evidence_references.append({"tag": tag, "stage": self.stage.value, "path": path, "timestamp": utc_now_iso()})
        if path and self._media_recorder is not None:
            key = None
            result = "NOT_VERIFIED"
            extra = {}
            if tag.startswith("stage2_ocr_"):
                key, result = "packet_number", self._identity_status
                extra = {"confidence": self._ocr_confidence}
            elif tag == "stage4_cutting_observed":
                key, result = "cutting_activity", "CUTTING_OBSERVED"
                extra = {"hand_present": True, "scissor_present": True}
            elif tag.startswith("stage5_cut_"):
                key, result = "cut_packet", self._cut_result["state"]
                extra = {"reason": self._cut_result.get("reason"),
                         "pieces_count": self._cut_result.get("pieces_count"),
                         "min_gap_px": self._cut_result.get("min_gap_px"),
                         "confidence": self._cut_result.get("cut_model_confidence")}
            elif tag.startswith("stage6_mouth_"):
                key, result = "packet_mouth", self._mouth_status
            elif tag.startswith("stage7_provisional_"):
                key, result = "jewellery_count", self._jewel_count_status
                extra = {"count": self._provisional_jewel_count,
                         "confirmed_unambiguous_count": self._confirmed_unambiguous_count}
            if key:
                self._media_recorder.event(key, path, {
                    "packet_id": self._packet_id, "stage": self.stage.value,
                    "result": result, **extra,
                })
        return path

    def _record_mistake(self, code: str, message: str, stage: str, evidence: Optional[str] = None) -> None:
        self._observed_mistakes.append({
            "code": code,
            "message": message,
            "stage": stage,
            "packet_id": self._packet_id,
            "timestamp": utc_now_iso(),
            "evidence_image": evidence,
        })

    def _record_unverified_action(self, code: str, reason: str, stage: str) -> None:
        if any(u["code"] == code and u["stage"] == stage for u in self._unverified_actions):
            return
        self._unverified_actions.append({
            "code": code,
            "reason": reason,
            "stage": stage,
            "packet_id": self._packet_id,
            "timestamp": utc_now_iso(),
        })

    def _transition(
        self,
        target: AutoStage,
        ts: float,
        actual_event: str,
        model_used: str,
        instruction: str,
        confidence: float = 1.0,
        result: str = "VERIFIED",
        compliance_status: str = "COMPLIANT",
        evidence_reference: Optional[str] = None,
    ) -> None:
        """Record a verified check or a strict-mode stage transition."""
        if self._observer is not None and self._observation_context is not None:
            self._observer.transition_observed(self._observation_context, result, actual_event, evidence_reference)
            return
        self.state_machine.transition(
            target=target,
            packet_id=self._packet_id,
            actual_detected_event=actual_event,
            model_used=model_used,
            confidence=confidence,
            result=result,
            compliance_status=compliance_status,
            evidence_reference=evidence_reference,
        )
        self._clear_stage_counters()
        self.stage_entered_mono = ts
        self._last_timeout_alert_mono = ts
        self._latest_observed_action = actual_event
        self._instruction = instruction

    def _emit_timeout_reminder(
        self, ts: float, timeout_sec: float, code: str, message: str, model_used: str
    ) -> None:
        """Timers ONLY raise non-interactive alerts/logs — NEVER advance a state."""
        if self._observer is not None:
            return
        elapsed = ts - self.stage_entered_mono
        if elapsed < timeout_sec:
            return
        if (ts - self._last_timeout_alert_mono) < max(timeout_sec, self.wf_cfg.alerts.suppression_cooldown_seconds):
            return
        self._last_timeout_alert_mono = ts
        self.recovery.timeout_alerts_emitted += 1
        self.recovery.last_recovery = f"Waiting for physical event in {self.stage.value} ({elapsed:.0f}s)"
        if self.alert_manager is not None:
            self.alert_manager.trigger(
                code=code,
                packet_id=self._packet_id,
                stage=self.stage.value,
                custom_message=message,
                severity=AlertSeverity.INFO,
                now_sec=ts,
            )
        self.state_machine.record_event(
            packet_id=self._packet_id,
            actual_detected_event=f"Still waiting after {elapsed:.1f}s — no premature advancement.",
            model_used=model_used,
            confidence=0.0,
            result="WAITING_REMINDER",
            compliance_status="UNVERIFIED",
        )

    def _safe_infer(self, model_key: str, fn: Callable[[], Any], default: Any, models_used: List[str]) -> Any:
        models_used.append(model_key)
        attempts = max(1, self.pipe_cfg.infer_max_retries + 1)
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                out = fn()
                if attempt > 0:
                    self.recovery.recovered_errors += 1
                    self.recovery.last_recovery = f"Recovered '{model_key}' on retry #{attempt}"
                return out
            except Exception as exc:
                last_exc = exc
                self.recovery.total_inference_errors += 1
                self.recovery.last_error = f"{model_key}: {type(exc).__name__}: {exc}"
                if attempt < attempts - 1:
                    self.recovery.total_retries += 1
                    if self.pipe_cfg.infer_backoff_sec > 0:
                        time.sleep(self.pipe_cfg.infer_backoff_sec * (attempt + 1))
        if self.activity_logger is not None:
            self.activity_logger.log(
                packet_id=self._packet_id,
                stage=self.stage.value,
                procedure_step="INFERENCE_ERROR",
                expected_action=f"Run {model_key}",
                actual_action=f"Failed after {attempts} attempts: {last_exc}",
                compliance_status=ComplianceStatus.UNVERIFIED,
                reason=str(last_exc),
            )
        return default

    # ==================================================================
    # STAGE-SCOPED MODEL HELPERS
    # ==================================================================
    def _run_packet_seg(self, frame: np.ndarray, ov: Dict[str, Any], models_used: List[str]) -> Dict[str, Any]:
        if "packet_mask" in ov:
            models_used.append("packet-seg.onnx")
            m = (ov["packet_mask"] > 0).astype(np.uint8) * 255
            num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
            pieces = []
            for lbl in range(1, num):
                area = int(stats[lbl, cv2.CC_STAT_AREA])
                if area < packet_cut_core.MIN_PIECE_AREA_PX:
                    continue
                pm = np.zeros_like(m)
                pm[labels == lbl] = 255
                pieces.append({
                    "mask": pm,
                    "area": area,
                    "centroid": (float(centroids[lbl][0]), float(centroids[lbl][1])),
                    "bbox": (
                        int(stats[lbl, cv2.CC_STAT_LEFT]),
                        int(stats[lbl, cv2.CC_STAT_TOP]),
                        int(stats[lbl, cv2.CC_STAT_WIDTH]),
                        int(stats[lbl, cv2.CC_STAT_HEIGHT]),
                    ),
                })
            pieces.sort(key=lambda p: p["area"], reverse=True)
            return self._filter_background_packets(
                frame, {"pieces": pieces, "union_mask": m, "total_area": int(np.count_nonzero(m))}
            )

        empty_info = {"pieces": [], "union_mask": np.zeros(frame.shape[:2], dtype=np.uint8), "total_area": 0}
        if not self.model_manager.has("packet"):
            return empty_info

        def _call() -> Dict[str, Any]:
            res = self.model_manager.get("packet").predict(
                source=frame, imgsz=self.wf_cfg.models.imgsz, conf=desk_cfg.PACKET_CONF,
                device=self.model_manager.device, verbose=False,
            )[0]
            return packet_cut_core.extract_packet_pieces(
                res, frame.shape[:2], packet_ids=self.model_manager.get_class_ids("packet"),
                reference_area=self._intact_baseline_area if self._intact_baseline_area > 0 else None,
                min_conf=desk_cfg.PACKET_CONF, verbose=False,
            )

        info = self._safe_infer("packet-seg.onnx", _call, empty_info, models_used)
        return self._filter_background_packets(frame, info)

    def set_empty_reference(self, frame: np.ndarray) -> None:
        """Calibrate from the empty preview explicitly selected by the operator."""
        self.testbed_roi.capture_empty_baseline(frame)
        self._has_calibrated_empty_reference = True

    def _filter_background_packets(self, frame: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
        """Ignore packet predictions on unchanged, calibrated surface markings.

        Apply in every stage so the same rectangle cannot become a phantom
        front-side packet after empty-bed verification has passed.
        """
        if not self._has_calibrated_empty_reference:
            return info
        foreground = self.testbed_roi.foreground_mask(frame)
        if foreground is None:
            return info
        pieces = []
        union = np.zeros(frame.shape[:2], dtype=np.uint8)
        for piece in info.get("pieces", []):
            mask = piece["mask"] > 0
            area = int(np.count_nonzero(mask))
            changed_area = int(np.count_nonzero(mask & (foreground > 0)))
            if changed_area < max(120, int(area * 0.10)):
                continue
            pieces.append(piece)
            union[mask] = 255
        return {
            **info,
            "pieces": pieces,
            "union_mask": union,
            "total_area": int(np.count_nonzero(union)),
            "accepted_piece_count": len(pieces),
            "background_predictions_ignored": len(info.get("pieces", [])) - len(pieces),
        }

    def _run_number_obb(
        self, frame: np.ndarray, ov: Dict[str, Any], models_used: List[str]
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]], float]:
        if "ocr_roi" in ov:
            models_used.append("packet-no-obb.onnx")
            roi = ov["ocr_roi"]
            if roi is None:
                return False, None, 0.0
            conf = float(ov.get("frontside_conf", 0.90))
            area = max(0, roi[2] - roi[0]) * max(0, roi[3] - roi[1])
            ok = conf >= self.pipe_cfg.frontside_min_conf and area >= self.pipe_cfg.frontside_min_box_area
            return ok, roi if ok else None, conf

        if not self.model_manager.has("number"):
            return False, None, 0.0

        def _call() -> Tuple[bool, Optional[Tuple[int, int, int, int]], float]:
            h, w = frame.shape[:2]
            res = self.model_manager.get("number").predict(
                source=frame, imgsz=self.wf_cfg.models.number_imgsz,
                conf=max(self.pipe_cfg.frontside_min_conf, desk_cfg.NUMBER_CONF),
                device=self.model_manager.device, verbose=False,
            )[0]
            boxes = res.obb.xyxy.detach().cpu().numpy() if (hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0) else (
                res.boxes.xyxy.detach().cpu().numpy() if (hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0) else None
            )
            confs = res.obb.conf.detach().cpu().numpy() if (hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0) else (
                res.boxes.conf.detach().cpu().numpy() if (hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0) else None
            )
            if boxes is None or len(boxes) == 0:
                return False, None, 0.0
            i = int(np.argmax(confs))
            conf = float(confs[i])
            b = boxes[i].astype(int)
            x1, y1 = max(0, min(int(b[0]), w - 1)), max(0, min(int(b[1]), h - 1))
            x2, y2 = max(0, min(int(b[2]), w)), max(0, min(int(b[3]), h))
            area = max(0, x2 - x1) * max(0, y2 - y1)
            ok = conf >= self.pipe_cfg.frontside_min_conf and area >= self.pipe_cfg.frontside_min_box_area
            return ok, (x1, y1, x2, y2) if ok else None, conf

        return self._safe_infer("packet-no-obb.onnx", _call, (False, None, 0.0), models_used)

    def _run_hand_and_scissor(
        self, frame: np.ndarray, ov: Dict[str, Any], models_used: List[str], run_scissor: bool = True,
        *, full_view: bool = False, require_clearance: bool = False,
    ) -> Tuple[bool, np.ndarray, bool, np.ndarray, bool]:
        h, w = frame.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)

        if "hand_present" in ov or "hand_mask" in ov:
            models_used.append("hand-seg.onnx")
            hand_mask = (ov["hand_mask"] > 0).astype(np.uint8) * 255 if "hand_mask" in ov else empty
            raw_hp = bool(ov.get("hand_present", np.any(hand_mask > 0)))
        elif self.model_manager.has("hand"):
            def _hcall() -> Tuple[bool, np.ndarray]:
                hr = self.model_manager.get("hand").predict(
                    source=frame, imgsz=self.wf_cfg.models.imgsz, conf=desk_cfg.HAND_CONF,
                    device=self.model_manager.device, verbose=False,
                )[0]
                top_m = ov.get("topstrip_mask") if "topstrip_mask" in ov else getattr(self, "_last_topstrip_mask", None)
                hp, hm, _ = packet_cut_core.evaluate_hand_detection(
                    hr, self.model_manager.get_class_ids("hand"), (h, w),
                    reference_mask=None if full_view else self._intact_baseline_mask,
                    topstrip_mask=None if full_view else top_m,
                    frame=frame,
                    color_only=full_view,
                )
                return hp, hm
            raw_hp, hand_mask = self._safe_infer("hand-seg.onnx", _hcall, (require_clearance, empty), models_used)
        else:
            raw_hp, hand_mask = require_clearance, empty

        raw_sp, scissor_mask = False, empty
        if run_scissor:
            roi = getattr(self, "cutting_roi", None) or desk_cfg.get_cutting_roi((h, w))
            rx1, ry1, rx2, ry2 = roi
            if "scissor_present" in ov or "scissor_mask" in ov:
                models_used.append("scissor-seg.pt")
                scissor_mask = (ov["scissor_mask"] > 0).astype(np.uint8) * 255 if "scissor_mask" in ov else empty
                raw_sp = bool(ov.get("scissor_present", np.any(scissor_mask > 0)))
            elif self.model_manager.has("scissor"):
                def _scall() -> Tuple[bool, np.ndarray]:
                    # Mask out bottom railing zone below ry2 so railing is never detected as scissors
                    frame_scissor = frame.copy()
                    if not full_view and ry2 < h:
                        frame_scissor[ry2:, :] = 0

                    sr = self.model_manager.get("scissor").predict(
                        source=frame_scissor, imgsz=self.wf_cfg.models.imgsz, conf=desk_cfg.SCISSOR_CONF,
                        device=self.model_manager.device, verbose=False,
                    )[0]
                    sp, sm, _ = packet_cut_core.evaluate_scissor_detection(
                        sr, self.model_manager.get_class_ids("scissor"), (h, w),
                        packet_mask=None if full_view else self._intact_baseline_mask,
                        roi=None if full_view else roi,
                        reject_bottom_rail=getattr(self, "cutting_roi", None) is None,
                    )
                    return sp, sm
                raw_sp, scissor_mask = self._safe_infer("scissor-seg.pt", _scall, (require_clearance, empty), models_used)
            else:
                raw_sp = require_clearance

        # Cutting uses one shared debounce for simultaneous detections in Stage 4.
        if full_view:
            return raw_hp, hand_mask, raw_sp, scissor_mask, False

        self._hand_streak = (self._hand_streak + 1) if raw_hp else 0
        self._scissor_streak = (self._scissor_streak + 1) if raw_sp else 0
        hp = self._hand_streak >= self.pipe_cfg.hand_confirm_frames
        sp = self._scissor_streak >= self.pipe_cfg.scissor_confirm_frames

        together = False
        if hp and sp:
            if np.any(hand_mask > 0) and np.any(scissor_mask > 0):
                hc, hb, _ = mask_centroid_and_bbox(hand_mask)
                sc, sb, _ = mask_centroid_and_bbox(scissor_mask)
                centroid_dist = float(np.hypot(hc[0] - sc[0], hc[1] - sc[1]))
                hx1, hy1, hx2, hy2 = hb
                sx1, sy1, sx2, sy2 = sb
                bbox_dx = max(0, max(hx1 - sx2, sx1 - hx2))
                bbox_dy = max(0, max(hy1 - sy2, sy1 - hy2))
                bbox_dist = float(np.hypot(bbox_dx, bbox_dy))

                together = (
                    bool(np.any((hand_mask > 0) & (scissor_mask > 0)))
                    or centroid_dist <= max(self.pipe_cfg.cutting_proximity_px, 280.0)
                    or bbox_dist <= 80.0
                )
            else:
                together = bool(ov.get("cutting_region_overlap", True))
        return hp, hand_mask, sp, scissor_mask, together

    def _run_topstrip(self, frame: np.ndarray, ov: Dict[str, Any], models_used: List[str]) -> Tuple[Optional[bool], np.ndarray]:
        h, w = frame.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)
        if "topstrip_visible" in ov:
            models_used.append("strip-seg.onnx")
            return bool(ov["topstrip_visible"]), empty
        if not self.model_manager.has("topstrip"):
            return None, empty

        def _call() -> Tuple[bool, np.ndarray]:
            res = self.model_manager.get("topstrip").predict(
                source=frame, imgsz=self.wf_cfg.models.imgsz, conf=desk_cfg.TOPSTRIP_CONF,
                device=self.model_manager.device, verbose=False,
            )[0]
            insts = extract_yolo_instances(
                res, (h, w), selected_ids=self.model_manager.get_class_ids("topstrip"),
                min_conf=desk_cfg.TOPSTRIP_CONF, min_area_px=60,
            )
            return len(insts) > 0, union_masks([i.mask for i in insts], (h, w))

        return self._safe_infer("strip-seg.onnx", _call, (None, empty), models_used)

    def _run_cut_model(self, frame: np.ndarray, ov: Dict[str, Any], models_used: List[str]) -> Dict[str, Any]:
        default_eval = {"is_cut_detected": False, "confidence": 0.0, "class_name": "unavailable", "reason": "No result"}
        if "cut_eval" in ov:
            models_used.append(self._cut_model_filename)
            return dict(ov["cut_eval"])
        if not self.model_manager.has("cut"):
            return default_eval

        def _call() -> Dict[str, Any]:
            res = self.model_manager.get("cut").predict(
                source=frame, imgsz=self.wf_cfg.models.imgsz, conf=0.25,
                device=self.model_manager.device, verbose=False,
            )[0]
            return evaluate_cut_model_prediction(
                res, model=self.model_manager.get("cut"),
                cut_class_id=desk_cfg.CUT_MODEL_CLASS_ID, min_conf=self.pipe_cfg.cut_min_conf,
            )

        return self._safe_infer(self._cut_model_filename, _call, default_eval, models_used)

    @staticmethod
    def _packet_appearance_signature(frame: np.ndarray, packet_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute normalized intensity/color histogram + printed-grid edge density inside packet."""
        eroded = cv2.erode((packet_mask > 0).astype(np.uint8) * 255, np.ones((15, 15), np.uint8))
        region = eroded if np.count_nonzero(eroded) >= 200 else (packet_mask > 0).astype(np.uint8) * 255
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], region, [8, 4, 8], [0, 180, 0, 256, 0, 256]).flatten()
        norm = float(np.linalg.norm(hist))
        if norm > 1e-6:
            hist = hist / norm
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 55, 145)
        area = max(1, int(np.count_nonzero(region)))
        grid_density = float(np.count_nonzero((edges > 0) & (region > 0))) / float(area)
        return hist, grid_density

    def _inspect_packet_mouth(
        self, frame: np.ndarray, packet_mask: np.ndarray, hand_mask: np.ndarray
    ) -> Tuple[str, str, Optional[np.ndarray]]:
        """Determine mouth state (OPEN / CLOSED / UNKNOWN) from visible mouth-edge
        separation and recessed interior exposure — NEVER from whole-packet mask alone!
        """
        h, w = frame.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)

        quality = evaluate_frame_quality(
            frame,
            min_blur_variance=self.pipe_cfg.mouth_min_focus_var,
            min_mean_brightness=self.insp_cfg.packet_inspection.min_mean_brightness,
            max_mean_brightness=self.insp_cfg.packet_inspection.max_mean_brightness,
            roi_mask=packet_mask,
        )
        if not quality.is_acceptable:
            return "UNKNOWN", f"Mouth view out of focus or poor lighting (blur={quality.blur_variance:.1f}, V={quality.mean_brightness:.1f}).", empty

        pkt_area = int(np.count_nonzero(packet_mask))
        if pkt_area < 2500:
            return "UNKNOWN", "Held packet too small to inspect mouth edges.", empty

        hand_overlap = float(np.count_nonzero((packet_mask > 0) & (hand_mask > 0))) / float(max(pkt_area, 1))
        if hand_overlap > 0.35:
            return "UNKNOWN", f"Hand occludes packet mouth ({hand_overlap:.0%}).", empty

        # Exclude outer packet walls & fingers to inspect interior mouth cavity vs lip rim
        pkt_bin = (packet_mask > 0).astype(np.uint8) * 255
        inset = cv2.erode(pkt_bin, np.ones((15, 15), np.uint8))
        if np.any(hand_mask > 0):
            inset[cv2.dilate((hand_mask > 0).astype(np.uint8) * 255, np.ones((7, 7), np.uint8)) > 0] = 0
        if int(np.count_nonzero(inset)) < 500:
            return "UNKNOWN", "Usable mouth ROI too small after rim/hand exclusion.", empty

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rim_ring = (pkt_bin > 0) & (inset == 0) & (hand_mask == 0)
        rim_brightness = float(np.median(gray[rim_ring])) if np.any(rim_ring) else float(np.percentile(gray[inset > 0], 85))

        # Exposed interior cavity inside separated mouth edges is visibly darker than the packet lip rim
        # or exposes gold jewellery inside the separated opening
        dark_cavity = (inset > 0) & (gray.astype(np.float32) <= (rim_brightness - self.pipe_cfg.mouth_min_recess_contrast))
        hsv_in = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab_in = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        glare_in = (hsv_in[:, :, 2] >= 245) & (hsv_in[:, :, 1] <= 45)
        gold_in_inset = (
            (hsv_in[:, :, 0] >= 10) & (hsv_in[:, :, 0] <= 38) &
            (hsv_in[:, :, 1] >= 50) & (hsv_in[:, :, 2] >= 45) &
            (lab_in[:, :, 2] >= 135) & (~glare_in) & (inset > 0)
        )
        interior = dark_cavity | gold_in_inset
        cavity_mask = np.zeros((h, w), dtype=np.uint8)
        cavity_mask[interior] = 255
        cavity_mask = cv2.morphologyEx(cavity_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cavity_mask = cv2.morphologyEx(cavity_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        cavity_area = int(np.count_nonzero(cavity_mask))
        if cavity_area >= self.pipe_cfg.mouth_min_cavity_area_px:
            # Measure minimum separation width across the exposed mouth cavity via distance transform
            dist = cv2.distanceTransform(cavity_mask, cv2.DIST_L2, 5)
            opening_width_px = float(dist.max()) * 2.0
            # Verify clear lip edges bordering the cavity
            edges = cv2.Canny(gray, 45, 130)
            cavity_border = cv2.dilate(cavity_mask, np.ones((7, 7), np.uint8)) & (~cavity_mask)
            edge_ratio = float(np.count_nonzero((edges > 0) & (cavity_border > 0))) / float(max(1, np.count_nonzero(cavity_border)))
            if opening_width_px >= self.pipe_cfg.mouth_min_open_width_px and edge_ratio >= 0.04:
                return (
                    "OPEN",
                    f"Mouth edges visibly separated (opening={opening_width_px:.1f}px, exposed cavity={cavity_area}px).",
                    cavity_mask,
                )

        # Check for visibly pressed-together closed mouth seam (sharp linear edge, no dark opening)
        edges = cv2.Canny(gray, 55, 150)
        inset_edges = (edges > 0) & (inset > 0)
        edge_px = int(np.count_nonzero(inset_edges))
        # A closed pressed mouth shows a focused thin seam (line segment) across the inset with NO recessed dark cavity
        lines = cv2.HoughLinesP(
            (inset_edges.astype(np.uint8) * 255),
            rho=1,
            theta=np.pi / 180.0,
            threshold=28,
            minLineLength=28,
            maxLineGap=8,
        )
        if lines is not None and len(lines) >= 1 and cavity_area < (self.pipe_cfg.mouth_min_cavity_area_px // 3):
            return (
                "CLOSED",
                f"Mouth edges visibly pressed together (seam detected, no exposed interior cavity; edge_px={edge_px}).",
                empty,
            )

        return "UNKNOWN", "Whole-packet view insufficient to verify mouth-edge separation.", empty

    # ==================================================================
    # MAIN ENTRY POINT — STRICTLY ONE STATE PER FRESH FRAME
    # ==================================================================
    def process_frame(
        self,
        frame: np.ndarray,
        now_sec: Optional[float] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> AutoStepSnapshot:
        """Process EXACTLY ONE active state on this fresh camera frame."""
        if self.alert_manager is None or self.activity_logger is None or self.procedure_monitor is None:
            raise RuntimeError("Call start_session() before processing frames.")

        ts = time.monotonic() if now_sec is None else float(now_sec)
        ov = overrides or {}
        self._frame_idx += 1
        models_used: List[str] = []

        if self.stage == AutoStage.STOPPED_OR_SAVED:
            return self._build_snapshot(frame.copy(), "Idle (Saved)", models_used, "Session already saved.", False, True)
        if self._media_recorder is not None:
            self._media_recorder.record_frame(frame, ts)

        if self._observer is not None:
            return self._observer.process(frame, ts, ov, models_used)
        if self.stage == AutoStage.WAIT_EMPTY_TESTBED:
            return self._handle_stage1_empty_testbed(frame, ts, ov, models_used)
        if self.stage == AutoStage.WAIT_FRONTSIDE_AND_OCR:
            return self._handle_stage2_frontside_and_ocr(frame, ts, ov, models_used)
        if self.stage == AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP:
            return self._handle_stage3_turn_and_backside(frame, ts, ov, models_used)
        if self.stage == AutoStage.WAIT_CUTTING_ACTIVITY:
            return self._handle_stage4_cutting_activity(frame, ts, ov, models_used)
        if self.stage == AutoStage.VERIFY_PHYSICAL_CUT:
            return self._handle_stage5_verify_cut(frame, ts, ov, models_used)
        if self.stage == AutoStage.WAIT_UPRIGHT_MOUTH:
            return self._handle_stage6_upright_mouth(frame, ts, ov, models_used)
        if self.stage in {AutoStage.WAIT_GOLD_PLACEMENT, AutoStage.WAIT_JEWELLERY_AND_COUNT}:
            return self._handle_stage7_jewellery_counting(frame, ts, ov, models_used)

        return self._build_snapshot(frame.copy(), "Idle", models_used, "Idle.", False, False)

    # ------------------------------------------------------------------
    # STAGE 1: WAIT FOR EMPTY TESTBED (packet-seg.onnx ONLY)
    # ------------------------------------------------------------------
    def _handle_stage1_empty_testbed(
        self, frame: np.ndarray, ts: float, ov: Dict[str, Any], models_used: List[str]
    ) -> AutoStepSnapshot:
        active_model = "packet-seg.onnx + Empty Testbed Check"
        vis = frame.copy()
        # A clear, smooth tray can have almost no edges once cropped to the ROI.
        # Focus still matters for later stages that inspect packet details.
        quality = evaluate_frame_quality(
            frame,
            min_blur_variance=0.0,
            min_mean_brightness=38.0,
            max_mean_brightness=245.0,
        )
        # Stage 1 executes packet segmentation only
        p_info = self._run_packet_seg(frame, ov, models_used)
        pkt_present = p_info["total_area"] >= 1200

        # Hand/scissor/jewel models are strictly skipped in Stage 1 unless test overrides supply them
        hp = False
        if "hand_present" in ov or "hand_mask" in ov:
            hp, _, _, _, _ = self._run_hand_and_scissor(frame, ov, models_used, run_scissor=False)

        jewel_count = len(ov.get("jewel_instances", []))
        if "jewel_instances" in ov:
            models_used.append("jeweltrack.onnx")

        empty_res = self.testbed_roi.check_empty(
            frame,
            jewel_count_in_roi=jewel_count,
            hand_present_in_roi=hp,
            packet_present_in_roi=pkt_present,
        )

        bed_clear = quality.is_acceptable and empty_res.raw_is_empty and not pkt_present and not hp and jewel_count == 0

        if bed_clear:
            self._empty_streak += 1
        else:
            self._empty_streak = 0

        if self._empty_streak >= self.pipe_cfg.empty_testbed_frames:
            self._empty_verified = True
            ev = self._save_evidence(vis, "stage1_empty_testbed")
            assert self.procedure_monitor is not None
            self.procedure_monitor.on_testbed_empty_confirmed("UNRESOLVED", self.stage.value)
            self._transition(
                AutoStage.WAIT_FRONTSIDE_AND_OCR,
                ts=ts,
                actual_event=f"Testbed verified visible, clear, and stable across {self._empty_streak} fresh frames.",
                model_used=active_model,
                instruction="Please place the packet FRONT side up.",
                result="EMPTY_TESTBED_VERIFIED",
                evidence_reference=ev,
            )
            return self._build_snapshot(vis, active_model, models_used, "Testbed clear. Please place the packet FRONT side up.", True, False)

        if quality.is_too_dark:
            reason = "Selected testbed region is too dark."
        elif quality.is_overexposed:
            reason = "Selected testbed region is overexposed."
        else:
            reason = empty_res.reason
        self._packet_det_summary = reason
        self._emit_timeout_reminder(ts, self.pipe_cfg.empty_testbed_timeout_sec, "TESTBED_NOT_EMPTY", "Clear the testbed.", active_model)
        return self._build_snapshot(
            vis, active_model, models_used,
            f"Clear the testbed ({self._empty_streak}/{self.pipe_cfg.empty_testbed_frames}) — {reason}",
            False, False,
        )

    # ------------------------------------------------------------------
    # STAGE 2: WAIT FOR FRONT-SIDE PLACEMENT, THEN READ THE NUMBER
    # ------------------------------------------------------------------
    def _handle_stage2_frontside_and_ocr(
        self, frame: np.ndarray, ts: float, ov: Dict[str, Any], models_used: List[str]
    ) -> AutoStepSnapshot:
        active_model = "packet-seg.onnx + packet-no-obb.onnx + PaddleOCR"
        vis = frame.copy()
        tb_roi = self.testbed_roi.get_roi(frame)
        tb_mask = self.testbed_roi.get_roi_mask(frame)

        p_info = self._run_packet_seg(frame, ov, models_used)
        pieces = p_info["pieces"]
        vis = draw_mask_overlay(vis, p_info["union_mask"], (255, 130, 30), alpha=0.18)

        # Check if a packet is actually placed inside the testbed ROI
        pkt_mask = pieces[0]["mask"] if len(pieces) >= 1 else None
        pkt_in_bed = False
        if pkt_mask is not None and p_info["total_area"] >= 1500:
            in_tb = int(np.count_nonzero((pkt_mask > 0) & (tb_mask > 0)))
            pkt_in_bed = (float(in_tb) / float(max(1, int(np.count_nonzero(pkt_mask))))) >= 0.30

        if not pkt_in_bed:
            self._frontside_streak = 0
            self._ocr_missing_streak = 0
            self._prev_pkt_centroid = None
            self._packet_det_summary = "No packet placed inside testbed"
            self._emit_timeout_reminder(
                ts, self.pipe_cfg.frontside_timeout_sec, "OCR_BEFORE_CUTTING_REQUIRED",
                "Please place the packet FRONT side up.", active_model,
            )
            return self._build_snapshot(vis, active_model, models_used, "Please place the packet FRONT side up.", False, False)

        # Packet is inside testbed -> detect OBB number region and verify it belongs to THIS packet!
        assert pkt_mask is not None
        pkt_centroid = pieces[0]["centroid"]
        pkt_area = float(pieces[0]["area"])
        stationary = True
        if self._prev_pkt_centroid is not None:
            shift = float(np.hypot(pkt_centroid[0] - self._prev_pkt_centroid[0], pkt_centroid[1] - self._prev_pkt_centroid[1]))
            area_delta = abs(pkt_area - self._prev_pkt_area) / max(self._prev_pkt_area, 1.0)
            stationary = (shift <= self.pipe_cfg.max_stationary_shift_px) and (area_delta <= self.pipe_cfg.max_stationary_area_ratio)
        self._prev_pkt_centroid = pkt_centroid
        self._prev_pkt_area = pkt_area

        obb_ok, obb_box, obb_conf = self._run_number_obb(frame, ov, models_used)
        obb_belongs_to_packet = False
        if obb_ok and obb_box is not None:
            ox1, oy1, ox2, oy2 = obb_box
            box_area = max(1, (ox2 - ox1) * (oy2 - oy1))
            in_pkt = int(np.count_nonzero(pkt_mask[oy1:oy2, ox1:ox2] > 0))
            obb_belongs_to_packet = (float(in_pkt) / float(box_area)) >= self.pipe_cfg.frontside_min_packet_overlap

        if not (obb_ok and obb_belongs_to_packet and obb_box is not None):
            self._frontside_streak = 0
            # Check if packet was turned away too early after front was seen (or backside placed first)
            if self._frontside_seen_once:
                self._ocr_missing_streak += 1
                if self._ocr_missing_streak >= self.pipe_cfg.ocr_turn_early_frames:
                    ev = self._save_evidence(vis, "stage2_turned_before_ocr")
                    self._record_mistake(
                        "OCR_TURNED_EARLY",
                        "Packet was turned before OCR succeeded; prompted appraiser to show front side again.",
                        self.stage.value,
                        ev,
                    )
                    assert self.procedure_monitor is not None
                    self.procedure_monitor.on_backside_without_ocr("UNRESOLVED", self.stage.value, frame=frame, now_sec=ts)
                    self._instruction = "OCR missed — please show the packet FRONT side up again."
                    self._frontside_seen_once = False
            self._packet_det_summary = "Packet present without readable frontside number region"
            return self._build_snapshot(
                vis, active_model, models_used,
                "Packet detected without frontside number region — please show the FRONT side clearly.",
                False, False,
            )

        # Valid number box belongs to packet! Check stability and sharpness before PaddleOCR
        self._frontside_seen_once = True
        self._ocr_missing_streak = 0
        ox1, oy1, ox2, oy2 = obb_box
        crop = frame[oy1:oy2, ox1:ox2]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size > 0 else np.zeros((2, 2), dtype=np.uint8)
        crop_sharpness = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())
        sharp_ok = (crop_sharpness >= self.pipe_cfg.ocr_min_crop_sharpness) or ("ocr_text" in ov)

        if stationary and sharp_ok:
            self._frontside_streak += 1
        else:
            self._frontside_streak = 0

        required_stable = self.pipe_cfg.pass_by_ocr_stable_frames if self.pipe_cfg.pass_by else self.pipe_cfg.frontside_stable_frames
        if self._frontside_streak < required_stable:
            self._packet_det_summary = (
                f"Frontside number region stabilizing ({self._frontside_streak}/{required_stable})"
            )
            return self._build_snapshot(
                vis, active_model, models_used,
                f"Frontside detected — stabilizing sharp number crop ({self._frontside_streak}/{required_stable})...",
                False, False,
            )

        # Run PaddleOCR on this independent fresh crop
        models_used.append("PaddleOCR")
        self.recovery.ocr_attempts += 1
        raw_text: Optional[str] = None
        raw_conf: float = 0.0
        ocr_started = time.perf_counter()
        if "ocr_text" in ov:
            raw_text, raw_conf = ov["ocr_text"]
        else:
            runner = self._get_ocr_runner()
            if runner is not None:
                raw_text, raw_conf, _ = self._safe_infer("PaddleOCR", lambda: runner(crop), (None, 0.0, []), models_used)

        elapsed_ms = (time.perf_counter() - ocr_started) * 1000
        self._ocr_timing["ocr_calls"] += 1
        self._ocr_timing["ocr_total_ms"] += elapsed_ms
        self._ocr_timing["last_ocr_ms"] = elapsed_ms
        cleaned = re.sub(r"[^0-9A-Za-z]", "", str(raw_text).strip()).upper() if raw_text else ""
        if len(cleaned) >= self.pipe_cfg.ocr_min_id_length and float(raw_conf) >= self.pipe_cfg.ocr_min_confidence:
            self._ocr_votes[cleaned] += 1
            self._ocr_best_conf[cleaned] = max(self._ocr_best_conf.get(cleaned, 0.0), float(raw_conf))
            votes = self._ocr_votes[cleaned]
            self._packet_det_summary = f"OCR candidate '{cleaned}' ({votes}/{self.pipe_cfg.ocr_min_consistent_crops} fresh crops)"

            # Require consistent reads across independent fresh crops (NEVER lock on 1 single frame!)
            if votes >= self.pipe_cfg.ocr_min_consistent_crops:
                self._packet_id = cleaned
                self._identity_status = "RESOLVED"
                self._ocr_confidence = self._ocr_best_conf[cleaned]
                self._frontside_verified = True
                self._frontside_centroid = pkt_centroid
                self._frontside_ref_hist, self._frontside_grid_density = self._packet_appearance_signature(frame, pkt_mask)
                cv2.rectangle(vis, (ox1, oy1), (ox2, oy2), (50, 230, 130), 2)
                cv2.putText(vis, f"Packet ID: {cleaned}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 160, 30), 2)
                self._ocr_crop_evidence = self.activity_logger.save_evidence_frame(crop, "packet_number_crop")
                ev = self._save_evidence(vis, f"stage2_ocr_{cleaned}")
                self._ocr_evidence = ev
                assert self.procedure_monitor is not None
                self.procedure_monitor.on_ocr_resolved(cleaned, self._ocr_confidence, self.stage.value, False)
                self._transition(
                    AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP,
                    ts=ts,
                    actual_event=(
                        f"Frontside verified and PaddleOCR locked '{cleaned}' "
                        f"({self._ocr_confidence:.0%}) across {votes} independent fresh crops."
                    ),
                    model_used=active_model,
                    instruction="Please turn the packet to the BACK side.",
                    confidence=self._ocr_confidence,
                    result="OCR_LOCKED",
                    evidence_reference=ev,
                )
                return self._build_snapshot(
                    vis, active_model, models_used,
                    f"Packet ID {cleaned} locked. Please turn the packet to the BACK side.",
                    True, False,
                )
        else:
            self.recovery.ocr_failed_attempts += 1
            self.recovery.last_recovery = f"Auto-retrying OCR (failed reads: {self.recovery.ocr_failed_attempts})"

        return self._build_snapshot(
            vis, active_model, models_used,
            f"Reading packet number automatically (attempts: {self.recovery.ocr_attempts})...",
            False, False,
        )

    # ------------------------------------------------------------------
    # STAGE 3: WAIT FOR THE TURN, THEN CHECK THE BACKSIDE STRIP
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # STAGE 3: WAIT FOR THE TURN, THEN CHECK THE BACKSIDE STRIP
    # ------------------------------------------------------------------
    def _handle_stage3_turn_and_backside(
        self, frame: np.ndarray, ts: float, ov: Dict[str, Any], models_used: List[str]
    ) -> AutoStepSnapshot:
        active_model = "strip-seg.onnx"
        vis = frame.copy()

        # In Stage 3, run packet-seg to locate packet
        p_info = self._run_packet_seg(frame, ov, models_used)
        pieces = p_info["pieces"]

        hp, hand_mask, sp, _, both_together = self._run_hand_and_scissor(frame, ov, models_used)

        vis = draw_mask_overlay(vis, p_info["union_mask"], (255, 130, 30), alpha=0.18)
        if np.any(hand_mask > 0):
            vis = draw_mask_overlay(vis, hand_mask, (50, 50, 230), alpha=0.30)

        # Cutting seen BEFORE backside readiness is verified!
        if hp and sp and both_together:
            if not self._early_cutting_reported:
                ev = self._save_evidence(vis, "mistake_cutting_before_readiness")
                self._record_mistake(
                    "CUTTING_BEFORE_READINESS",
                    "Cutting activity (hand + scissors) observed before backside TopStrip readiness was verified.",
                    self.stage.value,
                    ev,
                )
                assert self.procedure_monitor is not None
                self.procedure_monitor.on_cutting_started(
                    self._packet_id, self._identity_status == "RESOLVED",
                    False, self._strip_result == "STRIP_VISIBLE", AutoStage.WAIT_CUTTING_ACTIVITY.value,
                    frame=frame, now_sec=ts,
                )
                if self._intact_baseline_mask is None and len(pieces) == 1:
                    self._intact_baseline_mask = pieces[0]["mask"]
                    self._intact_baseline_area = float(pieces[0]["area"])
            self._early_cutting_reported = True
            self._backside_ready_streak = 0
            return self._build_snapshot(
                vis, active_model, models_used, "Cutting detected before readiness. Verify the Top Strip first.", False, False
            )

        # If test overrides indicate the front OCR ROI is still facing the camera, keep waiting for the turn
        if "ocr_roi" in ov and ov.get("ocr_roi") is not None:
            models_used.append("packet-no-obb.onnx")
            self._backside_ready_streak = 0
            self._packet_det_summary = "FRONT side still visible — waiting for turn"
            self._emit_timeout_reminder(
                ts, self.pipe_cfg.backside_timeout_sec, "ORIENTATION_NOT_FOLLOWED",
                "Please turn the packet to the BACK side.", active_model,
            )
            return self._build_snapshot(vis, active_model, models_used, "Please turn the packet to the BACK side.", False, False)

        # 2. Require a single clear, focused packet
        if len(pieces) != 1 or p_info["total_area"] < 1500:
            self._backside_ready_streak = 0
            self._strip_result = "UNKNOWN_UNCLEAR_VIEW"
            self._packet_det_summary = "Waiting for packet on the testbed"
            return self._build_snapshot(vis, active_model, models_used, "Please place packet on the testbed.", False, False)

        # Real hand covering the packet prevents backside verification
        if hp or (len(pieces) >= 1 and np.any((pieces[0]["mask"] > 0) & (hand_mask > 0))):
            self._turn_motion_observed = True
            self._backside_ready_streak = 0
            self._strip_result = "UNKNOWN_HAND_OCCLUDED"
            self._packet_det_summary = "Hand covering packet — backside UNKNOWN"
            self._record_unverified_action("BACKSIDE_HAND_OCCLUDED", "Hand covering packet prevented backside verification.", self.stage.value)
            return self._build_snapshot(
                vis, active_model, models_used,
                "Hand obstructing packet — clear hands after turning to BACK side.",
                False, False,
            )

        pkt_mask = pieces[0]["mask"]
        pkt_centroid = pieces[0]["centroid"]
        pkt_area = float(pieces[0]["area"])
        moving = False
        if self._prev_pkt_centroid is not None:
            shift = float(np.hypot(pkt_centroid[0] - self._prev_pkt_centroid[0], pkt_centroid[1] - self._prev_pkt_centroid[1]))
            area_delta = abs(pkt_area - self._prev_pkt_area) / max(self._prev_pkt_area, 1.0)
            if shift >= 15.0 or area_delta > 0.15:
                self._turn_motion_observed = True
            if shift > 25.0 or area_delta > 0.25:
                moving = True
        self._prev_pkt_centroid = pkt_centroid
        self._prev_pkt_area = pkt_area

        if moving:
            self._backside_ready_streak = 0
            self._strip_result = "UNKNOWN_MOVING_PACKET"
            self._packet_det_summary = "Packet moving — hold steady on BACK side"
            return self._build_snapshot(vis, active_model, models_used, "Packet moving — hold steady on BACK side.", False, False)

        # Verify backside visual cues (turn motion, appearance difference, or grid disappearance)
        curr_hist, curr_grid = self._packet_appearance_signature(frame, pkt_mask)
        if self._frontside_ref_hist is not None:
            cos_sim = float(np.dot(curr_hist, self._frontside_ref_hist))
            appearance_diff = max(0.0, 1.0 - cos_sim)
        else:
            appearance_diff = 1.0
        grid_dropped = (self._frontside_grid_density > 0.03) and (curr_grid < self._frontside_grid_density * 0.65)
        backside_cues_ok = (
            (self._turn_motion_observed and appearance_diff >= 0.05)
            or appearance_diff >= self.pipe_cfg.min_backside_appearance_diff
            or grid_dropped
            or "backside_visually_verified" in ov
        )
        if not backside_cues_ok:
            self._backside_ready_streak = 0
            self._strip_result = "UNKNOWN_BACKSIDE_UNVERIFIED"
            self._packet_det_summary = "Waiting for packet turn to BACK side"
            return self._build_snapshot(vis, active_model, models_used, "Please turn the packet to the BACK side.", False, False)

        # 3. DIRECT STRIP INSPECTION (strip-seg.onnx ONLY):
        # Business rule: Strip visible = OPEN / NOT SEALED; Strip not visible = CLOSED / SEALED
        top_vis, top_mask = self._run_topstrip(frame, ov, models_used)
        if top_vis is None:
            self._backside_ready_streak = 0
            return self._build_snapshot(vis, active_model, models_used, "Waiting for Top Strip verification.", False, False)
        if top_vis:
            vis = draw_mask_overlay(vis, top_mask, (255, 0, 255), alpha=0.35)
            self._backside_ready_streak = 0
            self._strip_result = "STRIP_VISIBLE"
            self._packet_det_summary = "Top seal strip visible (OPEN / NOT SEALED)"
            self._instruction = "Strip visible. Cutting not ready."
            return self._build_snapshot(vis, active_model, models_used, "Strip visible. Cutting not ready.", False, False)

        # 4. TopStrip absent: verify closed orientation across stable consecutive frames
        self._backside_ready_streak += 1
        if self._backside_ready_streak < self.pipe_cfg.backside_confirm_frames:
            self._packet_det_summary = (
                f"Backside verified (strip absent) — stabilizing ({self._backside_ready_streak}/{self.pipe_cfg.backside_confirm_frames})"
            )
            return self._build_snapshot(
                vis, active_model, models_used,
                f"Backside detected — verifying closed orientation ({self._backside_ready_streak}/{self.pipe_cfg.backside_confirm_frames})...",
                False, False,
            )

        # Readiness confirmed! Store intact baseline and transition to Stage 4 (Cutting Activity)
        self._backside_verified = True
        self._strip_result = "STRIP_ABSENT_READY"
        self._intact_baseline_mask = pkt_mask.copy()
        self._intact_baseline_area = pkt_area
        self._packet_det_summary = f"Backside verified — intact baseline saved ({int(pkt_area)}px)"
        ev = self._save_evidence(vis, "stage3_backside_ready")
        assert self.procedure_monitor is not None
        self.procedure_monitor.on_orientation_verified(self._packet_id, self.stage.value)
        self.procedure_monitor.on_baseline_captured(self._packet_id, self._intact_baseline_area, self.stage.value)
        self._transition(
            AutoStage.WAIT_CUTTING_ACTIVITY,
            ts=ts,
            actual_event=f"Clear backside verified with TopStrip absent across {self._backside_ready_streak} fresh frames; intact mask saved ({pkt_area:.0f}px).",
            model_used=active_model,
            instruction="READY TO CUT",
            result="READY_TO_CUT",
            evidence_reference=ev,
        )
        return self._build_snapshot(vis, active_model, models_used, "READY TO CUT", True, False)

    # ------------------------------------------------------------------
    # STAGE 4: DISPLAY CUTTING ONLY DURING OBSERVED ACTIVITY
    # ------------------------------------------------------------------
    def _handle_stage4_cutting_activity(
        self, frame: np.ndarray, ts: float, ov: Dict[str, Any], models_used: List[str]
    ) -> AutoStepSnapshot:
        active_model = "hand-seg.onnx + scissor-seg.pt"
        vis = frame.copy()
        hp, hand_mask, sp, scissor_mask, _ = self._run_hand_and_scissor(
            frame, ov, models_used, full_view=True, require_clearance=self._cutting_activity_observed
        )
        vis = draw_mask_overlay(vis, hand_mask, (50, 50, 230), alpha=0.30)
        vis = draw_mask_overlay(vis, scissor_mask, (230, 180, 40), alpha=0.35)

        if hp and sp:
            self._cutting_both_streak += 1
            if self._cutting_both_streak >= self.pipe_cfg.cutting_both_confirm_frames:
                if not self._cutting_activity_observed:
                    self._cutting_activity_observed = True
                    ev = self._save_evidence(vis, "stage4_cutting_observed")
                    self._cutting_evidence = ev
                    assert self.procedure_monitor is not None
                    self.procedure_monitor.on_cutting_started(
                        self._packet_id, self._identity_status == "RESOLVED",
                        self._backside_verified, False, self.stage.value,
                    )
                    self.state_machine.record_event(
                        packet_id=self._packet_id,
                        actual_detected_event="Cutting activity started: Hand and scissors detected together in view.",
                        model_used=active_model,
                        confidence=1.0,
                        result="CUTTING_OBSERVED",
                        compliance_status="COMPLIANT",
                        evidence_reference=ev,
                    )
                self._instruction = "CUTTING ACTIVITY STARTED"
                self._latest_observed_action = "Cutting activity started (hand and scissors detected)."
                return self._build_snapshot(vis, active_model, models_used, "CUTTING ACTIVITY STARTED", False, False)
            return self._build_snapshot(
                vis, active_model, models_used,
                f"Hand + scissors detected ({self._cutting_both_streak}/{self.pipe_cfg.cutting_both_confirm_frames})...",
                False, False,
            )

        if self._cutting_activity_observed:
            if hp or sp:
                return self._build_snapshot(
                    vis, active_model, models_used,
                    "Waiting for both the hand and scissors to leave the inspection region.",
                    False, False,
                )
            ev = self._save_evidence(vis, "stage4_clear_after_cut")
            assert self.procedure_monitor is not None
            self.procedure_monitor.on_scene_cleared_after_cut(self._packet_id, self.stage.value)
            self._transition(
                AutoStage.VERIFY_PHYSICAL_CUT,
                ts=ts,
                actual_event="Hand and scissors left the inspection region after cutting activity.",
                model_used=active_model,
                instruction="Verifying the cut. Keep the inspection region clear.",
                result="CLEAR_VIEW_READY",
                evidence_reference=ev,
            )
            return self._build_snapshot(vis, active_model, models_used, self._instruction, True, False)

        # Hand alone or scissors alone NEVER starts cutting!
        self._cutting_both_streak = 0
        self._instruction = "READY TO CUT"
        if sp and not hp:
            msg = "Scissors alone detected — cutting has not started (hand + scissors required)."
        elif hp and not sp:
            msg = "Hand alone detected — cutting has not started (hand + scissors required)."
        else:
            msg = "READY TO CUT"
        return self._build_snapshot(vis, active_model, models_used, msg, False, False)

    # ------------------------------------------------------------------
    # STAGE 5: VERIFY THE PHYSICAL CUT WITH A CLEAR CAMERA VIEW
    # ------------------------------------------------------------------
    def _handle_stage5_verify_cut(
        self, frame: np.ndarray, ts: float, ov: Dict[str, Any], models_used: List[str]
    ) -> AutoStepSnapshot:
        active_model = f"{self._cut_model_filename} + packet-seg.onnx + OpenCV Gap"

        hp, _, sp, _, _ = self._run_hand_and_scissor(frame, ov, models_used, full_view=True, require_clearance=True)
        if hp or sp:
            self._cut_analysis_count = 0
            self._cut_valid_streak = 0
            self._cut_valid_samples_in_batch = 0
            self._cut_prev_sig = None
            return self._build_snapshot(
                frame.copy(), active_model, models_used, "Clear hands and scissors to restart cut analysis.", False, False
            )
        if self._cut_analysis_count >= self.pipe_cfg.cut_analysis_frames:
            self._cut_analysis_count = 0
            self._cut_valid_samples_in_batch = 0
        self._cut_analysis_count += 1

        p_info = self._run_packet_seg(frame, ov, models_used)
        cut_eval = self._run_cut_model(frame, ov, models_used)
        pieces = p_info["pieces"]
        ref_area = self._intact_baseline_area if self._intact_baseline_area > 0 else max(float(p_info["total_area"]), 1.0)

        gap_info: Optional[Dict[str, Any]] = None
        frame_valid = False
        reason = "CUT_UNVERIFIED"

        if len(pieces) != 2:
            reason = f"CUT_UNVERIFIED: Expected 2 detached pieces, found {len(pieces)}."
        else:
            a, b = pieces[:2]
            gap_info = packet_cut_core.analyze_packet_gap(a["mask"], b["mask"], frame=frame, reference_area=ref_area)
            area_ratio = (a["area"] + b["area"]) / max(ref_area, 1.0)
            if not gap_info["valid"] or gap_info.get("overlap_px", 0) > 0:
                reason = f"CUT_UNVERIFIED: Pieces touching or overlapping ({gap_info['reason']})."
            elif not (0.50 <= area_ratio <= 1.50):
                reason = f"CUT_UNVERIFIED: Combined piece area ratio {area_ratio:.2f} outside intact reference bounds."
            elif not cut_eval.get("is_cut_detected"):
                reason = f"CUT_UNVERIFIED: {self._cut_model_filename} did not confirm CUT ({cut_eval.get('class_name')}, {cut_eval.get('confidence', 0.0):.0%})."
            else:
                frame_valid = True

        if frame_valid and gap_info is not None:
            self._cut_valid_samples_in_batch += 1
            sig = packet_cut_core.piece_signature(pieces)
            if self._cut_prev_sig is None or packet_cut_core.is_stable_pair(self._cut_prev_sig, sig):
                self._cut_valid_streak += 1
            else:
                self._cut_valid_streak = 1
            self._cut_prev_sig = sig
        else:
            self._cut_valid_streak = 0
            self._cut_prev_sig = None

        vis = packet_cut_core.draw_verification_overlay(frame, pieces, gap_info)
        cut_confirmed = (
            self._cut_analysis_count >= self.pipe_cfg.cut_analysis_frames and (
                self._cut_valid_streak >= self.pipe_cfg.cut_confirm_frames or
                (self._cut_valid_samples_in_batch >= 6 and frame_valid)
            )
        )
        if cut_confirmed:
            reason = f"CUT_CONFIRMED: stable separated pieces and positive {self._cut_model_filename} evidence."
        self._cut_result = {
            "state": "CUT_CONFIRMED" if cut_confirmed else "CUT_UNVERIFIED",
            "cut_model_filename": self._cut_model_filename,
            "pieces_count": len(pieces),
            "two_pieces_confirmed": bool(frame_valid and len(pieces) == 2),
            "min_gap_px": round(float(gap_info["min_gap"]), 2) if gap_info else 0.0,
            "cut_model_confidence": round(float(cut_eval.get("confidence", 0.0)), 4),
            "stable_frames": self._cut_valid_streak,
            "analysis_frames": self._cut_analysis_count,
            "required_analysis_frames": self.pipe_cfg.cut_analysis_frames,
            "required_stable_frames": self.pipe_cfg.cut_confirm_frames,
            "evidence_image": self._cut_evidence_path,
            "reason": (
                f"Positive cut candidate; full confirmation pending: "
                f"{self._cut_analysis_count}/{self.pipe_cfg.cut_analysis_frames} analysis frames, "
                f"{self._cut_valid_streak}/{self.pipe_cfg.cut_confirm_frames} stable frames."
            ) if frame_valid and not cut_confirmed else reason,
        }

        # Preserve bounded partial evidence too. A missing confirmation must not
        # produce a blank report when a real cutting/cut candidate was observed.
        score = (100 + min(self._cut_valid_streak, self.pipe_cfg.cut_confirm_frames)
                 if frame_valid else min(len(pieces), 2))
        if (not cut_confirmed and pieces and (self._cutting_activity_observed or len(pieces) == 2)
                and score > self._cut_evidence_score):
            self._cut_evidence_score = score
            self._cut_evidence_path = self._save_evidence(vis, "stage5_cut_candidate_unverified")
            self._cut_result["evidence_image"] = self._cut_evidence_path

        if cut_confirmed and gap_info is not None:
            self._postcut_resting_centroids = [tuple(p["centroid"]) for p in pieces[:2]]
            ev = self._save_evidence(vis, "stage5_cut_confirmed")
            self._cut_evidence_path = ev
            self._cut_result["evidence_image"] = ev
            summary_reason = (
                f"CUT_CONFIRMED via {self._cut_model_filename} ({cut_eval['confidence']:.0%}), "
                f"2 detached pieces, gap={gap_info['min_gap']:.1f}px."
            )
            self._cut_result["reason"] = summary_reason
            assert self.procedure_monitor is not None
            self.procedure_monitor.on_cut_confirmed(self._packet_id, summary_reason, self.stage.value)
            self._transition(
                AutoStage.WAIT_UPRIGHT_MOUTH,
                ts=ts,
                actual_event=summary_reason,
                model_used=active_model,
                instruction="SHOW THE PACKET MOUTH TO THE CAMERA. Hold the packet upright.",
                confidence=float(cut_eval["confidence"]),
                result="CUT_CONFIRMED",
                evidence_reference=ev,
            )
            return self._build_snapshot(
                vis, active_model, models_used,
                "CUT_CONFIRMED. SHOW THE PACKET MOUTH TO THE CAMERA. Hold the packet upright.",
                True, False,
            )

        self._packet_det_summary = self._cut_result["reason"]
        return self._build_snapshot(
            vis, active_model, models_used,
            f"Analyzing cut ({self._cut_analysis_count}/{self.pipe_cfg.cut_analysis_frames} frames). "
            f"{self._cut_result['reason']}",
            False, False,
        )

    # ------------------------------------------------------------------
    # STAGE 6: WAIT FOR THE PACKET TO BE HELD UPRIGHT & INSPECT MOUTH
    # ------------------------------------------------------------------
    def _handle_stage6_upright_mouth(
        self, frame: np.ndarray, ts: float, ov: Dict[str, Any], models_used: List[str]
    ) -> AutoStepSnapshot:
        active_model = "packet-seg.onnx + jeweltrack.onnx"
        vis = frame.copy()

        # Stage 6 executes packet-seg.onnx and jeweltrack.onnx inside mouth cavity
        p_info = self._run_packet_seg(frame, ov, models_used)
        pieces = p_info["pieces"]
        hand_mask = (ov["hand_mask"] > 0).astype(np.uint8) * 255 if "hand_mask" in ov else np.zeros(frame.shape[:2], dtype=np.uint8)

        # Wait while both cut pieces remain flat on the bed!
        if len(pieces) == 2 and len(self._postcut_resting_centroids) == 2:
            c0, c1 = pieces[0]["centroid"], pieces[1]["centroid"]
            r0, r1 = self._postcut_resting_centroids[0], self._postcut_resting_centroids[1]
            d_direct = max(np.hypot(c0[0] - r0[0], c0[1] - r0[1]), np.hypot(c1[0] - r1[0], c1[1] - r1[1]))
            d_swap = max(np.hypot(c0[0] - r1[0], c0[1] - r1[1]), np.hypot(c1[0] - r0[0], c1[1] - r0[1]))
            if min(d_direct, d_swap) < self.pipe_cfg.vertical_lift_shift_px:
                self._mouth_open_streak = 0
                self._mouth_closed_streak = 0
                self._mouth_analysis_count = 0
                self._open_streak_gold_confidence = 0.0
                self._mouth_reason = "Both cut pieces remain flat on the testbed."
                return self._build_snapshot(
                    vis, active_model, models_used,
                    "SHOW THE PACKET MOUTH TO THE CAMERA. Hold the packet upright.",
                    False, False,
                )

        if len(pieces) == 0:
            self._mouth_open_streak = 0
            self._mouth_closed_streak = 0
            self._mouth_analysis_count = 0
            self._open_streak_gold_confidence = 0.0
            return self._build_snapshot(
                vis, active_model, models_used,
                "SHOW THE PACKET MOUTH TO THE CAMERA. Hold the packet upright.",
                False, False,
            )

        held_mask = pieces[0]["mask"]
        vis = draw_mask_overlay(vis, held_mask, (255, 140, 40), alpha=0.20)

        # Determine mouth OPEN / CLOSED / UNKNOWN from visible mouth edges & cavity (NOT whole-packet mask alone!)
        if "mouth_state_override" in ov:
            mouth_raw = str(ov["mouth_state_override"])
            mouth_reason = str(ov.get("mouth_reason_override", f"Mouth state: {mouth_raw}"))
            cavity_mask = ov.get("mouth_cavity_mask", np.zeros(frame.shape[:2], dtype=np.uint8))
        else:
            mouth_raw, mouth_reason, cavity_mask = self._inspect_packet_mouth(frame, held_mask, hand_mask)

        if cavity_mask is not None and np.any(cavity_mask > 0):
            vis = draw_mask_overlay(vis, cavity_mask, (230, 210, 40), alpha=0.28)

        # Record gold visibility SEPARATELY inside the exposed mouth cavity (an open packet can be empty!)
        gold_vis: Optional[bool] = None
        gold_conf: float = 0.0
        if mouth_raw == "OPEN" and cavity_mask is not None and np.any(cavity_mask > 0):
            inside_gold = []
            if "mouth_jewel_instances" in ov:
                models_used.append("jeweltrack.onnx")
                j_insts = list(ov["mouth_jewel_instances"])
            elif self.model_manager.has("jewel"):
                def _jcall() -> Optional[List[RawSegInstance]]:
                    res = self.model_manager.get("jewel").predict(
                        source=frame, imgsz=self.wf_cfg.models.imgsz,
                        conf=self.insp_cfg.packet_inspection.jewel_conf,
                        device=self.model_manager.device, verbose=False,
                    )[0]
                    return extract_yolo_instances(
                        res, frame.shape[:2], selected_ids=self.model_manager.get_class_ids("jewel"),
                        min_conf=self.insp_cfg.packet_inspection.jewel_conf, min_area_px=60,
                    )
                j_insts = self._safe_infer("jeweltrack.onnx", _jcall, None, models_used)
            else:
                j_insts = None

            if j_insts:
                target_region = (held_mask > 0) if held_mask is not None else np.ones(frame.shape[:2], dtype=bool)
                inside_gold = [
                    inst for inst in j_insts
                    if int(np.count_nonzero((inst.mask > 0) & target_region)) >= 40
                ]

            if inside_gold:
                gold_vis = True
                gold_conf = max(i.confidence for i in inside_gold)
                vis = draw_mask_overlay(vis, union_masks([i.mask for i in inside_gold], frame.shape[:2]), (80, 220, 140), alpha=0.45)
            else:
                # Multi-space gold chromaticity & texture analysis within held packet / mouth
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                glare = (hsv[:, :, 2] >= 245) & (hsv[:, :, 1] <= 45)
                target_mask = (held_mask > 0) if held_mask is not None and np.any(held_mask > 0) else (cavity_mask > 0)
                gold_chroma = (
                    (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 38) &
                    (hsv[:, :, 1] >= 50) & (hsv[:, :, 2] >= 45) &
                    (lab[:, :, 2] >= 135) & (~glare) & target_mask
                )
                gold_chroma_px = int(np.count_nonzero(gold_chroma))
                if gold_chroma_px >= 120:
                    gold_vis = True
                    gold_conf = min(0.95, 0.50 + float(gold_chroma_px) / 2000.0)
                    vis = draw_mask_overlay(vis, (gold_chroma.astype(np.uint8) * 255), (80, 220, 140), alpha=0.45)
                elif j_insts is not None or cavity_mask is not None:
                    gold_vis = False
                    gold_conf = 0.0

        if self._mouth_analysis_count >= self.pipe_cfg.mouth_analysis_frames:
            self._mouth_analysis_count = 0
            self._mouth_open_samples_in_batch = 0
            self._mouth_closed_samples_in_batch = 0
            self._batch_gold_seen = False

        if self._mouth_analysis_count or mouth_raw in {"OPEN", "CLOSED"}:
            self._mouth_analysis_count += 1
        self._mouth_open_streak = self._mouth_open_streak + 1 if mouth_raw == "OPEN" else 0
        self._mouth_closed_streak = self._mouth_closed_streak + 1 if mouth_raw == "CLOSED" else 0
        if mouth_raw == "OPEN":
            self._mouth_open_samples_in_batch += 1
            if gold_vis is True:
                self._batch_gold_seen = True
        elif mouth_raw == "CLOSED":
            self._mouth_closed_samples_in_batch += 1

        self._open_streak_gold_confidence = (
            max(self._open_streak_gold_confidence, gold_conf) if mouth_raw == "OPEN" else 0.0
        )
        has_gold_info = (gold_vis is not None) or self._batch_gold_seen
        open_verified = (
            (self._mouth_open_streak >= self.pipe_cfg.mouth_confirm_frames or self._mouth_open_samples_in_batch >= 6)
            and has_gold_info
        )
        closed_verified = (
            self._mouth_closed_streak >= self.pipe_cfg.mouth_confirm_frames or
            self._mouth_closed_samples_in_batch >= 6
        )

        if self._mouth_analysis_count >= self.pipe_cfg.mouth_analysis_frames and (open_verified or closed_verified):
            self._mouth_status = "OPEN" if open_verified else "CLOSED"
            self._mouth_reason = mouth_reason
            if open_verified:
                self._gold_visibility_in_mouth = (self._open_streak_gold_confidence > 0.0 or self._batch_gold_seen or gold_vis is True)
                self._gold_mouth_confidence = max(self._open_streak_gold_confidence, 0.85 if (self._batch_gold_seen or gold_vis is True) else 0.0)
            else:
                self._gold_visibility_in_mouth = None
                self._gold_mouth_confidence = 0.0
            self._mouth_evidence = self._save_evidence(vis, f"stage6_mouth_{self._mouth_status.lower()}")
            assert self.procedure_monitor is not None
            self.procedure_monitor.on_post_cut_packet_status_confirmed(
                self._packet_id, self._mouth_status, self._gold_visibility_in_mouth, self.stage.value
            )
            self.jewel_tracker.reset()
            self._provisional_jewel_count = 0
            self._confirmed_unambiguous_count = 0
            self._jewel_count_status = "WAITING_PLACEMENT"
            result = self._mouth_result_text()
            instruction = "Remove the gold from the packet and place it on the testbed."
            self._transition(
                AutoStage.WAIT_GOLD_PLACEMENT,
                ts=ts,
                actual_event=result,
                model_used=active_model,
                instruction=instruction,
                result=f"MOUTH_{self._mouth_status}",
                evidence_reference=self._mouth_evidence,
            )
            return self._build_snapshot(vis, active_model, models_used, f"{result}. {instruction}", True, False)

        if 0 < self._mouth_analysis_count < self.pipe_cfg.mouth_analysis_frames:
            msg = f"Analyzing packet mouth ({self._mouth_analysis_count}/{self.pipe_cfg.mouth_analysis_frames} frames). Hold it steady."
        elif mouth_raw == "OPEN" and not has_gold_info:
            msg = "Keep the inside of the packet visible for the gold check."
        else:
            msg = "SHOW THE PACKET MOUTH TO THE CAMERA. Hold the packet upright."
        return self._build_snapshot(vis, active_model, models_used, msg, False, False)

    def _mouth_result_text(self) -> str:
        if self._mouth_status == "OPEN":
            visibility = "VISIBLE" if self._gold_visibility_in_mouth else "NOT VISIBLE"
            return f"Mouth OPEN | Gold {visibility}"
        return f"Mouth {self._mouth_status}"

    # ------------------------------------------------------------------
    # STAGE 7: WAIT FOR JEWELLERY PLACEMENT & COUNT EACH ARTICLE (PROVISIONAL)
    # ------------------------------------------------------------------
    def _handle_stage7_jewellery_counting(
        self, frame: np.ndarray, ts: float, ov: Dict[str, Any], models_used: List[str]
    ) -> AutoStepSnapshot:
        active_model = "jeweltrack.onnx + hand-seg.onnx + packet-seg.onnx"
        tb_roi = self.testbed_roi.get_roi(frame)

        p_info = self._run_packet_seg(frame, ov, models_used)
        hp, hand_mask, _, _, _ = self._run_hand_and_scissor(frame, ov, models_used, run_scissor=False)

        # Detect jewellery inside testbed ROI (automatically excludes packet interior & outside ROI)
        models_used.append("jeweltrack.onnx")
        det_res = self.jewel_detector.detect(
            frame,
            tb_roi,
            jewel_instances_override=ov.get("jewel_instances"),
            hand_mask_override=hand_mask,
            packet_mask_override=p_info["union_mask"],
        )

        # Additionally exclude jewellery still held IN THE HAND!
        placed_detections: List[RawSegInstance] = []
        excluded_in_hand = 0
        hand_bool = hand_mask > 0
        for inst in det_res.detections:
            if np.any(hand_bool):
                in_hand = int(np.count_nonzero((inst.mask > 0) & hand_bool))
                if float(in_hand) / float(max(1, inst.area)) >= 0.30:
                    excluded_in_hand += 1
                    continue
            placed_detections.append(inst)

        # Outside-testbed alert
        if det_res.excluded_outside_roi > 0:
            self._outside_testbed_streak += 1
            if self._outside_testbed_streak >= self.pipe_cfg.outside_testbed_alert_frames:
                assert self.procedure_monitor is not None
                self.procedure_monitor.on_jewellery_outside_testbed(
                    self._packet_id, det_res.excluded_outside_roi, self.stage.value, frame=frame, now_sec=ts
                )
        else:
            self._outside_testbed_streak = 0

        # Separate touching articles & detect overlap uncertainty
        sep_res = self.jewel_separator.separate(frame, placed_detections)
        trk_snap = self.jewel_tracker.update(sep_res.articles)

        # Confirmed unambiguous count = tracks whose masks are NOT uncertain
        confirmed_unambiguous = sum(1 for t in trk_snap.confirmed_tracks if not t.uncertain)
        total_live = trk_snap.live_count

        if sep_res.is_uncertain:
            self._jewel_uncertainty_recorded = True
            for r in sep_res.uncertainty_reasons:
                if r not in self._jewel_uncertainty_reasons:
                    self._jewel_uncertainty_reasons.append(r)
            self._uncertain_count_streak += 1
            if self._uncertain_count_streak >= self.pipe_cfg.uncertain_count_alert_frames:
                assert self.procedure_monitor is not None
                self.procedure_monitor.on_jewellery_count_uncertain(
                    self._packet_id, "; ".join(sep_res.uncertainty_reasons[:2]), self.stage.value, frame=frame, now_sec=ts
                )
        else:
            self._uncertain_count_streak = 0

        # Count stability filter:
        # If hands are present, do NOT update provisional count with occluded/fragmented detections.
        # Wait for hands to clear and count to be stable across consecutive frames.
        if hp:
            self._count_stable_streak = 0
            if self._provisional_jewel_count == 0:
                self._jewel_count_status = "WAITING_PLACEMENT"
                msg = "Remove the gold from the packet and place it on the testbed. Clear hands for stable counting."
            else:
                msg = f"Hand in work area — clear hands for a stable jewellery count (stable count: {self._provisional_jewel_count})."
        else:
            if total_live == self._count_candidate:
                self._count_stable_streak += 1
            else:
                self._count_candidate = total_live
                self._count_stable_streak = 1

            self._provisional_jewel_count = total_live
            self._confirmed_unambiguous_count = confirmed_unambiguous

            if total_live == 0:
                self._jewel_count_status = "WAITING_PLACEMENT"
                msg = "Remove the gold from the packet and place it on the testbed."
            elif sep_res.is_uncertain:
                self._jewel_count_status = "PROVISIONAL_WITH_UNCERTAINTY"
                msg = (
                    f"Overlap uncertainty recorded — confirmed unambiguous count: {confirmed_unambiguous} "
                    "(separate overlapping pieces)."
                )
            else:
                self._jewel_count_status = "PROVISIONAL"
                msg = (
                    f"Provisional count: {total_live} article(s) on testbed "
                    "(session stays active across pauses; press STOP when finished)."
                )

        # Draw tracked articles
        vis = frame.copy()
        for trk in trk_snap.confirmed_tracks:
            color = (40, 130, 255) if trk.uncertain else (50, 230, 130)
            vis = draw_mask_overlay(vis, trk.mask, color, alpha=0.35)
            bx1, by1, bx2, by2 = trk.bbox
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
            lbl = f"{trk.track_id} [UNCERTAIN]" if trk.uncertain else trk.track_id
            cv2.putText(vis, lbl, (bx1, max(18, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        # Save numbered overlays matching the reported count, including an empty
        # bed after articles are removed. Keep the last hand-free evidence while occluded.
        if self._observer is None and not hp and (total_live > 0 or self.stage == AutoStage.WAIT_JEWELLERY_AND_COUNT):
            if (self._jewel_evidence is None or self._jewel_evidence_count != total_live
                    or self._jewel_evidence_status != self._jewel_count_status):
                cv2.putText(vis, f"Packet {self._packet_id} | Count: {total_live}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 160, 30), 2)
                self._jewel_evidence = self._save_evidence(vis, f"stage7_provisional_{total_live}")
                self._jewel_evidence_count = total_live
                self._jewel_evidence_status = self._jewel_count_status

        # IMPORTANT (Requirement #7 & #8):
        # A pause after placing the first article MUST leave the count PROVISIONAL.
        # No observable completion event exists in the project, so NEVER prematurely finalize!
        placement_confirmed = (self.stage == AutoStage.WAIT_GOLD_PLACEMENT and confirmed_unambiguous > 0)
        if placement_confirmed:
            assert self.procedure_monitor is not None
            self.procedure_monitor.on_jewellery_placed_inside_testbed(
                self._packet_id, confirmed_unambiguous, self.stage.value, frame=frame
            )
            self._transition(
                AutoStage.WAIT_JEWELLERY_AND_COUNT,
                ts=ts,
                actual_event="Gold placement confirmed on the testbed.",
                model_used=active_model,
                instruction=STATE_MACHINE_SPECS[AutoStage.WAIT_JEWELLERY_AND_COUNT].display_instruction,
                result="GOLD_PLACEMENT_CONFIRMED",
                evidence_reference=self._jewel_evidence,
            )
        elif confirmed_unambiguous > 0 and self.procedure_monitor is not None:
            step = self.procedure_monitor._steps.get("JEWELLERY_TESTBED_PLACEMENT")
            if step and step.status != "COMPLIANT":
                self.procedure_monitor.on_jewellery_placed_inside_testbed(
                    self._packet_id, confirmed_unambiguous, self.stage.value, frame=frame
                )
        return self._build_snapshot(vis, active_model, models_used, msg, placement_confirmed, False)

    # ------------------------------------------------------------------
    # STAGE 8: STOP & SAVE FULL / PARTIAL RESULT
    # ------------------------------------------------------------------
    def stop_session(self, now_sec: Optional[float] = None) -> Dict[str, Any]:
        """STOP always saves all evidence and full/partial results, including stage reached."""
        self.camera_status = "STOPPED"
        if self._final_report is not None:
            return self._final_report
        ts = time.monotonic() if now_sec is None else float(now_sec)
        stage_reached = self.stage.value

        if self.stage != AutoStage.STOPPED_OR_SAVED:
            self.state_machine.transition(
                AutoStage.STOPPED_OR_SAVED,
                packet_id=self._packet_id,
                actual_detected_event=f"STOP pressed at stage '{stage_reached}' — saving full/partial result.",
                model_used="None",
                confidence=1.0,
                result="SAVED_ON_STOP",
                compliance_status="INFO",
            )

        assert self.alert_manager is not None and self.activity_logger is not None and self.procedure_monitor is not None
        self.procedure_monitor.finalize(self._packet_id, stage_reached)

        from workflow_monitor.ocr_protector import OCRIdentitySummary
        ocr_sum = OCRIdentitySummary(
            packet_id=self._packet_id,
            identity_status=self._identity_status,
            confidence=self._ocr_confidence,
            total_attempts=self.recovery.ocr_attempts,
            evidence_image=self._ocr_evidence,
        )
        post_cut_dict = {
            "status": self._mouth_status,
            "mouth_status": self._mouth_status,
            "gold_visible": self._gold_visibility_in_mouth,
            "gold_confidence": round(self._gold_mouth_confidence, 4),
            "inspection_valid": self._mouth_status in ("OPEN", "CLOSED"),
            "reason": self._mouth_reason,
            "evidence_image": self._mouth_evidence,
        }
        jewel_dict = {
            "count": self._provisional_jewel_count,
            "provisional_count": self._provisional_jewel_count,
            "confirmed_unambiguous_count": self._confirmed_unambiguous_count,
            "count_status": self._jewel_count_status,
            "uncertainty_recorded": self._jewel_uncertainty_recorded,
            "uncertainty_reasons": list(self._jewel_uncertainty_reasons),
            "evidence_image": self._jewel_evidence,
            "reason": MISSING_COMPLETION_EVENT_NOTE,
        }

        if self._observer is not None:
            jewel_dict.update(self._observer.count_result())
        report = FinalReportGenerator.build_report(
            session_id=self.session_id,
            started_at=self.started_at,
            completed_at=utc_now_iso(),
            duration_seconds=max(0.0, ts - self.started_mono),
            ocr_summary=ocr_sum,
            cut_result=self._cut_result,
            post_cut_packet_result=post_cut_dict,
            jewellery_result=jewel_dict,
            procedure_monitor=self.procedure_monitor,
            alert_manager=self.alert_manager,
            activity_logger=self.activity_logger,
            processing_completed=False,
            model_report=self.model_manager.availability_report(),
        )
        if self._observer is not None:
            report["flow_mode"] = "PASS_BY"
            report["release_checks"] = list(self._observer.checks.values())
            report["final_summary"]["observation_session_saved"] = True
            report["final_summary"]["summary_text"] = (
                "Release observation saved. Missing checks did not block other observations. "
                "Review each check's status; no physical release approval is inferred."
            )
        report["session_information"]["stage_reached"] = stage_reached
        report["operational_results"]["empty_testbed_verified"] = self._empty_verified
        report["operational_results"]["frontside_verified"] = self._frontside_verified
        report["operational_results"]["backside_verified"] = self._backside_verified
        report["operational_results"]["strip_result"] = self._strip_result
        report["operational_results"]["backside_limitation_note"] = BACKSIDE_CLASSIFIER_LIMITATION
        report["operational_results"]["cutting_activity_observed"] = self._cutting_activity_observed
        report["operational_results"]["cutting_activity"] = {
            "status": "CUTTING_OBSERVED" if self._cutting_activity_observed else "NOT_VERIFIED",
            "evidence_image": self._cutting_evidence,
            "reason": "Hand and scissors observed together." if self._cutting_activity_observed else "Cutting action not observed.",
        }
        report["operational_results"]["cut_model_filename"] = self._cut_model_filename
        report["operational_results"]["missing_completion_event_rule"] = MISSING_COMPLETION_EVENT_NOTE
        report["observed_mistakes"] = list(self._observed_mistakes)
        report["camera_unverifiable_actions"] = list(self._unverified_actions)
        report["evidence_images"] = list(self._evidence_references)
        report["operational_results"]["ocr_identification"]["number_crop_image"] = self._ocr_crop_evidence
        report["operational_results"]["ocr_identification"]["timing"] = dict(self._ocr_timing)
        report["evidence_media"] = self._media_recorder.finish(ts) if self._media_recorder else []
        report["media_errors"] = list(self._media_recorder.errors) if self._media_recorder else []
        report["stage_transition_log"] = self.state_machine.to_list()
        report["recovery_attempts"] = self.recovery.to_dict()
        self._final_report = report

        if self.pipe_cfg.save_outputs:
            out_dir = Path(self.pipe_cfg.output_dir) / self.session_id
            self._report_paths = FinalReportGenerator.save_reports(report, out_dir, save_json=True, save_markdown=True)
        return report

    def _build_snapshot(
        self,
        vis: np.ndarray,
        active_model: str,
        models_used: List[str],
        status_msg: str,
        transitioned: bool,
        completed: bool,
    ) -> AutoStepSnapshot:
        assert self.alert_manager is not None and self.procedure_monitor is not None
        comp = self.procedure_monitor.compliance_metrics()

        count_display = (
            f"Confirmed: {self._confirmed_unambiguous_count} (UNCERTAIN OVERLAP)"
            if self._jewel_count_status == "PROVISIONAL_WITH_UNCERTAINTY"
            else (
                f"{self._provisional_jewel_count} (PROVISIONAL)"
                if self._jewel_count_status == "PROVISIONAL"
                else "0 (waiting)"
            )
        )

        if self._observer is not None:
            visible = self._observer.visible_count
            retained = self._observer.last_clear_count
            count_display = f"Visible: {visible if visible is not None else 'unverified'} | Retained clear: {retained if retained is not None else 'unverified'}"

        if self._mouth_status in {"OPEN", "CLOSED"}:
            cond_display = self._mouth_result_text()
        elif self.stage == AutoStage.WAIT_UPRIGHT_MOUTH:
            cond_display = "Waiting for mouth verification"
        elif self._strip_result == "STRIP_VISIBLE":
            cond_display = "OPEN / NOT SEALED (Strip visible)"
        elif self._backside_verified:
            cond_display = "CLOSED / SEALED (Strip absent)"
        else:
            cond_display = "Waiting for Top Strip verification"

        if self._observer is not None and cond_display.startswith("Waiting"):
            cond_display = "Not verified (observation continues)"

        live_monitoring = {
            "session_id": self.session_id,
            "stage": self.stage.value,
            "instruction": self._instruction,
            "packet_id": self._packet_id,
            "identity_status": self._identity_status,
            "active_model": active_model,
            "processing_status": status_msg,
            "packet_detection": self._packet_det_summary,
            "cutting_verification": f"{self._cut_result.get('state')} [{self._cut_model_filename}] ({self._cut_result.get('min_gap_px', 0.0):.1f}px)",
            "packet_condition": cond_display,
            "jewellery_count": count_display,
        }
        compliance_monitoring = {
            "latest_observed_action": self._latest_observed_action,
            "expected_procedure": STATE_MACHINE_SPECS[self.stage].expected_physical_action,
            "instruction": self._instruction,
            "active_alerts": [a.to_dict() for a in self.alert_manager.active_alerts],
            "correct_actions": comp["correctly_followed_procedures"],
            "total_expected": comp["total_expected_procedures"],
            "skipped_actions": comp["skipped_procedures"],
            "confirmed_deviations": comp["confirmed_deviations"],
            "corrected_mistakes": comp["corrected_mistakes"],
            "procedure_steps": comp["procedure_steps"],
        }
        avail = self.model_manager.availability_report()
        system_monitoring = {
            "model_loading_status": f"{len(avail['loaded'])} loaded ({avail['device']}) | Cut: {self._cut_model_filename}",
            "active_inference_model": active_model,
            "camera_status": self.camera_status,
            "processing_errors": f"{self.recovery.total_inference_errors} ({self.recovery.last_error})",
            "automatic_recovery_status": (
                f"Retries: {self.recovery.total_retries} | OCR attempts: {self.recovery.ocr_attempts} | "
                f"{self.recovery.last_recovery}"
            ),
        }

        return AutoStepSnapshot(
            session_id=self.session_id,
            stage=self.stage,
            active_model=active_model,
            models_executed_this_frame=list(models_used),
            instruction=self._instruction,
            annotated_frame=vis,
            live_monitoring=live_monitoring,
            compliance_monitoring=compliance_monitoring,
            system_monitoring=system_monitoring,
            transitioned_this_frame=transitioned,
            completed=completed,
            final_report=self._final_report,
            report_paths=dict(self._report_paths),
        )
