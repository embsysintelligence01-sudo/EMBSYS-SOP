"""Event-Driven Automated Workflow Engine (Sections 1–10).

Unites two independent responsibilities in a single-START automated loop:
A. Automated Packet Processing (packet detection, continuous OCR protection,
   orientation/seal check, automatic intact baseline, cutting detection,
   physical separation verification, post-cut OPEN/CLOSED inspection, and
   testbed jewellery detection/separation/counting + auto-completion).
B. Independent Appraiser Procedure Monitoring (SOP comparison, deviation &
   skip/out-of-order detection, real-time alerts with suppression, self-correction
   tracking, chronological activity log, and automatic comprehensive report).

No manual stage-progression buttons are required after START.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import packet_cut_core
from inspection.inspection_result import JewelleryCountingOutcome, PacketInspectionOutcome
from inspection.jewellery_counting.count_manager import JewelleryCountManager
from inspection.packet_inspection.packet_status import PacketStatusEvaluator
from inspection.utils.image_utils import RawSegInstance, draw_mask_overlay, extract_yolo_instances, union_masks
from inspection.utils.inspection_config import InspectionConfig
from inspection.utils.model_loader import InspectionModelRegistry
from packet_cut.cut_model import evaluate_cut_model_prediction

from .activity_log import ActivityEvent, ActivityLogger, ComplianceStatus
from .alerts import AlertRecord, RealTimeAlertManager, utc_now_iso
from .config import WorkflowMonitorConfig
from .models import UnifiedModelManager
from .ocr_protector import OCRCallable, OCRProtector, OCRStepObservation
from .procedure_monitor import AppraiserProcedureMonitor
from .report_generator import FinalReportGenerator


class AutomatedStage(str, Enum):
    IDLE = "IDLE"
    WAITING_FOR_PACKET = "WAITING_FOR_PACKET"
    PACKET_IDENTIFICATION = "PACKET_IDENTIFICATION"
    ORIENTATION_AND_BASELINE = "ORIENTATION_AND_BASELINE"
    CUTTING_MONITOR = "CUTTING_MONITOR"
    CUT_VERIFICATION = "CUT_VERIFICATION"
    POST_CUT_PACKET_INSPECTION = "POST_CUT_PACKET_INSPECTION"
    TESTBED_EMPTY_CHECK = "TESTBED_EMPTY_CHECK"
    JEWELLERY_COUNTING = "JEWELLERY_COUNTING"
    SESSION_COMPLETED = "SESSION_COMPLETED"


@dataclass
class AutomatedStepResult:
    session_id: str
    stage: AutomatedStage
    packet_id: str
    identity_status: str
    status_message: str
    annotated_frame: np.ndarray
    active_alerts: List[Dict[str, Any]]
    compliance_summary: Dict[str, Any]
    operational_snapshot: Dict[str, Any]
    completed: bool = False
    final_report: Optional[Dict[str, Any]] = None
    report_paths: Dict[str, Optional[str]] = field(default_factory=dict)


class AutomatedWorkflowEngine:
    """Fully automated, event-driven workflow & compliance orchestrator."""

    def __init__(
        self,
        workflow_config: Optional[WorkflowMonitorConfig] = None,
        inspection_config: Optional[InspectionConfig] = None,
        preloaded_models: Optional[Dict[str, Any]] = None,
        auto_load_models: bool = True,
        ocr_runner: Optional[OCRCallable] = None,
        on_alert: Optional[Any] = None,
        on_alert_resolved: Optional[Any] = None,
        on_activity: Optional[Any] = None,
    ) -> None:
        self.wf_cfg = workflow_config if workflow_config is not None else WorkflowMonitorConfig.load()
        self.insp_cfg = inspection_config if inspection_config is not None else InspectionConfig.load()
        self.proc_cfg = self.wf_cfg.automated_processing

        # Align Stage 2 auto-completion stability settings
        self.insp_cfg.jewellery_counting.stability.required_stable_frames = self.proc_cfg.auto_complete_stable_frames
        self.insp_cfg.jewellery_counting.stability.min_stable_seconds = self.proc_cfg.auto_complete_min_seconds

        self.model_manager = UnifiedModelManager(self.wf_cfg, preloaded_models=preloaded_models)
        if auto_load_models:
            self.model_manager.ensure_loaded()

        # Post-cut registry reuses the already-loaded packet/jewel/hand models (never TopStrip!)
        self.post_cut_registry = InspectionModelRegistry(
            self.insp_cfg,
            shared_models=self.model_manager.post_cut_shared_models(),
        )

        self._on_alert_cb = on_alert
        self._on_alert_resolved_cb = on_alert_resolved
        self._on_activity_cb = on_activity
        self._ocr_runner = ocr_runner

        self.ocr_protector = OCRProtector(
            self.wf_cfg,
            model_manager=self.model_manager,
            ocr_runner=ocr_runner,
        )
        self.stage1_evaluator = PacketStatusEvaluator(self.insp_cfg, self.post_cut_registry)
        self.stage2_manager = JewelleryCountManager(self.insp_cfg, self.post_cut_registry)

        # Session runtime state
        self.session_id: str = ""
        self.stage: AutomatedStage = AutomatedStage.IDLE
        self.started_at: str = ""
        self.started_mono: float = 0.0
        self.alert_manager: Optional[RealTimeAlertManager] = None
        self.activity_logger: Optional[ActivityLogger] = None
        self.procedure_monitor: Optional[AppraiserProcedureMonitor] = None

        # Pre-cut / cutting runtime state
        self._reset_session_internals()

    def _reset_session_internals(self) -> None:
        self._ocr_missing_packet_frames: int = 0
        self._orientation_closed_streak: int = 0
        self._orientation_verified: bool = False
        self._baseline_mask: Optional[np.ndarray] = None
        self._baseline_area: float = 0.0
        self._baseline_streak: int = 0

        self._hand_streak: int = 0
        self._scissor_streak: int = 0
        self._cutting_started: bool = False
        self._clear_started_mono: Optional[float] = None

        self._verify_started_mono: Optional[float] = None
        self._next_verify_sample_mono: float = 0.0
        self._verify_samples: int = 0
        self._verify_valid_samples: int = 0
        self._verify_stable_streak: int = 0
        self._verify_prev_sig: Optional[Any] = None
        self._last_pieces: List[Dict[str, Any]] = []
        self._last_gap: Optional[Dict[str, Any]] = None
        self._cut_result: Dict[str, Any] = {
            "state": "NOT_STARTED",
            "pieces_count": 0,
            "min_gap_px": 0.0,
            "cut_model_confidence": 0.0,
            "reason": "Cutting not yet verified.",
        }

        self._post_cut_unknown_streak: int = 0
        self._post_cut_outcome: PacketInspectionOutcome = PacketInspectionOutcome()
        self._outside_testbed_streak: int = 0
        self._uncertain_count_streak: int = 0
        self._jewel_placed_logged: bool = False
        self._jewellery_outcome: JewelleryCountingOutcome = JewelleryCountingOutcome()

        self._final_report: Optional[Dict[str, Any]] = None
        self._report_paths: Dict[str, Optional[str]] = {}

    def start_session(self, now_sec: Optional[float] = None) -> str:
        """Single manual trigger: initializes an automated monitoring session."""
        self.session_id = f"SES-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        self.started_at = utc_now_iso()
        self.started_mono = time.monotonic() if now_sec is None else float(now_sec)
        self.stage = AutomatedStage.WAITING_FOR_PACKET

        self.ocr_protector.reset()
        self.stage1_evaluator.reset()
        self.stage2_manager.reset(require_empty_check=True)
        self._reset_session_internals()

        evidence_dir = self.wf_cfg.output.resolved_output_dir() / self.session_id / "evidence"
        self.alert_manager = RealTimeAlertManager(
            session_id=self.session_id,
            suppression_cooldown_seconds=self.wf_cfg.alerts.suppression_cooldown_seconds,
            on_alert=self._on_alert_cb,
            on_resolved=self._on_alert_resolved_cb,
        )
        self.activity_logger = ActivityLogger(
            session_id=self.session_id,
            evidence_dir=evidence_dir,
            save_evidence_images=self.wf_cfg.output.save_evidence_images,
            on_event=self._on_activity_cb,
        )
        self.procedure_monitor = AppraiserProcedureMonitor(
            session_id=self.session_id,
            alert_manager=self.alert_manager,
            activity_logger=self.activity_logger,
        )

        self.activity_logger.log(
            packet_id="UNRESOLVED",
            stage=self.stage.value,
            procedure_step="SESSION_START",
            expected_action="Initialize automated inspection session after START.",
            actual_action="Session started automatically; monitoring camera for packet presentation.",
            compliance_status=ComplianceStatus.INFO,
        )
        return self.session_id

    def _require_started(self) -> Tuple[RealTimeAlertManager, ActivityLogger, AppraiserProcedureMonitor]:
        if self.alert_manager is None or self.activity_logger is None or self.procedure_monitor is None:
            raise RuntimeError("Call start_session() before processing frames.")
        return self.alert_manager, self.activity_logger, self.procedure_monitor

    # ==================================================================
    # PERCEPTION HELPERS (real models with optional test overrides)
    # ==================================================================
    def _predict_packet_pieces(
        self,
        frame: np.ndarray,
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        if "packet_pieces" in overrides:
            return overrides["packet_pieces"]
        if "packet_mask" in overrides:
            m = (overrides["packet_mask"] > 0).astype(np.uint8) * 255
            num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
            pieces = []
            for lbl in range(1, num):
                area = int(stats[lbl, cv2.CC_STAT_AREA])
                if area < packet_cut_core.MIN_PIECE_AREA_PX:
                    continue
                pm = np.zeros_like(m)
                pm[labels == lbl] = 255
                x, y, bw, bh = stats[lbl, cv2.CC_STAT_LEFT], stats[lbl, cv2.CC_STAT_TOP], stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT]
                pieces.append({
                    "mask": pm,
                    "area": area,
                    "centroid": (float(centroids[lbl][0]), float(centroids[lbl][1])),
                    "bbox": (int(x), int(y), int(bw), int(bh)),
                })
            pieces.sort(key=lambda p: p["area"], reverse=True)
            return {
                "pieces": pieces,
                "rejected_pieces": [],
                "union_mask": m,
                "total_area": int(np.count_nonzero(m)),
                "yolo_instance_count": len(pieces),
                "raw_cc_count": len(pieces),
                "accepted_piece_count": len(pieces),
                "rejected_piece_count": 0,
                "rejection_summary": [],
            }

        if not self.model_manager.has("packet"):
            empty = np.zeros(frame.shape[:2], dtype=np.uint8)
            return {
                "pieces": [],
                "rejected_pieces": [],
                "union_mask": empty,
                "total_area": 0,
                "yolo_instance_count": 0,
                "raw_cc_count": 0,
                "accepted_piece_count": 0,
                "rejected_piece_count": 0,
                "rejection_summary": [],
            }

        res = self.model_manager.get("packet").predict(
            source=frame,
            imgsz=self.wf_cfg.models.imgsz,
            conf=self.proc_cfg.packet_conf,
            device=self.model_manager.device,
            verbose=False,
        )[0]
        return packet_cut_core.extract_packet_pieces(
            res,
            frame.shape[:2],
            packet_ids=self.model_manager.get_class_ids("packet"),
            reference_area=self._baseline_area if self._baseline_area > 0 else None,
            min_conf=self.proc_cfg.packet_conf,
            verbose=False,
        )

    def _predict_hand_and_scissor(
        self,
        frame: np.ndarray,
        overrides: Dict[str, Any],
    ) -> Tuple[bool, np.ndarray, bool, np.ndarray]:
        h, w = frame.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)

        if "hand_present" in overrides or "hand_mask" in overrides:
            hand_mask = (overrides["hand_mask"] > 0).astype(np.uint8) * 255 if "hand_mask" in overrides else empty
            raw_hp = bool(overrides.get("hand_present", np.any(hand_mask > 0)))
        elif self.model_manager.has("hand"):
            hr = self.model_manager.get("hand").predict(
                source=frame,
                imgsz=self.wf_cfg.models.imgsz,
                conf=self.proc_cfg.hand_conf,
                device=self.model_manager.device,
                verbose=False,
            )[0]
            raw_hp, hand_mask, _ = packet_cut_core.evaluate_hand_detection(
                hr,
                self.model_manager.get_class_ids("hand"),
                (h, w),
                reference_mask=self._baseline_mask,
                topstrip_mask=overrides.get("topstrip_mask"),
                frame=frame,
            )
        else:
            raw_hp, hand_mask = False, empty

        rx1, ry1, rx2, ry2 = (40, 40, min(600, w - 20), min(400, int(h * 0.8333)))
        if "scissor_present" in overrides or "scissor_mask" in overrides:
            scissor_mask = (overrides["scissor_mask"] > 0).astype(np.uint8) * 255 if "scissor_mask" in overrides else empty
            raw_sp = bool(overrides.get("scissor_present", np.any(scissor_mask > 0)))
        elif self.model_manager.has("scissor"):
            frame_scissor = frame.copy()
            frame_scissor[:ry1, :] = 0
            frame_scissor[ry2:, :] = 0
            frame_scissor[:, :rx1] = 0
            frame_scissor[:, rx2:] = 0
            sr = self.model_manager.get("scissor").predict(
                source=frame_scissor,
                imgsz=self.wf_cfg.models.imgsz,
                conf=self.proc_cfg.scissor_conf,
                device=self.model_manager.device,
                verbose=False,
            )[0]
            raw_sp, scissor_mask, _ = packet_cut_core.evaluate_scissor_detection(
                sr,
                self.model_manager.get_class_ids("scissor"),
                (h, w),
                packet_mask=self._baseline_mask,
                roi=(rx1, ry1, rx2, ry2),
            )
        else:
            raw_sp, scissor_mask = False, empty

        self._hand_streak = (self._hand_streak + 1) if raw_hp else 0
        self._scissor_streak = (self._scissor_streak + 1) if raw_sp else 0
        hp = self._hand_streak >= self.proc_cfg.hand_confirm_frames
        sp = self._scissor_streak >= self.proc_cfg.scissor_confirm_frames
        return hp, hand_mask, sp, scissor_mask

    def _predict_topstrip_precut(
        self,
        frame: np.ndarray,
        overrides: Dict[str, Any],
    ) -> Tuple[bool, np.ndarray]:
        """Pre-cut orientation/seal check ONLY. Never called after CUT_CONFIRMED."""
        h, w = frame.shape[:2]
        if "topstrip_visible" in overrides:
            return bool(overrides["topstrip_visible"]), np.zeros((h, w), dtype=np.uint8)
        if not self.model_manager.has("topstrip"):
            return False, np.zeros((h, w), dtype=np.uint8)

        res = self.model_manager.get("topstrip").predict(
            source=frame,
            imgsz=self.wf_cfg.models.imgsz,
            conf=self.proc_cfg.topstrip_conf,
            device=self.model_manager.device,
            verbose=False,
        )[0]
        insts = extract_yolo_instances(
            res,
            (h, w),
            selected_ids=self.model_manager.get_class_ids("topstrip"),
            min_conf=self.proc_cfg.topstrip_conf,
            min_area_px=60,
        )
        return len(insts) > 0, union_masks([i.mask for i in insts], (h, w))

    def _predict_cut_model(
        self,
        frame: np.ndarray,
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        if "cut_eval" in overrides:
            return dict(overrides["cut_eval"])
        if not self.model_manager.has("cut"):
            return {"is_cut_detected": False, "confidence": 0.0, "class_name": "unavailable", "reason": "Cut model missing"}
        res = self.model_manager.get("cut").predict(
            source=frame,
            imgsz=self.wf_cfg.models.imgsz,
            conf=self.proc_cfg.cut_model_conf,
            device=self.model_manager.device,
            verbose=False,
        )[0]
        return evaluate_cut_model_prediction(
            res,
            model=self.model_manager.get("cut"),
            cut_class_id=self.proc_cfg.cut_model_class_id,
            min_conf=self.proc_cfg.cut_model_min_conf,
        )

    def _predict_jewels_raw(
        self,
        frame: np.ndarray,
        overrides: Dict[str, Any],
    ) -> List[RawSegInstance]:
        if "jewel_instances" in overrides:
            return list(overrides["jewel_instances"])
        if not self.model_manager.has("jewel"):
            return []
        res = self.model_manager.get("jewel").predict(
            source=frame,
            imgsz=self.wf_cfg.models.imgsz,
            conf=self.insp_cfg.jewellery_counting.jewel_conf,
            device=self.model_manager.device,
            verbose=False,
        )[0]
        return extract_yolo_instances(
            res,
            frame.shape[:2],
            selected_ids=self.model_manager.get_class_ids("jewel"),
            min_conf=self.insp_cfg.jewellery_counting.jewel_conf,
            min_area_px=60,
        )

    # ==================================================================
    # MAIN EVENT-DRIVEN FRAME PROCESSOR
    # ==================================================================
    def process_frame(
        self,
        frame: np.ndarray,
        now_sec: Optional[float] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> AutomatedStepResult:
        """Process one camera frame fully automatically without manual stage input."""
        alerts, logger, monitor = self._require_started()
        ov = overrides or {}
        ts = time.monotonic() if now_sec is None else float(now_sec)

        if self.stage == AutomatedStage.SESSION_COMPLETED:
            return self._build_step_result(frame.copy(), "Inspection session completed automatically.", completed=True)

        # ------------------------------------------------------------------
        # CONTINUOUS OCR PROTECTION (runs in EVERY stage until ID is locked!)
        # ------------------------------------------------------------------
        is_recovery = self.stage not in (AutomatedStage.WAITING_FOR_PACKET, AutomatedStage.PACKET_IDENTIFICATION)
        ocr_obs: OCRStepObservation = self.ocr_protector.observe(
            frame,
            current_stage=self.stage.value,
            is_recovery_mode=is_recovery,
            roi_override=ov.get("ocr_roi"),
            ocr_text_override=ov.get("ocr_text"),
        )
        if ocr_obs.newly_resolved:
            ev_img = logger.save_evidence_frame(frame, f"ocr_{ocr_obs.packet_id}")
            self.ocr_protector.summary.evidence_image = ev_img
            monitor.on_ocr_resolved(
                packet_id=ocr_obs.packet_id,
                confidence=self.ocr_protector.summary.confidence,
                stage=self.stage.value,
                was_late_recovery=ocr_obs.is_recovery,
                frame=None,
            )

        pid = self.ocr_protector.packet_id
        status_msg = ""
        vis = frame.copy()

        # ------------------------------------------------------------------
        # PRE-CUT & CUTTING AUTOMATED PIPELINE
        # ------------------------------------------------------------------
        if self.stage in (
            AutomatedStage.WAITING_FOR_PACKET,
            AutomatedStage.PACKET_IDENTIFICATION,
            AutomatedStage.ORIENTATION_AND_BASELINE,
            AutomatedStage.CUTTING_MONITOR,
            AutomatedStage.CUT_VERIFICATION,
        ):
            p_info = self._predict_packet_pieces(frame, ov)
            pieces = p_info["pieces"]
            pkt_present = p_info["total_area"] >= 1000 and len(pieces) >= 1
            # Only run models required for the active stage!
            need_hand_scissor = (
                self.stage in (AutomatedStage.CUTTING_MONITOR, AutomatedStage.CUT_VERIFICATION)
                or "hand_present" in ov or "hand_mask" in ov
                or "scissor_present" in ov or "scissor_mask" in ov
            )
            if need_hand_scissor:
                hp, hand_mask, sp, scissor_mask = self._predict_hand_and_scissor(frame, ov)
            else:
                h_dim, w_dim = frame.shape[:2]
                hp, hand_mask, sp, scissor_mask = False, np.zeros((h_dim, w_dim), dtype=np.uint8), False, np.zeros((h_dim, w_dim), dtype=np.uint8)

            need_topstrip = (
                self.stage == AutomatedStage.ORIENTATION_AND_BASELINE
                or "topstrip_visible" in ov
            )
            if need_topstrip:
                topstrip_vis, topstrip_mask = self._predict_topstrip_precut(frame, ov)
            else:
                topstrip_vis, topstrip_mask = False, np.zeros(frame.shape[:2], dtype=np.uint8)

            vis = draw_mask_overlay(vis, p_info["union_mask"], (255, 130, 30), alpha=0.18, draw_contour=True)
            if np.any(hand_mask > 0):
                vis = draw_mask_overlay(vis, hand_mask, (50, 50, 230), alpha=0.28)
            if np.any(scissor_mask > 0):
                vis = draw_mask_overlay(vis, scissor_mask, (0, 230, 230), alpha=0.35)
            if ocr_obs.roi_bbox is not None:
                rx1, ry1, rx2, ry2 = ocr_obs.roi_bbox
                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (40, 230, 120), 2)

            # 1) WAITING_FOR_PACKET -> PACKET_IDENTIFICATION
            if self.stage == AutomatedStage.WAITING_FOR_PACKET:
                if pkt_present or ocr_obs.roi_detected:
                    self.stage = AutomatedStage.PACKET_IDENTIFICATION
                    status_msg = "Packet detected automatically — reading OCR packet ID..."
                    logger.log(
                        packet_id=pid,
                        stage=self.stage.value,
                        procedure_step="PACKET_OCR_IDENTIFICATION",
                        expected_action="Present front identification side for OCR.",
                        actual_action="Packet detected under camera; automatic OCR started.",
                        compliance_status=ComplianceStatus.INFO,
                    )
                else:
                    status_msg = "Monitoring active — waiting for packet on work mat."

            # 2) PACKET_IDENTIFICATION (adaptive: handles normal OCR OR backside/premature cutting!)
            if self.stage == AutomatedStage.PACKET_IDENTIFICATION:
                if self.ocr_protector.is_resolved:
                    self.stage = AutomatedStage.ORIENTATION_AND_BASELINE
                    status_msg = f"OCR locked: {pid}. Checking orientation & intact baseline automatically..."
                elif sp or (hp and len(pieces) == 2):
                    # Appraiser started cutting BEFORE OCR completed!
                    # Record deviation & alert, capture fallback baseline if intact, and ADAPT without stopping!
                    if self._baseline_mask is None and len(pieces) == 1:
                        self._baseline_mask = pieces[0]["mask"]
                        self._baseline_area = float(pieces[0]["area"])
                    monitor.on_cutting_started(
                        packet_id=pid,
                        ocr_resolved=False,
                        orientation_verified=self._orientation_verified,
                        topstrip_visible=topstrip_vis,
                        stage=AutomatedStage.CUTTING_MONITOR.value,
                        frame=frame,
                        now_sec=ts,
                    )
                    self._cutting_started = True
                    self._clear_started_mono = None
                    self.stage = AutomatedStage.CUTTING_MONITOR
                    status_msg = "Cutting detected BEFORE OCR! Alert sent — continuing workflow & OCR recovery."
                elif pkt_present and not ocr_obs.roi_detected:
                    # Packet visible, no ID ROI visible (likely backside presented before OCR)
                    self._ocr_missing_packet_frames += 1
                    if topstrip_vis or self._ocr_missing_packet_frames >= self.wf_cfg.ocr_protection.max_unresolved_backside_frames_before_alert:
                        monitor.on_backside_without_ocr(
                            packet_id=pid,
                            stage=self.stage.value,
                            frame=frame,
                            now_sec=ts,
                        )
                        status_msg = "Packet visible without ID side — alert issued to present front ID side."
                    else:
                        status_msg = "Searching for packet identification region (OCR)..."
                else:
                    self._ocr_missing_packet_frames = 0
                    status_msg = "Packet ID region detected — locking reliable OCR number..."

            # 3) ORIENTATION_AND_BASELINE (Check strip model ONLY — prompts to turn packet!)
            elif self.stage == AutomatedStage.ORIENTATION_AND_BASELINE:
                # Check if frontside number ROI is still showing (packet not yet turned)
                front_number_visible = False
                if "number_obb" in ov:
                    front_number_visible = bool(ov["number_obb"])
                elif self.model_manager.has("number"):
                    n_res = self.model_manager.get("number").predict(
                        source=frame,
                        imgsz=self.wf_cfg.models.number_imgsz,
                        conf=self.wf_cfg.ocr_protection.number_conf,
                        device=self.model_manager.device,
                        verbose=False,
                    )[0]
                    front_number_visible = (
                        n_res.boxes is not None
                        and len(n_res.boxes) > 0
                        and float(n_res.boxes.conf[0]) >= self.wf_cfg.ocr_protection.number_conf
                    )

                if front_number_visible:
                    # Packet is still front-side up! Keep waiting for the turn!
                    self._orientation_closed_streak = 0
                    status_msg = "Please turn the packet to the BACK side."
                elif pkt_present and len(pieces) == 1:
                    # Packet has been turned to backside! Check strip model ONLY:
                    if topstrip_vis:
                        self._orientation_closed_streak = 0
                        vis = draw_mask_overlay(vis, topstrip_mask, (255, 0, 255), alpha=0.35)
                        status_msg = "TopStrip seal visible — cutting not ready."
                    else:
                        self._orientation_closed_streak += 1
                        if not self._orientation_verified and self._orientation_closed_streak >= self.proc_cfg.orientation_closed_confirm_frames:
                            self._orientation_verified = True
                            monitor.on_orientation_verified(pid, self.stage.value, frame=frame)

                        self._baseline_streak += 1
                        if self._baseline_mask is None and self._baseline_streak >= self.proc_cfg.baseline_stable_frames:
                            self._baseline_mask = pieces[0]["mask"]
                            self._baseline_area = float(pieces[0]["area"])
                            monitor.on_baseline_captured(pid, self._baseline_area, self.stage.value, frame=frame)

                        if self._orientation_verified and self._baseline_mask is not None:
                            self.stage = AutomatedStage.CUTTING_MONITOR
                            status_msg = "Backside confirmed & baseline captured — ready for cutting."
                        else:
                            status_msg = (
                                f"Backside detected — verifying closed orientation "
                                f"({self._orientation_closed_streak}/{self.proc_cfg.orientation_closed_confirm_frames})..."
                            )
                else:
                    status_msg = "Please turn the packet to the BACK side."

            # 4) CUTTING_MONITOR (auto-detects cutting & clear cooldown)
            elif self.stage == AutomatedStage.CUTTING_MONITOR:
                if hp or sp:
                    if not self._cutting_started:
                        self._cutting_started = True
                        monitor.on_cutting_started(
                            packet_id=pid,
                            ocr_resolved=self.ocr_protector.is_resolved,
                            orientation_verified=self._orientation_verified,
                            topstrip_visible=topstrip_vis,
                            stage=self.stage.value,
                            frame=frame,
                            now_sec=ts,
                        )
                    self._clear_started_mono = None
                    status_msg = "Cutting activity in progress — monitoring automatically."
                elif self._cutting_started:
                    if self._clear_started_mono is None:
                        self._clear_started_mono = ts
                    elapsed = ts - self._clear_started_mono
                    if elapsed >= self.proc_cfg.clear_wait_seconds:
                        monitor.on_scene_cleared_after_cut(pid, self.stage.value, frame=frame)
                        self.stage = AutomatedStage.CUT_VERIFICATION
                        self._verify_started_mono = ts
                        self._next_verify_sample_mono = ts
                        self._verify_samples = 0
                        self._verify_valid_samples = 0
                        self._verify_stable_streak = 0
                        self._verify_prev_sig = None
                        status_msg = "Scene clear — starting automatic physical cut verification."
                    else:
                        status_msg = f"Work area clear ({elapsed:.1f}/{self.proc_cfg.clear_wait_seconds:.1f}s)..."
                else:
                    # Handle case where packet was already cut into 2 pieces quickly between frames
                    if len(pieces) == 2:
                        self._cutting_started = True
                        self._clear_started_mono = ts
                        status_msg = "Two packet pieces detected — stabilizing for cut verification."
                    else:
                        status_msg = "Ready for cutting — observing automatically."

            # 5) CUT_VERIFICATION (auto physical separation check -> CUT_CONFIRMED -> Stage 1)
            elif self.stage == AutomatedStage.CUT_VERIFICATION:
                if hp or sp:
                    monitor.on_cut_verification_interrupted(pid, self.stage.value, frame=frame, now_sec=ts)
                    self.stage = AutomatedStage.CUTTING_MONITOR
                    self._clear_started_mono = None
                    status_msg = "Cut verification interrupted by hand/scissors — waiting for clear scene."
                else:
                    if self._verify_started_mono is None:
                        self._verify_started_mono = ts
                        self._next_verify_sample_mono = ts

                    if ts >= self._next_verify_sample_mono:
                        self._verify_samples += 1
                        self._last_pieces = pieces
                        cut_eval = self._predict_cut_model(frame, ov)
                        gap_info = None
                        is_valid = False
                        ref_area = self._baseline_area if self._baseline_area > 0 else max(float(p_info["total_area"]), 1.0)

                        if len(pieces) == 2:
                            a, b = pieces[:2]
                            gap_info = packet_cut_core.analyze_packet_gap(
                                a["mask"],
                                b["mask"],
                                frame=frame,
                                reference_area=ref_area,
                            )
                            self._last_gap = gap_info
                            area_ratio = (a["area"] + b["area"]) / max(ref_area, 1.0)
                            if gap_info["valid"] and (0.50 <= area_ratio <= 1.50):
                                is_valid = True
                                self._verify_valid_samples += 1
                                curr_sig = packet_cut_core.piece_signature(pieces)
                                if self._verify_prev_sig is None or packet_cut_core.is_stable_pair(self._verify_prev_sig, curr_sig):
                                    self._verify_stable_streak += 1
                                else:
                                    self._verify_stable_streak = 1
                                self._verify_prev_sig = curr_sig
                            else:
                                self._verify_stable_streak = 0
                                self._verify_prev_sig = None
                                if cut_eval.get("is_cut_detected"):
                                    monitor.on_cut_pieces_touching(pid, self.stage.value, now_sec=ts)
                        else:
                            self._verify_stable_streak = 0
                            self._verify_prev_sig = None
                            if cut_eval.get("is_cut_detected") and len(pieces) == 1:
                                monitor.on_cut_pieces_touching(pid, self.stage.value, now_sec=ts)

                        self._next_verify_sample_mono = ts + self.proc_cfg.verify_sample_interval
                        elapsed_v = ts - self._verify_started_mono
                        valid_ratio = float(self._verify_valid_samples) / float(max(self._verify_samples, 1))

                        # Auto-confirm as soon as verification window completes with valid criteria
                        if (
                            elapsed_v >= self.proc_cfg.cut_verify_seconds
                            and self._verify_samples >= self.proc_cfg.min_verify_samples
                            and valid_ratio >= self.proc_cfg.min_valid_sample_ratio
                            and self._verify_stable_streak >= self.proc_cfg.min_cut_streak
                            and cut_eval.get("is_cut_detected")
                            and self._last_gap is not None
                            and self._last_gap.get("valid")
                        ):
                            reason = (
                                f"CUT_CONFIRMED: model={cut_eval['confidence']:.1%}, "
                                f"pieces=2, gap={self._last_gap['min_gap']:.1f}px"
                            )
                            self._cut_result = {
                                "state": "CUT_CONFIRMED",
                                "pieces_count": 2,
                                "min_gap_px": round(float(self._last_gap["min_gap"]), 2),
                                "gap_coverage_pct": round(float(self._last_gap["coverage_pct"]), 1),
                                "cut_model_confidence": round(float(cut_eval["confidence"]), 4),
                                "valid_samples": self._verify_valid_samples,
                                "total_samples": self._verify_samples,
                                "reason": reason,
                            }
                            monitor.on_cut_confirmed(pid, reason, self.stage.value, frame=frame)
                            self.stage1_evaluator.reset()
                            self.stage = AutomatedStage.POST_CUT_PACKET_INSPECTION
                            status_msg = "CUT_CONFIRMED! Automatically advancing to Post-Cut Packet Inspection (OPEN/CLOSED)."
                        else:
                            status_msg = (
                                f"Verifying physical cut separation ({elapsed_v:.1f}/{self.proc_cfg.cut_verify_seconds:.1f}s | "
                                f"valid {self._verify_valid_samples}/{self._verify_samples} | streak {self._verify_stable_streak})"
                            )
                    else:
                        elapsed_v = ts - self._verify_started_mono
                        status_msg = f"Verifying physical cut separation ({elapsed_v:.1f}/{self.proc_cfg.cut_verify_seconds:.1f}s)..."

                    vis = packet_cut_core.draw_verification_overlay(frame, self._last_pieces, self._last_gap)

        # ------------------------------------------------------------------
        # STAGE 1: POST-CUT PACKET OPEN / CLOSED INSPECTION (NO TOPSTRIP!)
        # ------------------------------------------------------------------
        elif self.stage == AutomatedStage.POST_CUT_PACKET_INSPECTION:
            s1_ov = ov.get("stage1", {})
            s1_eval = self.stage1_evaluator.evaluate_frame(
                frame,
                packet_mask_override=s1_ov.get("packet_mask"),
                hand_mask_override=s1_ov.get("hand_mask"),
                jewel_instances_override=s1_ov.get("jewel_instances"),
            )
            self._post_cut_outcome = s1_eval.outcome
            vis = s1_eval.annotated_frame

            if s1_eval.is_confirmed and s1_eval.confirmed_status in ("OPEN", "CLOSED"):
                self._post_cut_unknown_streak = 0
                ev_path = logger.save_evidence_frame(vis, f"postcut_{s1_eval.confirmed_status.lower()}")
                self._post_cut_outcome.evidence_image = ev_path
                monitor.on_post_cut_packet_status_confirmed(
                    packet_id=pid,
                    status=s1_eval.confirmed_status,
                    gold_visible=s1_eval.outcome.gold_visible,
                    stage=self.stage.value,
                    frame=None,
                )
                self.stage2_manager.reset(require_empty_check=True)
                self.stage = AutomatedStage.TESTBED_EMPTY_CHECK
                status_msg = (
                    f"Packet condition confirmed: {s1_eval.confirmed_status}. "
                    "Automatically checking empty green testbed for Stage 2."
                )
            else:
                # Adaptive out-of-order detection: did appraiser skip Stage 1 and place jewellery directly on testbed?
                s2_ov = ov.get("stage2", {})
                tb_roi = self.stage2_manager.roi_manager.get_roi(frame)
                probe_det = self.stage2_manager.detector.detect(
                    frame,
                    tb_roi,
                    jewel_instances_override=s2_ov.get("jewel_instances"),
                    hand_mask_override=s2_ov.get("hand_mask", s1_ov.get("hand_mask")),
                    packet_mask_override=s2_ov.get("packet_mask", s1_ov.get("packet_mask")),
                )
                if len(probe_det.detections) > 0 and not probe_det.hand_present:
                    monitor.on_stage1_skipped_to_jewellery(pid, self.stage.value, frame=frame, now_sec=ts)
                    self.stage2_manager.reset(require_empty_check=False)
                    self.stage = AutomatedStage.JEWELLERY_COUNTING
                    status_msg = "Jewellery placed on testbed before Stage 1 completed — deviation logged, counting automatically."
                else:
                    if s1_eval.raw_status == "UNKNOWN" and s1_eval.roi_result.detected:
                        self._post_cut_unknown_streak += 1
                        if self._post_cut_unknown_streak >= self.proc_cfg.post_cut_unknown_alert_frames:
                            monitor.on_post_cut_view_unclear(
                                packet_id=pid,
                                reason=s1_eval.outcome.reason,
                                stage=self.stage.value,
                                frame=frame,
                                now_sec=ts,
                            )
                    else:
                        self._post_cut_unknown_streak = 0
                    status_msg = f"Stage 1 Packet Inspection: {s1_eval.outcome.reason}"

        # ------------------------------------------------------------------
        # STAGE 2: TESTBED EMPTY CHECK & JEWELLERY COUNTING
        # ------------------------------------------------------------------
        elif self.stage in (AutomatedStage.TESTBED_EMPTY_CHECK, AutomatedStage.JEWELLERY_COUNTING):
            s2_ov = ov.get("stage2", {})
            s2_eval = self.stage2_manager.evaluate_frame(
                frame,
                now_sec=ts,
                jewel_instances_override=s2_ov.get("jewel_instances"),
                hand_mask_override=s2_ov.get("hand_mask"),
                packet_mask_override=s2_ov.get("packet_mask"),
            )
            self._jewellery_outcome = s2_eval.outcome
            vis = s2_eval.annotated_frame

            # Check empty testbed completion
            if self.stage == AutomatedStage.TESTBED_EMPTY_CHECK:
                if self.stage2_manager.is_empty_verified:
                    monitor.on_testbed_empty_confirmed(pid, self.stage.value, frame=frame)
                    self.stage = AutomatedStage.JEWELLERY_COUNTING
                    status_msg = "Green testbed verified empty — waiting for jewellery placement."
                else:
                    if s2_eval.empty_result is not None and not s2_eval.empty_result.raw_is_empty:
                        alerts.trigger("TESTBED_NOT_EMPTY", pid, self.stage.value, now_sec=ts)
                    status_msg = s2_eval.outcome.reason
            else:
                # Monitor outside-testbed placement deviation
                if s2_eval.detection_result.excluded_outside_roi > 0:
                    self._outside_testbed_streak += 1
                    if self._outside_testbed_streak >= self.proc_cfg.outside_testbed_alert_frames:
                        monitor.on_jewellery_outside_testbed(
                            packet_id=pid,
                            outside_count=s2_eval.detection_result.excluded_outside_roi,
                            stage=self.stage.value,
                            frame=frame,
                            now_sec=ts,
                        )
                else:
                    self._outside_testbed_streak = 0

                # Monitor jewellery inside packet not removed yet
                if s2_eval.detection_result.excluded_inside_packet > 0 and s2_eval.outcome.live_count == 0:
                    alerts.trigger("JEWELLERY_IN_PACKET_NOT_REMOVED", pid, self.stage.value, now_sec=ts)

                # Monitor valid placement inside testbed
                if s2_eval.outcome.live_count > 0 and s2_eval.detection_result.excluded_outside_roi == 0:
                    if not self._jewel_placed_logged or alerts.active_alerts:
                        self._jewel_placed_logged = True
                        monitor.on_jewellery_placed_inside_testbed(
                            packet_id=pid,
                            count_detected=s2_eval.outcome.live_count,
                            stage=self.stage.value,
                            frame=frame,
                        )

                # Monitor COUNT_UNCERTAIN deviation
                if s2_eval.phase == "COUNT_UNCERTAIN":
                    self._uncertain_count_streak += 1
                    if self._uncertain_count_streak >= self.proc_cfg.uncertain_count_alert_frames:
                        monitor.on_jewellery_count_uncertain(
                            packet_id=pid,
                            reason=s2_eval.outcome.reason,
                            stage=self.stage.value,
                            frame=frame,
                            now_sec=ts,
                        )
                    status_msg = s2_eval.outcome.reason
                else:
                    self._uncertain_count_streak = 0
                    status_msg = s2_eval.outcome.reason

                # AUTOMATIC COMPLETION once placement stabilizes with valid non-uncertain count
                if s2_eval.phase == "COUNT_CONFIRMATION" and s2_eval.outcome.count is not None:
                    final_cnt = int(s2_eval.outcome.count)
                    confirmed_outcome = self.stage2_manager.confirm_count()
                    confirmed_outcome.count_status = "AUTO_CONFIRMED"
                    confirmed_outcome.reason = (
                        f"Final jewellery count ({final_cnt}) stabilized and auto-confirmed."
                    )
                    ev_path = logger.save_evidence_frame(vis, f"stage2_count_{final_cnt}")
                    confirmed_outcome.evidence_image = ev_path
                    self._jewellery_outcome = confirmed_outcome

                    monitor.on_jewellery_count_completed(pid, final_cnt, self.stage.value, frame=None)
                    self._finalize_session(now_sec=ts, processing_completed=True)
                    return self._build_step_result(
                        vis,
                        f"Session completed automatically — Packet {self.ocr_protector.packet_id}, Count: {final_cnt}.",
                        completed=True,
                    )

        # Render compact top automation HUD & active alert overlay on frame
        vis = self._overlay_workflow_hud(vis, status_msg, alerts.active_alerts)
        return self._build_step_result(vis, status_msg, completed=False)

    def stop_and_generate_report(self, now_sec: Optional[float] = None) -> Dict[str, Any]:
        """Gracefully end an in-progress session and generate the complete factual report."""
        if self._final_report is not None:
            return self._final_report
        ts = time.monotonic() if now_sec is None else float(now_sec)
        return self._finalize_session(now_sec=ts, processing_completed=False)

    def _finalize_session(self, now_sec: float, processing_completed: bool) -> Dict[str, Any]:
        alerts, logger, monitor = self._require_started()
        pid = self.ocr_protector.packet_id
        completed_at = utc_now_iso()
        duration_sec = max(0.0, float(now_sec) - float(self.started_mono))

        monitor.finalize(packet_id=pid, stage=self.stage.value)
        self.stage = AutomatedStage.SESSION_COMPLETED

        report = FinalReportGenerator.build_report(
            session_id=self.session_id,
            started_at=self.started_at,
            completed_at=completed_at,
            duration_seconds=duration_sec,
            ocr_summary=self.ocr_protector.summary,
            cut_result=self._cut_result,
            post_cut_packet_result=self._post_cut_outcome.to_dict(),
            jewellery_result=self._jewellery_outcome.to_dict(),
            procedure_monitor=monitor,
            alert_manager=alerts,
            activity_logger=logger,
            processing_completed=processing_completed,
            model_report=self.model_manager.availability_report(),
        )
        self._final_report = report

        out_dir = self.wf_cfg.output.resolved_output_dir() / self.session_id
        self._report_paths = FinalReportGenerator.save_reports(
            report,
            output_dir=out_dir,
            save_json=self.wf_cfg.output.save_json_report,
            save_markdown=self.wf_cfg.output.save_markdown_report,
        )
        return report

    def _build_step_result(
        self,
        annotated_frame: np.ndarray,
        status_message: str,
        completed: bool,
    ) -> AutomatedStepResult:
        alerts, _, monitor = self._require_started()
        return AutomatedStepResult(
            session_id=self.session_id,
            stage=self.stage,
            packet_id=self.ocr_protector.packet_id,
            identity_status=self.ocr_protector.summary.identity_status,
            status_message=status_message,
            annotated_frame=annotated_frame,
            active_alerts=[a.to_dict() for a in alerts.active_alerts],
            compliance_summary=monitor.compliance_metrics(),
            operational_snapshot={
                "ocr": self.ocr_protector.summary.to_dict(),
                "cut": dict(self._cut_result),
                "post_cut_packet": self._post_cut_outcome.to_dict(),
                "jewellery": self._jewellery_outcome.to_dict(),
            },
            completed=completed,
            final_report=self._final_report,
            report_paths=dict(self._report_paths),
        )

    def _overlay_workflow_hud(
        self,
        frame: np.ndarray,
        status_msg: str,
        active_alerts: List[AlertRecord],
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Bottom status strip so we don't overwrite Stage 1 / Stage 2 top HUDs
        strip_h = 46 if not active_alerts else 74
        y0 = max(0, h - strip_h)
        cv2.rectangle(vis, (0, y0), (w, h), (13, 21, 34), -1)

        pid_color = (60, 230, 140) if self.ocr_protector.is_resolved else (70, 190, 255)
        cv2.putText(
            vis,
            f"STAGE: {self.stage.value}  |  PACKET ID: {self.ocr_protector.packet_id} ({self.ocr_protector.summary.identity_status})",
            (10, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            pid_color,
            1,
        )
        cv2.putText(
            vis,
            status_msg[:95],
            (10, y0 + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (220, 230, 242),
            1,
        )
        if active_alerts:
            latest = active_alerts[-1]
            alert_color = (60, 100, 255) if latest.severity == "CRITICAL" else (50, 185, 255)
            cv2.putText(
                vis,
                f"ALERT [{latest.severity}]: {latest.message[:82]}",
                (10, y0 + 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                alert_color,
                2,
            )
        return vis

