"""Top-level Post-Cutting Inspection Engine (GUI-independent orchestrator).

Connects:
- InspectionStateMachine (all states after CUT_CONFIRMED)
- Stage 1: PacketStatusEvaluator (OPEN / CLOSED / UNKNOWN via gold visibility)
- Stage 2: JewelleryCountManager (testbed ROI, separation, unique ID tracking, stability)
- Structured JSON & evidence-image persistence

Can be driven frame-by-frame by `InspectionWorker` (PyQt6 QThread), CLI loops,
or automated tests without duplicating cameras or models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .inspection_result import PostCutInspectionResult
from .inspection_state_machine import InspectionState, InspectionStateMachine
from .jewellery_counting.count_manager import JewelleryCountManager, Stage2FrameEvaluation
from .packet_inspection.packet_status import PacketStatusEvaluator, Stage1FrameEvaluation
from .utils.image_utils import RawSegInstance
from .utils.inspection_config import InspectionConfig
from .utils.model_loader import InspectionModelRegistry, ModelAvailabilityReport


@dataclass
class InspectionStepOutput:
    state: InspectionState
    annotated_frame: np.ndarray
    result: PostCutInspectionResult
    stage1_eval: Optional[Stage1FrameEvaluation] = None
    stage2_eval: Optional[Stage2FrameEvaluation] = None
    message: str = ""


class PostCutInspectionEngine:
    """Runs independent Stage 1 and Stage 2 inspection after CUT_CONFIRMED."""

    def __init__(
        self,
        config: Optional[InspectionConfig] = None,
        shared_models: Optional[Dict[str, Any]] = None,
        auto_load_models: bool = True,
    ) -> None:
        self.config = config if config is not None else InspectionConfig.load()
        self.registry = InspectionModelRegistry(self.config, shared_models=shared_models)
        self.model_report: ModelAvailabilityReport = (
            self.registry.ensure_loaded() if auto_load_models else self.registry.report
        )

        self.state_machine = InspectionStateMachine()
        self.stage1 = PacketStatusEvaluator(self.config, self.registry)
        self.stage2 = JewelleryCountManager(self.config, self.registry)
        self.result: Optional[PostCutInspectionResult] = None

    def on_cut_confirmed(self, packet_id: str, cut_status: str = "CUTTED") -> PostCutInspectionResult:
        """Activate post-cutting inspection upon receiving upstream CUT_CONFIRMED."""
        self.stage1.reset()
        self.stage2.reset(require_empty_check=True)
        self.state_machine.reset()
        self.state_machine.activate_on_cut_confirmed(packet_id)

        self.result = PostCutInspectionResult(
            packet_id=str(packet_id),
            cut_status=str(cut_status),
            inspection_status="IN_PROGRESS",
            state_history=self.state_machine.history,
            model_report=self.model_report.to_dict(),
        )
        return self.result

    def _ensure_active(self) -> PostCutInspectionResult:
        if self.result is None:
            raise RuntimeError("PostCutInspectionEngine has not been activated via on_cut_confirmed().")
        return self.result

    def _save_evidence(self, frame: np.ndarray, tag: str) -> Optional[str]:
        if not self.config.output.save_evidence_images or self.result is None:
            return None
        out_dir = self.config.output.resolved_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in self.result.packet_id)
        path = out_dir / f"{safe_id}_{tag}.jpg"
        cv2.imwrite(str(path), frame)
        return str(path)

    def proceed_to_jewellery_counting(self, require_empty_check: bool = True) -> InspectionState:
        """Explicitly transition from Stage 1 (PACKET_OPEN / PACKET_CLOSED) to Stage 2."""
        res = self._ensure_active()
        st = self.state_machine.state
        if st not in (InspectionState.PACKET_OPEN, InspectionState.PACKET_CLOSED):
            raise RuntimeError(
                f"Cannot proceed to Stage 2 from state {st.value}; "
                "complete Stage 1 packet status verification (OPEN or CLOSED) first."
            )
        self.stage2.reset(require_empty_check=require_empty_check)
        self.state_machine.transition(
            InspectionState.JEWELLERY_REMOVAL,
            f"Proceeding to jewellery removal after packet status {st.value}",
        )
        self.state_machine.transition(
            InspectionState.TESTBED_EMPTY_CHECK,
            "Verifying green testbed is empty before jewellery placement",
        )
        if not require_empty_check:
            self.stage2.skip_empty_check()
            self.state_machine.transition(
                InspectionState.WAIT_FOR_JEWELLERY,
                "Testbed empty check bypassed by caller",
            )
        res.state_history = self.state_machine.history
        return self.state_machine.state

    def retry_packet_inspection(self) -> InspectionState:
        """Re-run Stage 1 packet open/closed check while staying independent of cut status."""
        res = self._ensure_active()
        self.stage1.reset()
        if self.state_machine.can_transition(InspectionState.PACKET_VIEW_CHECK):
            self.state_machine.transition(InspectionState.PACKET_VIEW_CHECK, "Appraiser requested Stage 1 retry")
        res.state_history = self.state_machine.history
        return self.state_machine.state

    def retry_jewellery_counting(self, require_empty_check: bool = False) -> InspectionState:
        """Reset Stage 2 tracking and recount jewellery on the testbed."""
        res = self._ensure_active()
        self.stage2.reset(require_empty_check=require_empty_check)
        target = InspectionState.TESTBED_EMPTY_CHECK if require_empty_check else InspectionState.WAIT_FOR_JEWELLERY
        if self.state_machine.can_transition(target):
            self.state_machine.transition(target, "Appraiser requested Stage 2 recount")
        res.jewellery_counting = self.stage2._last_outcome
        res.inspection_status = "IN_PROGRESS"
        res.state_history = self.state_machine.history
        return self.state_machine.state

    def confirm_final_count(self, override_count: Optional[int] = None) -> PostCutInspectionResult:
        """Appraiser confirms final jewellery count and attempts completion.

        Enforces rule: an uncertain inspection is NEVER marked COMPLETED.
        """
        res = self._ensure_active()
        updated_jwl = self.stage2.confirm_count(override_count=override_count)
        res.jewellery_counting = updated_jwl

        if res.mark_completed():
            if self.state_machine.can_transition(InspectionState.INSPECTION_COMPLETED):
                self.state_machine.transition(
                    InspectionState.INSPECTION_COMPLETED,
                    f"Completed with packet={res.packet_inspection.status}, count={res.jewellery_counting.count}",
                )
            res.state_history = self.state_machine.history
            if self.config.output.save_json_report:
                out_dir = self.config.output.resolved_output_dir()
                safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in res.packet_id)
                res.save_json(out_dir / f"{safe_id}_inspection_result.json")
        else:
            res.state_history = self.state_machine.history
        return res

    def process_frame(
        self,
        frame: np.ndarray,
        now_sec: Optional[float] = None,
        auto_advance_to_stage2: bool = False,
        stage1_overrides: Optional[Dict[str, Any]] = None,
        stage2_overrides: Optional[Dict[str, Any]] = None,
    ) -> InspectionStepOutput:
        """Process the next camera frame according to the current inspection state."""
        res = self._ensure_active()
        st = self.state_machine.state

        if st == InspectionState.INSPECTION_COMPLETED:
            return InspectionStepOutput(
                state=st,
                annotated_frame=frame.copy(),
                result=res,
                message="Inspection completed.",
            )

        # ==============================================================
        # STAGE 1: PACKET OPEN / CLOSED DETECTION
        # ==============================================================
        if self.state_machine.is_stage1():
            if st in (InspectionState.PACKET_OPEN, InspectionState.PACKET_CLOSED):
                if auto_advance_to_stage2:
                    self.proceed_to_jewellery_counting(require_empty_check=True)
                else:
                    return InspectionStepOutput(
                        state=st,
                        annotated_frame=frame.copy(),
                        result=res,
                        message=(
                            f"Packet status confirmed: {res.packet_inspection.status}. "
                            "Proceed to jewellery removal & testbed counting."
                        ),
                    )
            else:
                ov = stage1_overrides or {}
                s1 = self.stage1.evaluate_frame(
                    frame,
                    packet_mask_override=ov.get("packet_mask"),
                    hand_mask_override=ov.get("hand_mask"),
                    jewel_instances_override=ov.get("jewel_instances"),
                )
                res.packet_inspection = s1.outcome

                # Drive exact Stage 1 state transitions
                if not s1.roi_result.detected or not s1.roi_result.view_valid or (
                    s1.interior_result is not None and not s1.interior_result.interior_valid
                ):
                    self.state_machine.transition(
                        InspectionState.WAIT_FOR_VALID_VIEW,
                        s1.outcome.reason,
                    )
                else:
                    if self.state_machine.state != InspectionState.GOLD_VISIBILITY_CHECK:
                        self.state_machine.transition(
                            InspectionState.GOLD_VISIBILITY_CHECK,
                            "Valid packet interior extracted; evaluating gold visibility",
                        )
                    if s1.is_confirmed and s1.confirmed_status == "OPEN":
                        s1.outcome.evidence_image = self._save_evidence(s1.annotated_frame, "stage1_open")
                        self.state_machine.transition(
                            InspectionState.PACKET_OPEN,
                            "Gold jewellery confirmed inside packet interior",
                        )
                        if auto_advance_to_stage2:
                            self.proceed_to_jewellery_counting(require_empty_check=True)
                    elif s1.is_confirmed and s1.confirmed_status == "CLOSED":
                        s1.outcome.evidence_image = self._save_evidence(s1.annotated_frame, "stage1_closed")
                        self.state_machine.transition(
                            InspectionState.PACKET_CLOSED,
                            "No gold jewellery confirmed inside valid packet interior",
                        )
                        if auto_advance_to_stage2:
                            self.proceed_to_jewellery_counting(require_empty_check=True)
                    elif s1.raw_status == "UNKNOWN":
                        self.state_machine.transition(
                            InspectionState.WAIT_FOR_VALID_VIEW,
                            s1.outcome.reason,
                        )

                res.state_history = self.state_machine.history
                return InspectionStepOutput(
                    state=self.state_machine.state,
                    annotated_frame=s1.annotated_frame,
                    result=res,
                    stage1_eval=s1,
                    message=s1.outcome.reason,
                )

        # ==============================================================
        # STAGE 2: JEWELLERY DETECTION AND COUNTING ON TESTBED
        # ==============================================================
        ov2 = stage2_overrides or {}
        s2 = self.stage2.evaluate_frame(
            frame,
            now_sec=now_sec,
            jewel_instances_override=ov2.get("jewel_instances"),
            hand_mask_override=ov2.get("hand_mask"),
            packet_mask_override=ov2.get("packet_mask"),
        )
        res.jewellery_counting = s2.outcome

        # Drive exact Stage 2 state transitions
        if s2.phase == "TESTBED_EMPTY_CHECK":
            if self.state_machine.can_transition(InspectionState.TESTBED_EMPTY_CHECK):
                self.state_machine.transition(InspectionState.TESTBED_EMPTY_CHECK, s2.outcome.reason)
            res.inspection_status = "IN_PROGRESS"

        elif s2.phase == "WAIT_FOR_JEWELLERY":
            if self.state_machine.can_transition(InspectionState.WAIT_FOR_JEWELLERY):
                self.state_machine.transition(InspectionState.WAIT_FOR_JEWELLERY, s2.outcome.reason)
            res.inspection_status = "IN_PROGRESS"

        elif s2.phase == "COUNT_UNCERTAIN":
            if self.state_machine.state == InspectionState.WAIT_FOR_JEWELLERY:
                self.state_machine.transition(InspectionState.JEWEL_DETECTION, "Jewellery detected")
                self.state_machine.transition(InspectionState.INSTANCE_SEPARATION, "Running separation")
            elif self.state_machine.state == InspectionState.JEWEL_DETECTION:
                self.state_machine.transition(InspectionState.INSTANCE_SEPARATION, "Running separation")
            if self.state_machine.can_transition(InspectionState.COUNT_UNCERTAIN):
                self.state_machine.transition(InspectionState.COUNT_UNCERTAIN, s2.outcome.reason)
            res.inspection_status = "UNCERTAIN"

        elif s2.phase == "TRACKING":
            if self.state_machine.state in (
                InspectionState.WAIT_FOR_JEWELLERY,
                InspectionState.COUNT_UNCERTAIN,
                InspectionState.COUNT_CONFIRMATION,
            ):
                self.state_machine.transition(InspectionState.JEWEL_DETECTION, "Detecting jewellery on testbed")
            if self.state_machine.state == InspectionState.JEWEL_DETECTION:
                self.state_machine.transition(InspectionState.INSTANCE_SEPARATION, "Separating articles")
            if self.state_machine.state == InspectionState.INSTANCE_SEPARATION:
                self.state_machine.transition(InspectionState.OBJECT_ID_ASSIGNMENT, "Assigning unique IDs")
            if self.state_machine.state == InspectionState.OBJECT_ID_ASSIGNMENT:
                self.state_machine.transition(InspectionState.PLACEMENT_STABILITY, "Checking placement stability")
            res.inspection_status = "IN_PROGRESS"

        elif s2.phase == "COUNT_CONFIRMATION":
            if self.state_machine.state in (
                InspectionState.WAIT_FOR_JEWELLERY,
                InspectionState.COUNT_UNCERTAIN,
            ):
                self.state_machine.transition(InspectionState.JEWEL_DETECTION, "Detecting jewellery")
            if self.state_machine.state == InspectionState.JEWEL_DETECTION:
                self.state_machine.transition(InspectionState.INSTANCE_SEPARATION, "Separating articles")
            if self.state_machine.state == InspectionState.INSTANCE_SEPARATION:
                self.state_machine.transition(InspectionState.OBJECT_ID_ASSIGNMENT, "Assigning unique IDs")
            if self.state_machine.state == InspectionState.OBJECT_ID_ASSIGNMENT:
                self.state_machine.transition(InspectionState.PLACEMENT_STABILITY, "Placement stable")
            if self.state_machine.state == InspectionState.PLACEMENT_STABILITY:
                s2.outcome.evidence_image = self._save_evidence(s2.annotated_frame, "stage2_count")
                self.state_machine.transition(
                    InspectionState.COUNT_CONFIRMATION,
                    f"Stable count {s2.outcome.live_count} ready for appraiser confirmation",
                )
            res.inspection_status = "AWAITING_CONFIRMATION"

        res.state_history = self.state_machine.history
        return InspectionStepOutput(
            state=self.state_machine.state,
            annotated_frame=s2.annotated_frame,
            result=res,
            stage2_eval=s2,
            message=s2.outcome.reason,
        )

