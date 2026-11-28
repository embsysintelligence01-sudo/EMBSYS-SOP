"""Independent Appraiser Procedure Compliance Monitor (Sections 2B, 4, 5, 6, 7).

Operates strictly independently of packet processing:
- Maintains the predefined prescribed operational procedure (9 observable steps).
- Compares actual observed appraiser actions against expected actions continuously.
- Classifies COMPLIANT, SKIPPED, OUT_OF_ORDER, DEVIATION, RECOVERED, UNVERIFIED.
- Triggers real-time alerts with deduplication and tracks appraiser self-corrections.
- NEVER halts or terminates automated packet processing when a procedural mistake occurs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .activity_log import ActivityLogger, ComplianceStatus
from .alerts import AlertRecord, RealTimeAlertManager, utc_now_iso


@dataclass
class PrescribedStep:
    step_id: str
    order: int
    title: str
    expected_action: str
    status: str = "PENDING"  # PENDING | COMPLIANT | SKIPPED | OUT_OF_ORDER | DEVIATION | RECOVERED | UNVERIFIED
    actual_action: str = "Not observed"
    timestamp: Optional[str] = None
    notes: str = ""
    evidence_image: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeviationRecord:
    deviation_id: str
    code: str
    deviation_type: str  # SKIPPED | OUT_OF_ORDER | DEVIATION
    procedure_step: str
    expected_action: str
    actual_action: str
    why_non_compliant: str
    timestamp: str
    stage: str
    packet_id: str
    alert_id: Optional[str] = None
    alert_message: Optional[str] = None
    corrected: bool = False
    corrected_at: Optional[str] = None
    correction_action: str = ""
    processing_continued: bool = True
    evidence_image: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_prescribed_procedure() -> List[PrescribedStep]:
    """Ordered standard operating procedure (SOP) for EMBSYS Packet-Cut."""
    return [
        PrescribedStep(
            step_id="PACKET_OCR_IDENTIFICATION",
            order=1,
            title="Packet Front Presentation & OCR Identification",
            expected_action="Present the packet identification (front) side clearly under the camera so OCR reads the packet number before turning or cutting.",
        ),
        PrescribedStep(
            step_id="ORIENTATION_AND_SEAL_CHECK",
            order=2,
            title="Back/Seal Side Orientation Verification",
            expected_action="Turn packet to back/seal side and verify the top seal strip is closed before cutting.",
        ),
        PrescribedStep(
            step_id="INTACT_BASELINE_CAPTURE",
            order=3,
            title="Intact Packet Baseline Presentation",
            expected_action="Place the intact packet unobstructed on the mat so a clean single-piece baseline mask is captured.",
        ),
        PrescribedStep(
            step_id="CUTTING_EXECUTION",
            order=4,
            title="Controlled Cutting Execution & Clear Work Area",
            expected_action="Cut the packet cleanly with scissors and remove hands and scissors from the camera view after cutting.",
        ),
        PrescribedStep(
            step_id="CUT_SEPARATION_VERIFICATION",
            order=5,
            title="Physical Cut Separation Presentation",
            expected_action="Ensure the two severed packet pieces show a clear physical background gap without obstruction.",
        ),
        PrescribedStep(
            step_id="POST_CUT_PACKET_INSPECTION",
            order=6,
            title="Post-Cut Packet OPEN/CLOSED Inspection",
            expected_action="Hold the cut packet vertically and clearly under the overhead camera so interior gold visibility can be verified (OPEN/CLOSED).",
        ),
        PrescribedStep(
            step_id="TESTBED_EMPTY_VERIFICATION",
            order=7,
            title="Empty Green Testbed Verification",
            expected_action="Ensure the designated green rectangular testbed is empty before placing removed jewellery.",
        ),
        PrescribedStep(
            step_id="JEWELLERY_TESTBED_PLACEMENT",
            order=8,
            title="Jewellery Removal & Proper Testbed Placement",
            expected_action="Remove all jewellery from the packet and place individual articles clearly separated inside the designated green testbed.",
        ),
        PrescribedStep(
            step_id="JEWELLERY_COUNT_COMPLETION",
            order=9,
            title="Stable Jewellery Counting Completion",
            expected_action="Keep hands clear of the testbed and ensure articles do not overlap until the final count stabilizes.",
        ),
    ]


class AppraiserProcedureMonitor:
    """Monitors appraiser compliance against the prescribed SOP independently."""

    def __init__(
        self,
        session_id: str,
        alert_manager: RealTimeAlertManager,
        activity_logger: ActivityLogger,
    ) -> None:
        self.session_id = session_id
        self.alerts = alert_manager
        self.logger = activity_logger

        self._steps: Dict[str, PrescribedStep] = {
            s.step_id: s for s in build_prescribed_procedure()
        }
        self._step_order: List[str] = [s.step_id for s in sorted(self._steps.values(), key=lambda x: x.order)]
        self._deviations: List[DeviationRecord] = []
        self._deviations_by_code: Dict[str, DeviationRecord] = {}
        self._dev_counter: int = 0
        self._finalized: bool = False

    @property
    def steps(self) -> List[PrescribedStep]:
        return [self._steps[sid] for sid in self._step_order]

    @property
    def deviations(self) -> List[DeviationRecord]:
        return list(self._deviations)

    def get_step(self, step_id: str) -> PrescribedStep:
        return self._steps[step_id]

    def _record_deviation(
        self,
        code: str,
        deviation_type: str,
        step_id: str,
        actual_action: str,
        why_non_compliant: str,
        packet_id: str,
        stage: str,
        alert_code: Optional[str] = None,
        custom_alert_message: Optional[str] = None,
        frame: Optional[np.ndarray] = None,
        now_sec: Optional[float] = None,
        update_step_status: bool = True,
    ) -> Optional[DeviationRecord]:
        """Record a deviation (deduplicated per active code) and emit its alert."""
        step = self._steps[step_id]
        ev_img = self.logger.save_evidence_frame(frame, f"dev_{code.lower()}") if frame is not None else None

        alert_rec: Optional[AlertRecord] = None
        if alert_code:
            alert_rec = self.alerts.trigger(
                code=alert_code,
                packet_id=packet_id,
                stage=stage,
                custom_message=custom_alert_message,
                evidence_image=ev_img,
                now_sec=now_sec,
            )

        existing = self._deviations_by_code.get(code)
        if existing is not None and not existing.corrected:
            return existing

        self._dev_counter += 1
        dev = DeviationRecord(
            deviation_id=f"DEV-{self._dev_counter:04d}",
            code=code,
            deviation_type=deviation_type,
            procedure_step=step_id,
            expected_action=step.expected_action,
            actual_action=actual_action,
            why_non_compliant=why_non_compliant,
            timestamp=utc_now_iso(),
            stage=stage,
            packet_id=packet_id,
            alert_id=alert_rec.alert_id if alert_rec else None,
            alert_message=alert_rec.message if alert_rec else None,
            corrected=False,
            processing_continued=True,
            evidence_image=ev_img,
        )
        self._deviations.append(dev)
        self._deviations_by_code[code] = dev

        if update_step_status and step.status in ("PENDING", "COMPLIANT"):
            step.status = deviation_type
            step.actual_action = actual_action
            step.timestamp = dev.timestamp
            step.notes = why_non_compliant
            step.evidence_image = ev_img

        comp_status = {
            "SKIPPED": ComplianceStatus.SKIPPED,
            "OUT_OF_ORDER": ComplianceStatus.OUT_OF_ORDER,
        }.get(deviation_type, ComplianceStatus.DEVIATION)

        self.logger.log(
            packet_id=packet_id,
            stage=stage,
            procedure_step=step_id,
            expected_action=step.expected_action,
            actual_action=actual_action,
            compliance_status=comp_status,
            deviation_code=code,
            reason=why_non_compliant,
            alert_id=dev.alert_id,
            corrected=False,
            evidence_image=ev_img,
        )
        return dev

    def _mark_corrected(
        self,
        deviation_code: str,
        alert_code: Optional[str],
        correction_action: str,
        packet_id: str,
        stage: str,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        if alert_code:
            self.alerts.resolve(alert_code, correction_note=correction_action)

        dev = self._deviations_by_code.get(deviation_code)
        if dev is None or dev.corrected:
            return

        dev.corrected = True
        dev.corrected_at = utc_now_iso()
        dev.correction_action = correction_action
        step = self._steps[dev.procedure_step]
        if step.status in ("SKIPPED", "OUT_OF_ORDER", "DEVIATION"):
            step.status = "RECOVERED"
            step.actual_action = f"{step.actual_action}; later corrected: {correction_action}"
            step.timestamp = dev.corrected_at

        self.logger.log(
            packet_id=packet_id,
            stage=stage,
            procedure_step=dev.procedure_step,
            expected_action=step.expected_action,
            actual_action=correction_action,
            compliance_status=ComplianceStatus.RECOVERED,
            deviation_code=deviation_code,
            reason=f"Appraiser corrected prior deviation '{deviation_code}'.",
            alert_id=dev.alert_id,
            corrected=True,
            frame=frame,
            evidence_tag=f"recovered_{deviation_code.lower()}",
        )

    def mark_step_compliant(
        self,
        step_id: str,
        actual_action: str,
        packet_id: str,
        stage: str,
        frame: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        step = self._steps[step_id]
        if step.status == "COMPLIANT":
            return
        ev_img = self.logger.save_evidence_frame(frame, f"ok_{step_id.lower()}") if frame is not None else None
        if step.status == "PENDING":
            step.status = "COMPLIANT"
        elif step.status in ("SKIPPED", "OUT_OF_ORDER", "DEVIATION"):
            step.status = "RECOVERED"
        step.actual_action = actual_action
        step.timestamp = utc_now_iso()
        step.evidence_image = ev_img

        self.logger.log(
            packet_id=packet_id,
            stage=stage,
            procedure_step=step_id,
            expected_action=step.expected_action,
            actual_action=actual_action,
            compliance_status=ComplianceStatus.COMPLIANT if step.status == "COMPLIANT" else ComplianceStatus.RECOVERED,
            evidence_image=ev_img,
            metadata=metadata,
        )

    # ==================================================================
    # DOMAIN EVENT HOOKS (called automatically by AutomatedWorkflowEngine)
    # ==================================================================
    def on_ocr_resolved(
        self,
        packet_id: str,
        confidence: float,
        stage: str,
        was_late_recovery: bool,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        self.alerts.update_packet_id_link(packet_id)
        for dev in self._deviations:
            if dev.packet_id == "UNRESOLVED":
                dev.packet_id = f"{packet_id} (recovered; was UNRESOLVED)"

        if was_late_recovery:
            self._mark_corrected(
                deviation_code="OCR_SKIPPED_BEFORE_CUTTING",
                alert_code="OCR_BEFORE_CUTTING_REQUIRED",
                correction_action=f"Identification side presented later; OCR recovered packet ID '{packet_id}' ({confidence:.0%}).",
                packet_id=packet_id,
                stage=stage,
                frame=frame,
            )
            self._mark_corrected(
                deviation_code="BACKSIDE_PRESENTED_BEFORE_OCR",
                alert_code="OCR_INCOMPLETE",
                correction_action=f"OCR completed after alert; packet ID '{packet_id}' locked.",
                packet_id=packet_id,
                stage=stage,
                frame=None,
            )
            self.alerts.resolve("OCR_BEFORE_CUTTING_REQUIRED", f"OCR recovered '{packet_id}'.")
            self.alerts.resolve("OCR_INCOMPLETE", f"OCR recovered '{packet_id}'.")
            step = self._steps["PACKET_OCR_IDENTIFICATION"]
            step.status = "RECOVERED"
            step.actual_action = f"OCR recovered late at stage {stage}: {packet_id} ({confidence:.0%})."
            step.timestamp = utc_now_iso()
        else:
            self.alerts.resolve("OCR_INCOMPLETE", f"OCR resolved '{packet_id}'.")
            self.mark_step_compliant(
                step_id="PACKET_OCR_IDENTIFICATION",
                actual_action=f"Front identification side presented; OCR locked packet ID '{packet_id}' ({confidence:.0%}).",
                packet_id=packet_id,
                stage=stage,
                frame=frame,
                metadata={"ocr_confidence": round(confidence, 4)},
            )

    def on_backside_without_ocr(
        self,
        packet_id: str,
        stage: str,
        frame: Optional[np.ndarray] = None,
        now_sec: Optional[float] = None,
    ) -> None:
        self._record_deviation(
            code="BACKSIDE_PRESENTED_BEFORE_OCR",
            deviation_type="OUT_OF_ORDER",
            step_id="PACKET_OCR_IDENTIFICATION",
            actual_action="Appraiser presented the back/seal side before completing front-side OCR identification.",
            why_non_compliant="Prescribed procedure requires presenting the packet identification side first so the packet number is captured before turning or cutting.",
            packet_id=packet_id,
            stage=stage,
            alert_code="OCR_INCOMPLETE",
            frame=frame,
            now_sec=now_sec,
        )
        self.alerts.trigger(
            code="OCR_BEFORE_CUTTING_REQUIRED",
            packet_id=packet_id,
            stage=stage,
            now_sec=now_sec,
        )

    def on_orientation_verified(
        self,
        packet_id: str,
        stage: str,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        self.alerts.resolve("ORIENTATION_NOT_FOLLOWED", "Back/seal orientation verified.")
        self.alerts.resolve("SEAL_STILL_OPEN_BEFORE_CUT", "TopStrip seal absent / closed.")
        self.mark_step_compliant(
            step_id="ORIENTATION_AND_SEAL_CHECK",
            actual_action="Packet oriented to closed back/seal side with no exposed TopStrip seal.",
            packet_id=packet_id,
            stage=stage,
            frame=frame,
        )

    def on_baseline_captured(
        self,
        packet_id: str,
        area_px: float,
        stage: str,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        self.mark_step_compliant(
            step_id="INTACT_BASELINE_CAPTURE",
            actual_action=f"Single intact packet baseline captured automatically ({area_px:.0f} px).",
            packet_id=packet_id,
            stage=stage,
            frame=frame,
            metadata={"baseline_area_px": round(area_px, 1)},
        )

    def on_cutting_started(
        self,
        packet_id: str,
        ocr_resolved: bool,
        orientation_verified: bool,
        topstrip_visible: bool,
        stage: str,
        frame: Optional[np.ndarray] = None,
        now_sec: Optional[float] = None,
    ) -> None:
        if not ocr_resolved:
            self._record_deviation(
                code="OCR_SKIPPED_BEFORE_CUTTING",
                deviation_type="SKIPPED",
                step_id="PACKET_OCR_IDENTIFICATION",
                actual_action="Appraiser started cutting the packet before OCR identification succeeded.",
                why_non_compliant="Cutting before OCR risks permanent loss of packet identity.",
                packet_id=packet_id,
                stage=stage,
                alert_code="OCR_BEFORE_CUTTING_REQUIRED",
                frame=frame,
                now_sec=now_sec,
            )

        if not orientation_verified:
            self._record_deviation(
                code="ORIENTATION_SKIPPED_BEFORE_CUTTING",
                deviation_type="SKIPPED",
                step_id="ORIENTATION_AND_SEAL_CHECK",
                actual_action="Appraiser started cutting without completing back/seal orientation verification.",
                why_non_compliant="Packet orientation and seal closure must be verified prior to cutting.",
                packet_id=packet_id,
                stage=stage,
                alert_code="ORIENTATION_NOT_FOLLOWED",
                frame=frame,
                now_sec=now_sec,
            )

        if topstrip_visible:
            self._record_deviation(
                code="CUTTING_WHILE_SEAL_OPEN",
                deviation_type="DEVIATION",
                step_id="ORIENTATION_AND_SEAL_CHECK",
                actual_action="Cutting activity detected while TopStrip seal strip remained visible.",
                why_non_compliant="Packet should be closed/oriented properly without an exposed open TopStrip seal before cutting.",
                packet_id=packet_id,
                stage=stage,
                alert_code="SEAL_STILL_OPEN_BEFORE_CUT",
                frame=frame,
                now_sec=now_sec,
            )

        self.logger.log(
            packet_id=packet_id,
            stage=stage,
            procedure_step="CUTTING_EXECUTION",
            expected_action=self._steps["CUTTING_EXECUTION"].expected_action,
            actual_action="Cutting activity detected automatically (hands/scissors active near packet).",
            compliance_status=ComplianceStatus.INFO,
            frame=frame,
            evidence_tag="cutting_started",
        )

    def on_scene_cleared_after_cut(
        self,
        packet_id: str,
        stage: str,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        self._mark_corrected(
            deviation_code="CUT_VERIFICATION_INTERRUPTED",
            alert_code="CUT_VERIFICATION_INTERRUPTED",
            correction_action="Hands and scissors cleared from work area.",
            packet_id=packet_id,
            stage=stage,
        )
        self.mark_step_compliant(
            step_id="CUTTING_EXECUTION",
            actual_action="Cutting completed and work area cleared for automated separation verification.",
            packet_id=packet_id,
            stage=stage,
            frame=frame,
        )

    def on_cut_verification_interrupted(
        self,
        packet_id: str,
        stage: str,
        frame: Optional[np.ndarray] = None,
        now_sec: Optional[float] = None,
    ) -> None:
        self._record_deviation(
            code="CUT_VERIFICATION_INTERRUPTED",
            deviation_type="DEVIATION",
            step_id="CUT_SEPARATION_VERIFICATION",
            actual_action="Hands or scissors re-entered camera view during post-cut separation verification.",
            why_non_compliant="Work area must stay clear during physical separation verification.",
            packet_id=packet_id,
            stage=stage,
            alert_code="CUT_VERIFICATION_INTERRUPTED",
            frame=frame,
            now_sec=now_sec,
            update_step_status=False,
        )

    def on_cut_pieces_touching(
        self,
        packet_id: str,
        stage: str,
        now_sec: Optional[float] = None,
    ) -> None:
        self.alerts.trigger(
            code="PULL_CUT_PIECES_APART",
            packet_id=packet_id,
            stage=stage,
            now_sec=now_sec,
        )

    def on_cut_confirmed(
        self,
        packet_id: str,
        reason: str,
        stage: str,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        self.alerts.resolve("PULL_CUT_PIECES_APART", "Physical separation confirmed.")
        self._mark_corrected(
            deviation_code="CUT_VERIFICATION_INTERRUPTED",
            alert_code="CUT_VERIFICATION_INTERRUPTED",
            correction_action="Cut separation verification completed after clearing scene.",
            packet_id=packet_id,
            stage=stage,
        )
        self.mark_step_compliant(
            step_id="CUT_SEPARATION_VERIFICATION",
            actual_action=f"CUT_CONFIRMED automatically: {reason}",
            packet_id=packet_id,
            stage=stage,
            frame=frame,
        )

    def on_post_cut_view_unclear(
        self,
        packet_id: str,
        reason: str,
        stage: str,
        frame: Optional[np.ndarray] = None,
        now_sec: Optional[float] = None,
    ) -> None:
        self._record_deviation(
            code="POST_CUT_VIEW_UNCLEAR",
            deviation_type="DEVIATION",
            step_id="POST_CUT_PACKET_INSPECTION",
            actual_action=f"Packet held unclearly/occluded/rotated under inspection camera: {reason}",
            why_non_compliant="Appraiser must hold the cut packet vertically and unobstructed under the overhead camera for gold visibility inspection.",
            packet_id=packet_id,
            stage=stage,
            alert_code="PACKET_VIEW_UNCLEAR",
            frame=frame,
            now_sec=now_sec,
            update_step_status=False,
        )

    def on_post_cut_packet_status_confirmed(
        self,
        packet_id: str,
        status: str,
        gold_visible: Optional[bool],
        stage: str,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        self._mark_corrected(
            deviation_code="POST_CUT_VIEW_UNCLEAR",
            alert_code="PACKET_VIEW_UNCLEAR",
            correction_action=f"Packet presented clearly; Stage 1 status confirmed as {status}.",
            packet_id=packet_id,
            stage=stage,
        )
        self.mark_step_compliant(
            step_id="POST_CUT_PACKET_INSPECTION",
            actual_action=f"Post-cut packet condition verified automatically: {status} (gold_visible={gold_visible}).",
            packet_id=packet_id,
            stage=stage,
            frame=frame,
        )

    def on_stage1_skipped_to_jewellery(
        self,
        packet_id: str,
        stage: str,
        frame: Optional[np.ndarray] = None,
        now_sec: Optional[float] = None,
    ) -> None:
        self._record_deviation(
            code="POST_CUT_INSPECTION_SKIPPED",
            deviation_type="SKIPPED",
            step_id="POST_CUT_PACKET_INSPECTION",
            actual_action="Jewellery was placed directly on the green testbed before completing vertical post-cut packet inspection.",
            why_non_compliant="Prescribed procedure requires inspecting the cut packet interior (OPEN/CLOSED) before removing jewellery onto the testbed.",
            packet_id=packet_id,
            stage=stage,
            alert_code="STAGE1_SKIPPED_TO_JEWELLERY",
            frame=frame,
            now_sec=now_sec,
        )

    def on_testbed_empty_confirmed(
        self,
        packet_id: str,
        stage: str,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        self.alerts.resolve("TESTBED_NOT_EMPTY", "Green testbed cleared and verified empty.")
        self.mark_step_compliant(
            step_id="TESTBED_EMPTY_VERIFICATION",
            actual_action="Designated green testbed verified empty prior to jewellery placement.",
            packet_id=packet_id,
            stage=stage,
            frame=frame,
        )

    def on_jewellery_outside_testbed(
        self,
        packet_id: str,
        outside_count: int,
        stage: str,
        frame: Optional[np.ndarray] = None,
        now_sec: Optional[float] = None,
    ) -> None:
        self._record_deviation(
            code="JEWELLERY_OUTSIDE_TESTBED",
            deviation_type="DEVIATION",
            step_id="JEWELLERY_TESTBED_PLACEMENT",
            actual_action=f"Detected {outside_count} jewellery item(s) placed outside the designated green testbed ROI.",
            why_non_compliant="All jewellery articles must be placed inside the designated green rectangular testbed for valid counting.",
            packet_id=packet_id,
            stage=stage,
            alert_code="JEWELLERY_OUTSIDE_TESTBED",
            frame=frame,
            now_sec=now_sec,
            update_step_status=False,
        )

    def on_jewellery_placed_inside_testbed(
        self,
        packet_id: str,
        count_detected: int,
        stage: str,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        self._mark_corrected(
            deviation_code="JEWELLERY_OUTSIDE_TESTBED",
            alert_code="JEWELLERY_OUTSIDE_TESTBED",
            correction_action=f"Jewellery moved inside designated green testbed ({count_detected} article(s) visible).",
            packet_id=packet_id,
            stage=stage,
            frame=frame,
        )
        self.mark_step_compliant(
            step_id="JEWELLERY_TESTBED_PLACEMENT",
            actual_action=f"Jewellery placed inside designated green testbed ({count_detected} article(s) detected).",
            packet_id=packet_id,
            stage=stage,
            frame=frame,
        )

    def on_jewellery_count_uncertain(
        self,
        packet_id: str,
        reason: str,
        stage: str,
        frame: Optional[np.ndarray] = None,
        now_sec: Optional[float] = None,
    ) -> None:
        self._record_deviation(
            code="JEWELLERY_SEPARATION_UNCERTAIN",
            deviation_type="DEVIATION",
            step_id="JEWELLERY_COUNT_COMPLETION",
            actual_action=f"Jewellery articles overlapping or unclear on testbed: {reason}",
            why_non_compliant="Individual jewellery articles must be separated and clearly visible on the testbed for accurate counting.",
            packet_id=packet_id,
            stage=stage,
            alert_code="JEWELLERY_COUNT_UNCERTAIN",
            frame=frame,
            now_sec=now_sec,
            update_step_status=False,
        )

    def on_jewellery_count_completed(
        self,
        packet_id: str,
        final_count: int,
        stage: str,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        self._mark_corrected(
            deviation_code="JEWELLERY_SEPARATION_UNCERTAIN",
            alert_code="JEWELLERY_COUNT_UNCERTAIN",
            correction_action=f"Jewellery separated clearly; stable final count ({final_count}) achieved.",
            packet_id=packet_id,
            stage=stage,
        )
        self.mark_step_compliant(
            step_id="JEWELLERY_COUNT_COMPLETION",
            actual_action=f"Placement stabilized hand-free with final count = {final_count} article(s).",
            packet_id=packet_id,
            stage=stage,
            frame=frame,
        )

    def finalize(self, packet_id: str, stage: str) -> Dict[str, Any]:
        """Mark any never-observed steps as UNVERIFIED (never fabricate compliance)."""
        if not self._finalized:
            self._finalized = True
            for step in self.steps:
                if step.status == "PENDING":
                    step.status = "UNVERIFIED"
                    step.actual_action = "Action was not observed before session ended."
                    self.logger.log(
                        packet_id=packet_id,
                        stage=stage,
                        procedure_step=step.step_id,
                        expected_action=step.expected_action,
                        actual_action=step.actual_action,
                        compliance_status=ComplianceStatus.UNVERIFIED,
                        reason="Prescribed procedure step was not verified during session.",
                    )
        return self.compliance_metrics()

    def compliance_metrics(self) -> Dict[str, Any]:
        total = len(self._steps)
        compliant = sum(1 for s in self._steps.values() if s.status == "COMPLIANT")
        recovered = sum(1 for s in self._steps.values() if s.status == "RECOVERED")
        skipped = sum(1 for d in self._deviations if d.deviation_type == "SKIPPED")
        out_of_order = sum(1 for d in self._deviations if d.deviation_type == "OUT_OF_ORDER")
        unverified = sum(1 for s in self._steps.values() if s.status == "UNVERIFIED")
        corrected = sum(1 for d in self._deviations if d.corrected)
        confirmed_deviations = len(self._deviations)
        rate = float(compliant) / float(max(total, 1))

        return {
            "total_expected_procedures": total,
            "correctly_followed_procedures": compliant,
            "recovered_procedures": recovered,
            "confirmed_deviations": confirmed_deviations,
            "skipped_procedures": skipped,
            "out_of_order_actions": out_of_order,
            "unverified_actions": unverified,
            "corrected_mistakes": corrected,
            "uncorrected_deviations": confirmed_deviations - corrected,
            "strict_compliance_rate": round(rate, 4),
            "procedure_steps": [s.to_dict() for s in self.steps],
        }

