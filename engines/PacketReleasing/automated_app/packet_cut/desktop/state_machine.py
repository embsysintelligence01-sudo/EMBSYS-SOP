"""Event-driven state machine for the EMBSYS Packet-Cut PyQt6 application.

Tracks continuous release observation or strict sequential inspection, and records events
with expected vs. actual evidence. No policy fabricates a pass from a timeout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from workflow_monitor.alerts import utc_now_iso


class AutoStage(str, Enum):
    IDLE = "IDLE"
    RELEASE_OBSERVATION = "RELEASE · CONTINUOUS OBSERVATION"
    WAIT_EMPTY_TESTBED = "01 · WAIT FOR EMPTY TESTBED"
    WAIT_FRONTSIDE_AND_OCR = "02 · WAIT FOR FRONTSIDE & OCR"
    WAIT_TURN_AND_BACKSIDE_STRIP = "03 · WAIT FOR TURN & BACKSIDE STRIP"
    WAIT_CUTTING_ACTIVITY = "04 · CUTTING ACTIVITY"
    VERIFY_PHYSICAL_CUT = "05 · VERIFY THE CUT"
    WAIT_UPRIGHT_MOUTH = "06 · WAIT FOR UPRIGHT MOUTH"
    WAIT_GOLD_PLACEMENT = "07A · WAIT FOR GOLD PLACEMENT"
    WAIT_JEWELLERY_AND_COUNT = "07B · COUNT JEWELLERY"
    STOPPED_OR_SAVED = "08 · SESSION SAVED"


@dataclass(frozen=True)
class StageSpec:
    stage: AutoStage
    required_models: Tuple[str, ...]
    expected_physical_action: str
    required_detection_event: str
    display_instruction: str
    stable_frames_required: int
    timeout_seconds: float


STATE_MACHINE_SPECS: Dict[AutoStage, StageSpec] = {
    AutoStage.RELEASE_OBSERVATION: StageSpec(
        stage=AutoStage.RELEASE_OBSERVATION, required_models=(),
        expected_physical_action="Continue packet release; independent checks record available evidence.",
        required_detection_event="No stage gate; each observation records its own verification status",
        display_instruction="Observing release — continue your work. STOP saves the report.",
        stable_frames_required=1, timeout_seconds=0.0,
    ),
    AutoStage.IDLE: StageSpec(
        stage=AutoStage.IDLE,
        required_models=(),
        expected_physical_action="Press START to begin a new inspection session.",
        required_detection_event="START pressed",
        display_instruction="Press START to begin.",
        stable_frames_required=1,
        timeout_seconds=0.0,
    ),
    AutoStage.WAIT_EMPTY_TESTBED: StageSpec(
        stage=AutoStage.WAIT_EMPTY_TESTBED,
        required_models=("packet-seg.onnx",),
        expected_physical_action="Clear the testbed of packets, hands, and jewellery.",
        required_detection_event="Visible, clear, and stable empty testbed across fresh frames",
        display_instruction="Clear the testbed.",
        stable_frames_required=5,
        timeout_seconds=15.0,
    ),
    AutoStage.WAIT_FRONTSIDE_AND_OCR: StageSpec(
        stage=AutoStage.WAIT_FRONTSIDE_AND_OCR,
        required_models=("packet-seg.onnx", "packet-no-obb.onnx", "PaddleOCR"),
        expected_physical_action="Place the packet FRONT side up inside the testbed and hold steady for OCR.",
        required_detection_event="Packet inside testbed + number OBB belonging to packet + consistent PaddleOCR reads across independent fresh crops",
        display_instruction="Please place the packet FRONT side up.",
        stable_frames_required=3,
        timeout_seconds=30.0,
    ),
    AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP: StageSpec(
        stage=AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP,
        required_models=("packet-seg.onnx", "strip-seg.onnx"),
        expected_physical_action="Turn the same packet to the BACK side, clear hands, and present a clear stationary backside view.",
        required_detection_event="Tracked packet turned + hands clear + stationary verified backside + TopStrip absent across clear stable frames",
        display_instruction="Please turn the packet to the BACK side.",
        stable_frames_required=4,
        timeout_seconds=30.0,
    ),
    AutoStage.WAIT_CUTTING_ACTIVITY: StageSpec(
        stage=AutoStage.WAIT_CUTTING_ACTIVITY,
        required_models=("hand-seg.onnx", "scissor-seg.pt"),
        expected_physical_action="Cut the packet with scissors (hand + scissors together at cutting region), then clear both from view.",
        required_detection_event="Hand and scissors detected together, then both absent from the inspection region",
        display_instruction="READY TO CUT",
        stable_frames_required=2,
        timeout_seconds=45.0,
    ),
    AutoStage.VERIFY_PHYSICAL_CUT: StageSpec(
        stage=AutoStage.VERIFY_PHYSICAL_CUT,
        required_models=("packet-seg.onnx", "cut-seg.pt", "hand-seg.onnx", "scissor-seg.pt"),
        expected_physical_action="Keep hands and scissors clear with two physically detached packet pieces separated by visible testbed background.",
        required_detection_event="Cut model positive + 2 detached pieces + exposed testbed background gap stable across fresh frames after 10 frames of clear-view analysis",
        display_instruction="Verifying physical cut — keep view clear.",
        stable_frames_required=4,
        timeout_seconds=25.0,
    ),
    AutoStage.WAIT_UPRIGHT_MOUTH: StageSpec(
        stage=AutoStage.WAIT_UPRIGHT_MOUTH,
        required_models=("packet-seg.onnx", "jeweltrack.onnx"),
        expected_physical_action="Hold the packet upright and present its mouth clearly in focus to the camera.",
        required_detection_event="Packet lifted upright (not flat pieces on bed) + focused physical mouth classified OPEN or CLOSED after 10 frames; gold visibility checked for OPEN",
        display_instruction="SHOW THE PACKET MOUTH TO THE CAMERA. Hold the packet upright.",
        stable_frames_required=4,
        timeout_seconds=30.0,
    ),
    AutoStage.WAIT_GOLD_PLACEMENT: StageSpec(
        stage=AutoStage.WAIT_GOLD_PLACEMENT,
        required_models=("packet-seg.onnx", "jeweltrack.onnx", "hand-seg.onnx"),
        expected_physical_action="Remove the gold from the packet and place it on the testbed.",
        required_detection_event="At least one confirmed article track on the testbed outside the packet and hand.",
        display_instruction="Remove the gold from the packet and place it on the testbed.",
        stable_frames_required=2,
        timeout_seconds=60.0,
    ),
    AutoStage.WAIT_JEWELLERY_AND_COUNT: StageSpec(
        stage=AutoStage.WAIT_JEWELLERY_AND_COUNT,
        required_models=("packet-seg.onnx", "jeweltrack.onnx", "hand-seg.onnx"),
        expected_physical_action="Place jewellery from this packet inside the testbed ROI (session stays active with provisional count until STOP).",
        required_detection_event="Placed articles tracked inside testbed ROI (excluding packet/hand); no automated completion event defined in workflow",
        display_instruction="Gold placement detected. Place any remaining gold on the testbed.",
        stable_frames_required=2,
        timeout_seconds=60.0,
    ),
    AutoStage.STOPPED_OR_SAVED: StageSpec(
        stage=AutoStage.STOPPED_OR_SAVED,
        required_models=(),
        expected_physical_action="Session saved. Press START for a new session.",
        required_detection_event="Session finalized and saved",
        display_instruction="Session saved. Review report or press START.",
        stable_frames_required=1,
        timeout_seconds=0.0,
    ),
}


ALLOWED_TRANSITIONS: Dict[AutoStage, Set[AutoStage]] = {
    AutoStage.IDLE: {AutoStage.WAIT_EMPTY_TESTBED, AutoStage.RELEASE_OBSERVATION},
    AutoStage.RELEASE_OBSERVATION: {AutoStage.STOPPED_OR_SAVED},
    AutoStage.WAIT_EMPTY_TESTBED: {AutoStage.WAIT_FRONTSIDE_AND_OCR, AutoStage.STOPPED_OR_SAVED},
    AutoStage.WAIT_FRONTSIDE_AND_OCR: {
        AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP,
        AutoStage.STOPPED_OR_SAVED,
    },
    AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP: {
        AutoStage.WAIT_CUTTING_ACTIVITY,
        AutoStage.STOPPED_OR_SAVED,
    },
    AutoStage.WAIT_CUTTING_ACTIVITY: {
        AutoStage.VERIFY_PHYSICAL_CUT,
        AutoStage.STOPPED_OR_SAVED,
    },
    AutoStage.VERIFY_PHYSICAL_CUT: {
        AutoStage.WAIT_UPRIGHT_MOUTH,
        AutoStage.STOPPED_OR_SAVED,
    },
    AutoStage.WAIT_UPRIGHT_MOUTH: {
        AutoStage.WAIT_GOLD_PLACEMENT,
        AutoStage.STOPPED_OR_SAVED,
    },
    AutoStage.WAIT_GOLD_PLACEMENT: {
        AutoStage.WAIT_JEWELLERY_AND_COUNT,
        AutoStage.STOPPED_OR_SAVED,
    },
    AutoStage.WAIT_JEWELLERY_AND_COUNT: {
        AutoStage.STOPPED_OR_SAVED,
    },
    AutoStage.STOPPED_OR_SAVED: {
        AutoStage.RELEASE_OBSERVATION,
        AutoStage.IDLE,
        AutoStage.WAIT_EMPTY_TESTBED,
    },
}


@dataclass
class StageTransitionEvent:
    session_id: str
    packet_id: str
    previous_stage: str
    current_stage: str
    expected_event: str
    actual_detected_event: str
    model_used: str
    timestamp: str
    confidence: float
    result: str
    compliance_status: str
    evidence_reference: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventDrivenStateMachine:
    """Tracks strict one-transition-per-frame progression and transition history."""

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        self._stage: AutoStage = AutoStage.IDLE
        self._transitions: List[StageTransitionEvent] = []

    @property
    def stage(self) -> AutoStage:
        return self._stage

    @property
    def spec(self) -> StageSpec:
        return STATE_MACHINE_SPECS[self._stage]

    @property
    def transitions(self) -> List[StageTransitionEvent]:
        return list(self._transitions)

    def reset(self, session_id: str) -> None:
        self.session_id = session_id
        self._stage = AutoStage.IDLE
        self._transitions.clear()

    def transition(
        self,
        target: AutoStage,
        packet_id: str,
        actual_detected_event: str,
        model_used: str,
        confidence: float = 1.0,
        result: str = "TRANSITIONED",
        compliance_status: str = "COMPLIANT",
        evidence_reference: Optional[str] = None,
    ) -> StageTransitionEvent:
        if target != self._stage and target not in ALLOWED_TRANSITIONS.get(self._stage, set()):
            raise ValueError(f"Disallowed transition: {self._stage.value} -> {target.value}")
        prev = self._stage
        expected_ev = STATE_MACHINE_SPECS[prev].required_detection_event
        self._stage = target
        evt = StageTransitionEvent(
            session_id=self.session_id,
            packet_id=packet_id,
            previous_stage=prev.value,
            current_stage=target.value,
            expected_event=expected_ev,
            actual_detected_event=actual_detected_event,
            model_used=model_used,
            timestamp=utc_now_iso(),
            confidence=round(float(confidence), 4),
            result=result,
            compliance_status=compliance_status,
            evidence_reference=evidence_reference,
        )
        self._transitions.append(evt)
        return evt

    def record_event(
        self,
        packet_id: str,
        actual_detected_event: str,
        model_used: str,
        confidence: float,
        result: str,
        compliance_status: str,
        evidence_reference: Optional[str] = None,
    ) -> StageTransitionEvent:
        evt = StageTransitionEvent(
            session_id=self.session_id,
            packet_id=packet_id,
            previous_stage=self._stage.value,
            current_stage=self._stage.value,
            expected_event=self.spec.required_detection_event,
            actual_detected_event=actual_detected_event,
            model_used=model_used,
            timestamp=utc_now_iso(),
            confidence=round(float(confidence), 4),
            result=result,
            compliance_status=compliance_status,
            evidence_reference=evidence_reference,
        )
        self._transitions.append(evt)
        return evt

    def to_list(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._transitions]
