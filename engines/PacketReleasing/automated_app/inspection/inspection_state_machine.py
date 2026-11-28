"""State machine for Post-Cutting Packet Inspection & Jewellery Counting.

Implements the exact post-CUT_CONFIRMED workflow without modifying any
upstream cutting verification state or decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

from .inspection_result import utc_now_iso


class InspectionState(str, Enum):
    IDLE = "IDLE"
    CUT_CONFIRMED = "CUT_CONFIRMED"
    POST_CUT_INSPECTION = "POST_CUT_INSPECTION"
    PACKET_VIEW_CHECK = "PACKET_VIEW_CHECK"
    GOLD_VISIBILITY_CHECK = "GOLD_VISIBILITY_CHECK"
    PACKET_OPEN = "PACKET_OPEN"
    PACKET_CLOSED = "PACKET_CLOSED"
    WAIT_FOR_VALID_VIEW = "WAIT_FOR_VALID_VIEW"
    JEWELLERY_REMOVAL = "JEWELLERY_REMOVAL"
    TESTBED_EMPTY_CHECK = "TESTBED_EMPTY_CHECK"
    WAIT_FOR_JEWELLERY = "WAIT_FOR_JEWELLERY"
    JEWEL_DETECTION = "JEWEL_DETECTION"
    INSTANCE_SEPARATION = "INSTANCE_SEPARATION"
    OBJECT_ID_ASSIGNMENT = "OBJECT_ID_ASSIGNMENT"
    PLACEMENT_STABILITY = "PLACEMENT_STABILITY"
    COUNT_CONFIRMATION = "COUNT_CONFIRMATION"
    COUNT_UNCERTAIN = "COUNT_UNCERTAIN"
    INSPECTION_COMPLETED = "INSPECTION_COMPLETED"


ALLOWED_TRANSITIONS: Dict[InspectionState, Set[InspectionState]] = {
    InspectionState.IDLE: {
        InspectionState.CUT_CONFIRMED,
    },
    InspectionState.CUT_CONFIRMED: {
        InspectionState.POST_CUT_INSPECTION,
    },
    InspectionState.POST_CUT_INSPECTION: {
        InspectionState.PACKET_VIEW_CHECK,
    },
    InspectionState.PACKET_VIEW_CHECK: {
        InspectionState.GOLD_VISIBILITY_CHECK,
        InspectionState.WAIT_FOR_VALID_VIEW,
    },
    InspectionState.WAIT_FOR_VALID_VIEW: {
        InspectionState.PACKET_VIEW_CHECK,
        InspectionState.GOLD_VISIBILITY_CHECK,
    },
    InspectionState.GOLD_VISIBILITY_CHECK: {
        InspectionState.PACKET_OPEN,
        InspectionState.PACKET_CLOSED,
        InspectionState.WAIT_FOR_VALID_VIEW,
        InspectionState.PACKET_VIEW_CHECK,
    },
    InspectionState.PACKET_OPEN: {
        InspectionState.JEWELLERY_REMOVAL,
        InspectionState.PACKET_VIEW_CHECK,
    },
    InspectionState.PACKET_CLOSED: {
        InspectionState.JEWELLERY_REMOVAL,
        InspectionState.PACKET_VIEW_CHECK,
    },
    InspectionState.JEWELLERY_REMOVAL: {
        InspectionState.TESTBED_EMPTY_CHECK,
    },
    InspectionState.TESTBED_EMPTY_CHECK: {
        InspectionState.WAIT_FOR_JEWELLERY,
        InspectionState.TESTBED_EMPTY_CHECK,
    },
    InspectionState.WAIT_FOR_JEWELLERY: {
        InspectionState.JEWEL_DETECTION,
        InspectionState.TESTBED_EMPTY_CHECK,
    },
    InspectionState.JEWEL_DETECTION: {
        InspectionState.INSTANCE_SEPARATION,
        InspectionState.WAIT_FOR_JEWELLERY,
    },
    InspectionState.INSTANCE_SEPARATION: {
        InspectionState.OBJECT_ID_ASSIGNMENT,
        InspectionState.COUNT_UNCERTAIN,
        InspectionState.WAIT_FOR_JEWELLERY,
    },
    InspectionState.OBJECT_ID_ASSIGNMENT: {
        InspectionState.PLACEMENT_STABILITY,
        InspectionState.JEWEL_DETECTION,
        InspectionState.WAIT_FOR_JEWELLERY,
    },
    InspectionState.PLACEMENT_STABILITY: {
        InspectionState.COUNT_CONFIRMATION,
        InspectionState.COUNT_UNCERTAIN,
        InspectionState.JEWEL_DETECTION,
        InspectionState.WAIT_FOR_JEWELLERY,
    },
    InspectionState.COUNT_UNCERTAIN: {
        InspectionState.JEWEL_DETECTION,
        InspectionState.WAIT_FOR_JEWELLERY,
        InspectionState.TESTBED_EMPTY_CHECK,
    },
    InspectionState.COUNT_CONFIRMATION: {
        InspectionState.INSPECTION_COMPLETED,
        InspectionState.JEWEL_DETECTION,
        InspectionState.WAIT_FOR_JEWELLERY,
        InspectionState.TESTBED_EMPTY_CHECK,
        InspectionState.COUNT_UNCERTAIN,
    },
    InspectionState.INSPECTION_COMPLETED: {
        InspectionState.IDLE,
        InspectionState.CUT_CONFIRMED,
    },
}


@dataclass
class StateTransitionRecord:
    from_state: str
    to_state: str
    reason: str
    timestamp: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class InspectionStateMachine:
    """Manages state transitions and history for Post-Cutting Inspection."""

    def __init__(
        self,
        on_transition: Optional[Callable[[InspectionState, InspectionState, str], None]] = None,
    ) -> None:
        self._state: InspectionState = InspectionState.IDLE
        self._history: List[StateTransitionRecord] = []
        self._on_transition = on_transition

    @property
    def state(self) -> InspectionState:
        return self._state

    @property
    def history(self) -> List[Dict[str, str]]:
        return [rec.to_dict() for rec in self._history]

    def reset(self) -> None:
        self._state = InspectionState.IDLE
        self._history.clear()

    def can_transition(self, target: InspectionState) -> bool:
        if target == self._state:
            return True
        return target in ALLOWED_TRANSITIONS.get(self._state, set())

    def transition(self, target: InspectionState, reason: str = "") -> InspectionState:
        if target == self._state:
            return self._state
        if not self.can_transition(target):
            raise ValueError(
                f"Invalid post-cut inspection transition: {self._state.value} -> {target.value} "
                f"(reason: {reason})"
            )
        prev = self._state
        self._state = target
        record = StateTransitionRecord(
            from_state=prev.value,
            to_state=target.value,
            reason=reason,
            timestamp=utc_now_iso(),
        )
        self._history.append(record)
        if self._on_transition is not None:
            self._on_transition(prev, target, reason)
        return self._state

    def activate_on_cut_confirmed(self, packet_id: str) -> InspectionState:
        """Entry point activated strictly by upstream CUT_CONFIRMED."""
        if self._state == InspectionState.INSPECTION_COMPLETED:
            self.reset()
        self.transition(InspectionState.CUT_CONFIRMED, f"Upstream CUT_CONFIRMED for packet {packet_id}")
        self.transition(InspectionState.POST_CUT_INSPECTION, "Starting post-cut inspection")
        return self.transition(InspectionState.PACKET_VIEW_CHECK, "Checking packet view")

    def is_stage1(self) -> bool:
        return self._state in {
            InspectionState.POST_CUT_INSPECTION,
            InspectionState.PACKET_VIEW_CHECK,
            InspectionState.GOLD_VISIBILITY_CHECK,
            InspectionState.PACKET_OPEN,
            InspectionState.PACKET_CLOSED,
            InspectionState.WAIT_FOR_VALID_VIEW,
        }

    def is_stage2(self) -> bool:
        return self._state in {
            InspectionState.JEWELLERY_REMOVAL,
            InspectionState.TESTBED_EMPTY_CHECK,
            InspectionState.WAIT_FOR_JEWELLERY,
            InspectionState.JEWEL_DETECTION,
            InspectionState.INSTANCE_SEPARATION,
            InspectionState.OBJECT_ID_ASSIGNMENT,
            InspectionState.PLACEMENT_STABILITY,
            InspectionState.COUNT_CONFIRMATION,
            InspectionState.COUNT_UNCERTAIN,
        }

