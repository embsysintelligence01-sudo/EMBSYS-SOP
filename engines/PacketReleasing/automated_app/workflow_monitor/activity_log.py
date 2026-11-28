"""Chronological Appraiser Activity Logger with evidence capture (Section 7)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from .alerts import utc_now_iso


class ComplianceStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    DEVIATION = "DEVIATION"
    SKIPPED = "SKIPPED"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    RECOVERED = "RECOVERED"
    UNVERIFIED = "UNVERIFIED"
    INFO = "INFO"


@dataclass
class ActivityEvent:
    event_id: str
    timestamp: str
    session_id: str
    packet_id: str
    stage: str
    procedure_step: str
    expected_action: str
    actual_action: str
    compliance_status: str
    deviation_code: Optional[str] = None
    reason: str = ""
    alert_id: Optional[str] = None
    corrected: Optional[bool] = None
    evidence_image: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActivityLogger:
    """Maintains an immutable chronological audit trail of observed session events.

    Never fabricates events: only records concrete observations, transitions,
    alerts, recoveries, and outcomes.
    """

    def __init__(
        self,
        session_id: str,
        evidence_dir: Optional[Path] = None,
        save_evidence_images: bool = True,
        on_event: Optional[Callable[[ActivityEvent], None]] = None,
    ) -> None:
        self.session_id = session_id
        self.evidence_dir = evidence_dir
        self.save_evidence_images = save_evidence_images
        self._on_event = on_event
        self._counter: int = 0
        self._events: List[ActivityEvent] = []

    @property
    def events(self) -> List[ActivityEvent]:
        return list(self._events)

    def save_evidence_frame(self, frame: Optional[np.ndarray], tag: str) -> Optional[str]:
        if not self.save_evidence_images or frame is None or frame.size == 0 or self.evidence_dir is None:
            return None
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag)
        path = self.evidence_dir / f"{self.session_id}_{self._counter + 1:04d}_{safe_tag}.jpg"
        if not cv2.imwrite(str(path), frame):
            return None
        return str(path)

    def log(
        self,
        packet_id: str,
        stage: str,
        procedure_step: str,
        expected_action: str,
        actual_action: str,
        compliance_status: ComplianceStatus | str,
        deviation_code: Optional[str] = None,
        reason: str = "",
        alert_id: Optional[str] = None,
        corrected: Optional[bool] = None,
        frame: Optional[np.ndarray] = None,
        evidence_tag: Optional[str] = None,
        evidence_image: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActivityEvent:
        saved_path = evidence_image
        if saved_path is None and frame is not None and evidence_tag:
            saved_path = self.save_evidence_frame(frame, evidence_tag)

        self._counter += 1
        status_str = (
            compliance_status.value
            if isinstance(compliance_status, ComplianceStatus)
            else str(compliance_status)
        )
        ev = ActivityEvent(
            event_id=f"EVT-{self._counter:04d}",
            timestamp=utc_now_iso(),
            session_id=self.session_id,
            packet_id=packet_id,
            stage=stage,
            procedure_step=procedure_step,
            expected_action=expected_action,
            actual_action=actual_action,
            compliance_status=status_str,
            deviation_code=deviation_code,
            reason=reason,
            alert_id=alert_id,
            corrected=corrected,
            evidence_image=saved_path,
            metadata=dict(metadata) if metadata else {},
        )
        self._events.append(ev)
        if self._on_event is not None:
            self._on_event(ev)
        return ev

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._events]

