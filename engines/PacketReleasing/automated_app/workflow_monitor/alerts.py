"""Real-time procedural alert manager with deduplication and correction tracking."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# Prescribed human-readable alert catalog (Section 6)
ALERT_MESSAGES: Dict[str, Dict[str, Any]] = {
    "OCR_BEFORE_CUTTING_REQUIRED": {
        "severity": AlertSeverity.CRITICAL,
        "message": "Please present the packet identification side before cutting.",
    },
    "ORIENTATION_NOT_FOLLOWED": {
        "severity": AlertSeverity.WARNING,
        "message": "Packet orientation procedure was not followed.",
    },
    "OCR_INCOMPLETE": {
        "severity": AlertSeverity.WARNING,
        "message": "OCR identification has not been completed.",
    },
    "SEAL_STILL_OPEN_BEFORE_CUT": {
        "severity": AlertSeverity.WARNING,
        "message": "Top seal strip is still visible — verify packet orientation and seal closure before cutting.",
    },
    "CUT_VERIFICATION_INTERRUPTED": {
        "severity": AlertSeverity.WARNING,
        "message": "Hands or scissors interrupted cut verification — clear the work area until separation is confirmed.",
    },
    "PULL_CUT_PIECES_APART": {
        "severity": AlertSeverity.INFO,
        "message": "Cut detected, but pieces are touching — gently pull pieces apart to confirm physical separation.",
    },
    "PACKET_VIEW_UNCLEAR": {
        "severity": AlertSeverity.WARNING,
        "message": "Please hold the packet clearly under the inspection camera.",
    },
    "STAGE1_SKIPPED_TO_JEWELLERY": {
        "severity": AlertSeverity.WARNING,
        "message": "Post-cut packet inspection was skipped before placing jewellery on the testbed.",
    },
    "TESTBED_NOT_EMPTY": {
        "severity": AlertSeverity.WARNING,
        "message": "Please clear the green testbed before placing jewellery for counting.",
    },
    "JEWELLERY_OUTSIDE_TESTBED": {
        "severity": AlertSeverity.WARNING,
        "message": "Jewellery placement was detected outside the designated testbed.",
    },
    "JEWELLERY_COUNT_UNCERTAIN": {
        "severity": AlertSeverity.CRITICAL,
        "message": "Please ensure all jewellery articles are visible for accurate counting.",
    },
    "JEWELLERY_IN_PACKET_NOT_REMOVED": {
        "severity": AlertSeverity.INFO,
        "message": "Jewellery detected inside packet — remove articles onto the green testbed for counting.",
    },
}


@dataclass
class AlertRecord:
    alert_id: str
    code: str
    severity: str
    message: str
    timestamp: str
    session_id: str
    packet_id: str
    stage: str
    active: bool = True
    corrected: bool = False
    corrected_at: Optional[str] = None
    correction_note: str = ""
    suppressed_occurrences: int = 0
    evidence_image: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RealTimeAlertManager:
    """Emits, deduplicates, links, and resolves real-time appraiser alerts.

    Suppression rule:
    - Repeated triggers of the same continuous issue (`code`) while that issue
      remains active (or within `suppression_cooldown_seconds`) increment
      `suppressed_occurrences` on the existing alert instead of spamming the UI.
    - When the appraiser corrects the issue, `resolve(code)` marks the alert
      as `corrected=True, active=False` with timestamp so the final report
      accurately documents self-corrections.
    """

    def __init__(
        self,
        session_id: str,
        suppression_cooldown_seconds: float = 8.0,
        on_alert: Optional[Callable[[AlertRecord], None]] = None,
        on_resolved: Optional[Callable[[AlertRecord], None]] = None,
    ) -> None:
        self.session_id = session_id
        self.cooldown_sec = max(0.0, float(suppression_cooldown_seconds))
        self._on_alert = on_alert
        self._on_resolved = on_resolved

        self._counter: int = 0
        self._alerts: List[AlertRecord] = []
        self._active_by_code: Dict[str, AlertRecord] = {}
        self._last_emit_mono: Dict[str, float] = {}

    @property
    def all_alerts(self) -> List[AlertRecord]:
        return list(self._alerts)

    @property
    def active_alerts(self) -> List[AlertRecord]:
        return [a for a in self._alerts if a.active]

    def trigger(
        self,
        code: str,
        packet_id: str,
        stage: str,
        custom_message: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        evidence_image: Optional[str] = None,
        now_sec: Optional[float] = None,
    ) -> Optional[AlertRecord]:
        """Emit an alert unless suppressed by active deduplication or cooldown."""
        mono = time.monotonic() if now_sec is None else float(now_sec)

        # Suppress repeated alerts for the same continuous active issue
        existing = self._active_by_code.get(code)
        if existing is not None:
            existing.suppressed_occurrences += 1
            return None

        # Also suppress rapid flapping within cooldown window after resolution
        last_t = self._last_emit_mono.get(code)
        if last_t is not None and (mono - last_t) < self.cooldown_sec:
            # Attribute suppression to the most recent alert with this code
            for prev in reversed(self._alerts):
                if prev.code == code:
                    prev.suppressed_occurrences += 1
                    break
            return None

        meta = ALERT_MESSAGES.get(code, {})
        sev = severity if severity is not None else meta.get("severity", AlertSeverity.WARNING)
        msg = custom_message if custom_message else str(meta.get("message", code))

        self._counter += 1
        rec = AlertRecord(
            alert_id=f"ALT-{self._counter:04d}",
            code=code,
            severity=sev.value if isinstance(sev, AlertSeverity) else str(sev),
            message=msg,
            timestamp=utc_now_iso(),
            session_id=self.session_id,
            packet_id=packet_id,
            stage=stage,
            active=True,
            corrected=False,
            evidence_image=evidence_image,
        )
        self._alerts.append(rec)
        self._active_by_code[code] = rec
        self._last_emit_mono[code] = mono

        if self._on_alert is not None:
            self._on_alert(rec)
        return rec

    def resolve(self, code: str, correction_note: str = "Issue corrected by appraiser.") -> Optional[AlertRecord]:
        """Mark an active alert as corrected/resolved when the appraiser fixes the issue."""
        rec = self._active_by_code.pop(code, None)
        if rec is None:
            return None
        rec.active = False
        rec.corrected = True
        rec.corrected_at = utc_now_iso()
        rec.correction_note = correction_note
        if self._on_resolved is not None:
            self._on_resolved(rec)
        return rec

    def update_packet_id_link(self, resolved_packet_id: str) -> None:
        """Link previously UNRESOLVED session alerts to the recovered packet ID without
        erasing the fact that the alert occurred while identity was unresolved."""
        for rec in self._alerts:
            if rec.packet_id == "UNRESOLVED":
                rec.packet_id = f"{resolved_packet_id} (recovered; was UNRESOLVED)"

    def to_list(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self._alerts]

