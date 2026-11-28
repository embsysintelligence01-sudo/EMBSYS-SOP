"""Structured result models for the Post-Cutting Inspection module."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PacketInspectionOutcome:
    """Stage 1 outcome: Packet OPEN / CLOSED / UNKNOWN based on gold visibility inside packet."""

    status: str = "UNKNOWN"  # OPEN | CLOSED | UNKNOWN
    gold_visible: Optional[bool] = None
    inspection_valid: bool = False
    confidence: float = 0.0
    reason: str = "Not evaluated"
    stable_frames: int = 0
    required_frames: int = 5
    gold_area_px: int = 0
    interior_area_px: int = 0
    hand_occlusion_ratio: float = 0.0
    timestamp: str = field(default_factory=utc_now_iso)
    evidence_image: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "gold_visible": self.gold_visible,
            "inspection_valid": bool(self.inspection_valid),
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "stable_frames": int(self.stable_frames),
            "required_frames": int(self.required_frames),
            "gold_area_px": int(self.gold_area_px),
            "interior_area_px": int(self.interior_area_px),
            "hand_occlusion_ratio": round(float(self.hand_occlusion_ratio), 4),
            "timestamp": self.timestamp,
            "evidence_image": self.evidence_image,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class TrackedArticleSummary:
    article_id: str
    bbox: List[int]
    centroid: List[float]
    area_px: int
    confidence: float
    separated_from_cluster: bool = False
    uncertain: bool = False


@dataclass
class JewelleryCountingOutcome:
    """Stage 2 outcome: Individual jewellery count on the testbed."""

    count: Optional[int] = None
    live_count: int = 0
    count_status: str = "NOT_STARTED"  # NOT_STARTED | WAITING_EMPTY | WAITING_JEWELLERY | TRACKING | PENDING_CONFIRMATION | CONFIRMED | COUNT_UNCERTAIN
    appraiser_confirmed: bool = False
    appraiser_override_count: Optional[int] = None
    confidence: float = 0.0
    reason: str = "Stage 2 not started"
    stable_frames: int = 0
    required_stable_frames: int = 8
    articles: List[TrackedArticleSummary] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_now_iso)
    evidence_image: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_uncertain(self) -> bool:
        return self.count_status == "COUNT_UNCERTAIN"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "live_count": int(self.live_count),
            "count_status": self.count_status,
            "appraiser_confirmed": bool(self.appraiser_confirmed),
            "appraiser_override_count": self.appraiser_override_count,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "stable_frames": int(self.stable_frames),
            "required_stable_frames": int(self.required_stable_frames),
            "articles": [asdict(a) for a in self.articles],
            "timestamp": self.timestamp,
            "evidence_image": self.evidence_image,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class PostCutInspectionResult:
    """Independent post-cutting inspection record linked to a confirmed cut packet."""

    packet_id: str
    cut_status: str = "CUTTED"  # Preserved from upstream CUT_CONFIRMED event
    packet_inspection: PacketInspectionOutcome = field(default_factory=PacketInspectionOutcome)
    jewellery_counting: JewelleryCountingOutcome = field(default_factory=JewelleryCountingOutcome)
    inspection_status: str = "IN_PROGRESS"  # IN_PROGRESS | AWAITING_CONFIRMATION | UNCERTAIN | COMPLETED | ABORTED
    started_at: str = field(default_factory=utc_now_iso)
    completed_at: Optional[str] = None
    state_history: List[Dict[str, str]] = field(default_factory=list)
    model_report: Dict[str, Any] = field(default_factory=dict)

    def can_complete(self) -> bool:
        """Enforce rule: never mark an uncertain or unconfirmed inspection as completed."""
        stage1_valid = (
            self.packet_inspection.inspection_valid
            and self.packet_inspection.status in ("OPEN", "CLOSED")
        )
        stage2_valid = (
            self.jewellery_counting.count_status == "CONFIRMED"
            and self.jewellery_counting.appraiser_confirmed
            and self.jewellery_counting.count is not None
        )
        return stage1_valid and stage2_valid

    def mark_completed(self) -> bool:
        if not self.can_complete():
            if (
                not self.packet_inspection.inspection_valid
                or self.jewellery_counting.count_status == "COUNT_UNCERTAIN"
            ):
                self.inspection_status = "UNCERTAIN"
            return False
        self.inspection_status = "COMPLETED"
        self.completed_at = utc_now_iso()
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "cut_status": self.cut_status,
            "packet_inspection": self.packet_inspection.to_dict(),
            "jewellery_counting": self.jewellery_counting.to_dict(),
            "inspection_status": self.inspection_status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "state_history": list(self.state_history),
            "model_report": dict(self.model_report),
        }

    def save_json(self, output_path: Path | str) -> Path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return target

