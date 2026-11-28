"""Continuous OCR Protection & Packet Identity Manager (Section 5).

Guarantees:
- Attempts OCR as soon as a packet is detected and continuously while the ID ROI is visible.
- Locks the first reliable packet number and preserves it across the entire session.
- Never fabricates or assigns an incorrect/stale packet number to an unidentified packet.
- Supports post-alert / post-cut OCR recovery if the appraiser presents the ID side later.
- Retains explicit `UNRESOLVED` status when OCR cannot recover the number while allowing
  packet processing to continue uninterrupted.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .alerts import utc_now_iso
from .config import WorkflowMonitorConfig
from .models import UnifiedModelManager


OCRCallable = Callable[[np.ndarray], Tuple[Optional[str], float, List[Tuple[str, float]]]]


@dataclass
class OCRStepObservation:
    roi_detected: bool
    roi_bbox: Optional[Tuple[int, int, int, int]]
    attempted_ocr: bool
    raw_text: Optional[str]
    raw_confidence: float
    packet_id: str  # Resolved ID or "UNRESOLVED"
    identity_status: str  # UNRESOLVED | RESOLVED | RECOVERED
    newly_resolved: bool = False
    is_recovery: bool = False
    crop: Optional[np.ndarray] = None


@dataclass
class OCRIdentitySummary:
    packet_id: str = "UNRESOLVED"
    identity_status: str = "UNRESOLVED"  # UNRESOLVED | RESOLVED | RECOVERED
    confidence: float = 0.0
    resolved_at_stage: Optional[str] = None
    resolved_at_timestamp: Optional[str] = None
    total_attempts: int = 0
    roi_detections: int = 0
    recovery_attempts: int = 0
    recovered_after_alert: bool = False
    candidates_seen: List[Dict[str, Any]] = field(default_factory=list)
    evidence_image: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OCRProtector:
    """Continuously protects packet identity before, during, and after cutting."""

    def __init__(
        self,
        config: WorkflowMonitorConfig,
        model_manager: Optional[UnifiedModelManager] = None,
        ocr_runner: Optional[OCRCallable] = None,
    ) -> None:
        self.config = config
        self.ocr_cfg = config.ocr_protection
        self.models = model_manager
        self._ocr_runner = ocr_runner
        self._lazy_ocr_loaded = False
        self._ocr_available = True

        self._frame_counter: int = 0
        self._vote_counts: Counter[str] = Counter()
        self._best_conf_by_text: Dict[str, float] = {}
        self.summary = OCRIdentitySummary()

    def reset(self) -> None:
        self._frame_counter = 0
        self._vote_counts.clear()
        self._best_conf_by_text.clear()
        self.summary = OCRIdentitySummary()

    @property
    def is_resolved(self) -> bool:
        return self.summary.identity_status in ("RESOLVED", "RECOVERED")

    @property
    def packet_id(self) -> str:
        return self.summary.packet_id

    def _get_ocr_runner(self) -> Optional[OCRCallable]:
        if self._ocr_runner is not None:
            return self._ocr_runner
        if not self._lazy_ocr_loaded:
            self._lazy_ocr_loaded = True
            try:
                from packet_cut.desktop.ocr import run_ocr  # type: ignore

                self._ocr_runner = run_ocr
                self._ocr_available = True
            except Exception:
                self._ocr_available = False
                self._ocr_runner = None
        return self._ocr_runner

    def _detect_number_roi(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        if self.models is None or not self.models.has("number"):
            return None, None

        h, w = frame.shape[:2]
        model = self.models.get("number")
        result = model.predict(
            source=frame,
            imgsz=self.config.models.number_imgsz,
            conf=self.ocr_cfg.number_conf,
            device=self.models.device,
            verbose=False,
        )[0]

        boxes = None
        if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
            boxes = result.obb.xyxy.detach().cpu().numpy()
        elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.detach().cpu().numpy()

        if boxes is None or len(boxes) == 0:
            return None, None

        box = boxes[0].astype(int)
        x1 = max(0, min(int(box[0]), w - 1))
        y1 = max(0, min(int(box[1]), h - 1))
        x2 = max(0, min(int(box[2]), w))
        y2 = max(0, min(int(box[3]), h))
        if x2 <= x1 or y2 <= y1:
            return None, None

        pad_x = int((x2 - x1) * 0.08)
        pad_y = int((y2 - y1) * 0.15)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        return frame[cy1:cy2, cx1:cx2].copy(), (x1, y1, x2, y2)

    def _normalize_candidate(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        cleaned = re.sub(r"[^0-9A-Za-z]", "", str(text).strip()).upper()
        if len(cleaned) < self.ocr_cfg.min_id_length:
            return None
        return cleaned

    def observe(
        self,
        frame: np.ndarray,
        current_stage: str,
        is_recovery_mode: bool = False,
        roi_override: Optional[Tuple[int, int, int, int]] = None,
        ocr_text_override: Optional[Tuple[Optional[str], float]] = None,
    ) -> OCRStepObservation:
        """Observe current frame; continuously attempt OCR until reliably locked."""
        self._frame_counter += 1

        # Once reliably locked, preserve identity throughout the entire session
        if self.is_resolved:
            return OCRStepObservation(
                roi_detected=False,
                roi_bbox=None,
                attempted_ocr=False,
                raw_text=None,
                raw_confidence=self.summary.confidence,
                packet_id=self.summary.packet_id,
                identity_status=self.summary.identity_status,
            )

        # 1. Detect packet number ROI
        if roi_override is not None:
            x1, y1, x2, y2 = roi_override
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)].copy()
            bbox: Optional[Tuple[int, int, int, int]] = roi_override
        elif ocr_text_override is not None:
            crop = frame
            bbox = (0, 0, frame.shape[1], frame.shape[0])
        else:
            crop, bbox = self._detect_number_roi(frame)

        if bbox is None or crop is None or crop.size == 0:
            return OCRStepObservation(
                roi_detected=False,
                roi_bbox=None,
                attempted_ocr=False,
                raw_text=None,
                raw_confidence=0.0,
                packet_id=self.summary.packet_id,
                identity_status=self.summary.identity_status,
            )

        self.summary.roi_detections += 1

        # Throttle heavy PaddleOCR calls unless explicit override or interval reached
        should_run_ocr = (
            ocr_text_override is not None
            or (self._frame_counter % max(1, self.ocr_cfg.ocr_attempt_interval_frames) == 0)
            or self.summary.total_attempts == 0
        )
        if not should_run_ocr:
            return OCRStepObservation(
                roi_detected=True,
                roi_bbox=bbox,
                attempted_ocr=False,
                raw_text=None,
                raw_confidence=0.0,
                packet_id=self.summary.packet_id,
                identity_status=self.summary.identity_status,
                crop=crop,
            )

        self.summary.total_attempts += 1
        if is_recovery_mode:
            self.summary.recovery_attempts += 1

        raw_text: Optional[str] = None
        raw_conf: float = 0.0
        if ocr_text_override is not None:
            raw_text, raw_conf = ocr_text_override
        else:
            runner = self._get_ocr_runner()
            if runner is not None:
                try:
                    raw_text, raw_conf, _ = runner(crop)
                except Exception:
                    raw_text, raw_conf = None, 0.0

        cleaned = self._normalize_candidate(raw_text)
        newly_resolved = False

        if cleaned is not None and raw_conf >= self.ocr_cfg.min_ocr_confidence:
            self._vote_counts[cleaned] += 1
            self._best_conf_by_text[cleaned] = max(self._best_conf_by_text.get(cleaned, 0.0), float(raw_conf))
            if len(self.summary.candidates_seen) < 25:
                self.summary.candidates_seen.append({
                    "text": cleaned,
                    "confidence": round(float(raw_conf), 4),
                    "stage": current_stage,
                    "timestamp": utc_now_iso(),
                })

            # Lock identity when we reach `min_reliable_reads` consistent reads OR a
            # high-confidence exact-length numeric read (>= 0.90)
            exact_digit_high_conf = (
                cleaned.isdigit()
                and len(cleaned) == self.ocr_cfg.preferred_digit_length
                and float(raw_conf) >= 0.90
            )
            if self._vote_counts[cleaned] >= self.ocr_cfg.min_reliable_reads or exact_digit_high_conf:
                self.summary.packet_id = cleaned
                self.summary.confidence = self._best_conf_by_text[cleaned]
                self.summary.identity_status = "RECOVERED" if is_recovery_mode else "RESOLVED"
                self.summary.resolved_at_stage = current_stage
                self.summary.resolved_at_timestamp = utc_now_iso()
                self.summary.recovered_after_alert = bool(is_recovery_mode)
                newly_resolved = True

        return OCRStepObservation(
            roi_detected=True,
            roi_bbox=bbox,
            attempted_ocr=True,
            raw_text=cleaned,
            raw_confidence=float(raw_conf),
            packet_id=self.summary.packet_id,
            identity_status=self.summary.identity_status,
            newly_resolved=newly_resolved,
            is_recovery=bool(is_recovery_mode and newly_resolved),
            crop=crop,
        )

