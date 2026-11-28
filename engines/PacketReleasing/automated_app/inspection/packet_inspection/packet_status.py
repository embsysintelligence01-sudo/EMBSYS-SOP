"""Stage 1 coordinator: Packet OPEN / CLOSED / UNKNOWN detection with temporal confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from ..inspection_result import PacketInspectionOutcome, utc_now_iso
from ..utils.image_utils import RawSegInstance, draw_mask_overlay
from ..utils.inspection_config import InspectionConfig
from ..utils.model_loader import InspectionModelRegistry
from ..utils.temporal_verification import TemporalVerifier
from .gold_visibility import GoldVisibilityDetector, GoldVisibilityResult
from .interior_extractor import InteriorExtractionResult, InteriorExtractor
from .packet_roi import PacketROILocator, PacketROIResult


@dataclass
class Stage1FrameEvaluation:
    raw_status: str  # OPEN | CLOSED | UNKNOWN
    confirmed_status: str  # OPEN | CLOSED | UNKNOWN
    is_confirmed: bool
    outcome: PacketInspectionOutcome
    roi_result: PacketROIResult
    interior_result: Optional[InteriorExtractionResult]
    gold_result: Optional[GoldVisibilityResult]
    annotated_frame: np.ndarray


class PacketStatusEvaluator:
    """Evaluates packet OPEN / CLOSED / UNKNOWN across consecutive stable frames.

    Business rules:
    - OPEN: Gold jewellery is visible inside the packet depth from the top camera.
    - CLOSED: No gold jewellery is visible inside a valid, unobstructed packet interior view.
    - UNKNOWN: Packet is occluded, rotated, poorly lit, blurry, missing, or detection is unreliable.
    - Never uses TopStrip.
    - Never classifies an occluded or invalid packet view as CLOSED.
    """

    def __init__(
        self,
        config: InspectionConfig,
        model_registry: Optional[InspectionModelRegistry] = None,
    ) -> None:
        self.config = config
        self.pkt_cfg = config.packet_inspection
        self.roi_locator = PacketROILocator(config, model_registry)
        self.interior_extractor = InteriorExtractor(config)
        self.gold_detector = GoldVisibilityDetector(config, model_registry)
        self.verifier: TemporalVerifier[str] = TemporalVerifier(
            required_frames=self.pkt_cfg.temporal_confirm_frames,
            window_size=self.pkt_cfg.temporal_window_size,
            reset_values={"UNKNOWN"},
        )

    def reset(self) -> None:
        self.verifier.reset()

    def evaluate_frame(
        self,
        frame: np.ndarray,
        packet_mask_override: Optional[np.ndarray] = None,
        hand_mask_override: Optional[np.ndarray] = None,
        jewel_instances_override: Optional[List[RawSegInstance]] = None,
    ) -> Stage1FrameEvaluation:
        """Process a single overhead camera frame for Stage 1 packet status."""
        roi_res = self.roi_locator.locate(
            frame,
            packet_mask_override=packet_mask_override,
            hand_mask_override=hand_mask_override,
        )

        interior_res: Optional[InteriorExtractionResult] = None
        gold_res: Optional[GoldVisibilityResult] = None

        if not roi_res.detected or not roi_res.view_valid or roi_res.packet_mask is None:
            raw_status = "UNKNOWN"
            gold_visible: Optional[bool] = None
            confidence = 0.0
            reason = roi_res.reason
        else:
            interior_res = self.interior_extractor.extract(
                frame,
                roi_res.packet_mask,
                hand_mask=roi_res.hand_mask,
            )
            if not interior_res.interior_valid:
                raw_status = "UNKNOWN"
                gold_visible = None
                confidence = 0.0
                reason = interior_res.reason
            else:
                gold_res = self.gold_detector.detect(
                    frame,
                    interior_res.interior_mask,
                    padded_roi=roi_res.padded_roi,
                    jewel_instances_override=jewel_instances_override,
                )
                if gold_res.decision == "GOLD_VISIBLE":
                    raw_status = "OPEN"
                    gold_visible = True
                    confidence = gold_res.confidence
                    reason = gold_res.reason
                elif gold_res.decision == "NO_GOLD":
                    raw_status = "CLOSED"
                    gold_visible = False
                    confidence = gold_res.confidence
                    reason = gold_res.reason
                else:
                    raw_status = "UNKNOWN"
                    gold_visible = None
                    confidence = gold_res.confidence
                    reason = gold_res.reason

        temporal = self.verifier.update(raw_status)
        confirmed_status = temporal.confirmed_value if temporal.is_confirmed and temporal.confirmed_value else "UNKNOWN"
        inspection_valid = bool(temporal.is_confirmed and confirmed_status in ("OPEN", "CLOSED"))

        diagnostics = {
            "raw_status": raw_status,
            "temporal_streak": temporal.streak,
            "window_agreement": round(temporal.window_agreement, 3),
            "roi": dict(roi_res.diagnostics),
            "interior": dict(interior_res.diagnostics) if interior_res else {},
            "gold": dict(gold_res.diagnostics) if gold_res else {},
        }

        outcome = PacketInspectionOutcome(
            status=confirmed_status if inspection_valid else "UNKNOWN",
            gold_visible=(
                True if confirmed_status == "OPEN" else (False if confirmed_status == "CLOSED" else gold_visible)
            ),
            inspection_valid=inspection_valid,
            confidence=confidence if inspection_valid else (confidence * 0.5),
            reason=reason,
            stable_frames=temporal.streak,
            required_frames=temporal.required_frames,
            gold_area_px=gold_res.gold_area_px if gold_res else 0,
            interior_area_px=interior_res.interior_area_px if interior_res else 0,
            hand_occlusion_ratio=roi_res.hand_occlusion_ratio,
            timestamp=utc_now_iso(),
            diagnostics=diagnostics,
        )

        annotated = self._render_overlay(
            frame=frame,
            roi_res=roi_res,
            interior_res=interior_res,
            gold_res=gold_res,
            raw_status=raw_status,
            confirmed_status=confirmed_status,
            streak=temporal.streak,
            required=temporal.required_frames,
            reason=reason,
        )

        return Stage1FrameEvaluation(
            raw_status=raw_status,
            confirmed_status=confirmed_status,
            is_confirmed=inspection_valid,
            outcome=outcome,
            roi_result=roi_res,
            interior_result=interior_res,
            gold_result=gold_res,
            annotated_frame=annotated,
        )

    @staticmethod
    def _render_overlay(
        frame: np.ndarray,
        roi_res: PacketROIResult,
        interior_res: Optional[InteriorExtractionResult],
        gold_res: Optional[GoldVisibilityResult],
        raw_status: str,
        confirmed_status: str,
        streak: int,
        required: int,
        reason: str,
    ) -> np.ndarray:
        vis = frame.copy()

        if roi_res.packet_mask is not None:
            vis = draw_mask_overlay(vis, roi_res.packet_mask, (255, 140, 40), alpha=0.18, draw_contour=True)
        if roi_res.hand_mask is not None and np.any(roi_res.hand_mask > 0):
            vis = draw_mask_overlay(vis, roi_res.hand_mask, (60, 60, 235), alpha=0.30, draw_contour=True)
        if interior_res is not None and np.any(interior_res.interior_mask > 0):
            vis = draw_mask_overlay(vis, interior_res.interior_mask, (230, 210, 40), alpha=0.20, draw_contour=True)
        if gold_res is not None and np.any(gold_res.gold_mask > 0):
            vis = draw_mask_overlay(vis, gold_res.gold_mask, (0, 215, 255), alpha=0.45, draw_contour=True)

        if roi_res.bbox is not None:
            x1, y1, x2, y2 = roi_res.bbox
            box_color = (60, 220, 120) if roi_res.view_valid else (60, 140, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                vis,
                f"PACKET ROI ({roi_res.confidence:.2f})",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2,
            )

        banner_color = {
            "OPEN": (50, 220, 120),
            "CLOSED": (240, 190, 50),
            "UNKNOWN": (60, 150, 255),
        }.get(confirmed_status if confirmed_status != "UNKNOWN" else raw_status, (60, 150, 255))

        cv2.rectangle(vis, (0, 0), (vis.shape[1], 56), (16, 24, 38), -1)
        cv2.putText(
            vis,
            f"STAGE 1 PACKET: {confirmed_status} (raw: {raw_status} | {streak}/{required})",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            banner_color,
            2,
        )
        cv2.putText(
            vis,
            reason[:88],
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (215, 225, 238),
            1,
        )
        return vis

