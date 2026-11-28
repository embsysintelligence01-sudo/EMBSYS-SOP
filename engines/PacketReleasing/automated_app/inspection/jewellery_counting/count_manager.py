"""Stage 2 coordinator: Placement stability, count confirmation, and visualization."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import cv2
import numpy as np

from ..inspection_result import JewelleryCountingOutcome, TrackedArticleSummary, utc_now_iso
from ..utils.image_utils import RawSegInstance, distinct_bgr_color, draw_mask_overlay
from ..utils.inspection_config import InspectionConfig
from ..utils.model_loader import InspectionModelRegistry
from .instance_separator import InstanceSeparator, SeparationResult
from .jewel_detector import JewelDetector, TestbedDetectionResult
from .object_tracker import JewelleryObjectTracker, TrackingSnapshot
from .testbed_roi import EmptyTestbedResult, TestbedROIManager


@dataclass
class Stage2FrameEvaluation:
    phase: str  # TESTBED_EMPTY_CHECK | WAIT_FOR_JEWELLERY | TRACKING | COUNT_CONFIRMATION | COUNT_UNCERTAIN | CONFIRMED
    outcome: JewelleryCountingOutcome
    empty_result: Optional[EmptyTestbedResult]
    detection_result: TestbedDetectionResult
    separation_result: SeparationResult
    tracking_snapshot: TrackingSnapshot
    annotated_frame: np.ndarray


class JewelleryCountManager:
    """Manages Stage 2 lifecycle:
    1. Empty testbed check
    2. Wait for jewellery placement on testbed
    3. Detection -> Instance separation -> Unique ID tracking
    4. Placement stability verification (hand-free + stationary tracks)
    5. Appraiser count confirmation (blocks completion when COUNT_UNCERTAIN)
    """

    def __init__(
        self,
        config: InspectionConfig,
        model_registry: Optional[InspectionModelRegistry] = None,
    ) -> None:
        self.config = config
        self.jwl_cfg = config.jewellery_counting
        self.stb_cfg = config.jewellery_counting.stability

        self.roi_manager = TestbedROIManager(config)
        self.detector = JewelDetector(config, model_registry)
        self.separator = InstanceSeparator(config, model_registry)
        self.tracker = JewelleryObjectTracker(config)

        self._empty_verified: bool = False
        self._stable_frames: int = 0
        self._stable_started_at: Optional[float] = None
        self._last_stable_count: Optional[int] = None
        self._uncertain_window: Deque[bool] = deque(maxlen=max(self.stb_cfg.required_stable_frames, 5))
        self._confirmed: bool = False
        self._confirmed_count: Optional[int] = None
        self._appraiser_override: Optional[int] = None
        self._last_outcome: JewelleryCountingOutcome = JewelleryCountingOutcome(
            required_stable_frames=self.stb_cfg.required_stable_frames
        )

    @property
    def is_empty_verified(self) -> bool:
        return self._empty_verified

    @property
    def is_confirmed(self) -> bool:
        return self._confirmed

    def reset(self, require_empty_check: bool = True) -> None:
        self.roi_manager.reset()
        self.tracker.reset()
        self._empty_verified = not require_empty_check
        self._reset_stability()
        self._confirmed = False
        self._confirmed_count = None
        self._appraiser_override = None
        self._last_outcome = JewelleryCountingOutcome(
            count_status="WAITING_EMPTY" if require_empty_check else "WAITING_JEWELLERY",
            required_stable_frames=self.stb_cfg.required_stable_frames,
        )

    def _reset_stability(self) -> None:
        self._stable_frames = 0
        self._stable_started_at = None
        self._last_stable_count = None
        self._uncertain_window.clear()

    def skip_empty_check(self) -> None:
        self._empty_verified = True

    def confirm_count(self, override_count: Optional[int] = None) -> JewelleryCountingOutcome:
        """Appraiser confirmation of the final jewellery count.

        Strictly refuses to confirm if the current count is `COUNT_UNCERTAIN`
        unless the appraiser resolves separation (or explicitly recounts).
        """
        if self._last_outcome.count_status == "COUNT_UNCERTAIN" and override_count is None:
            return self._last_outcome

        final_count = override_count if override_count is not None else self._last_outcome.live_count
        if final_count is None or final_count < 0:
            return self._last_outcome

        self._confirmed = True
        self._confirmed_count = int(final_count)
        self._appraiser_override = int(override_count) if override_count is not None else None
        self._last_outcome.count = self._confirmed_count
        self._last_outcome.count_status = "CONFIRMED"
        self._last_outcome.appraiser_confirmed = True
        self._last_outcome.appraiser_override_count = self._appraiser_override
        self._last_outcome.reason = f"Final jewellery count ({self._confirmed_count}) confirmed by appraiser."
        self._last_outcome.timestamp = utc_now_iso()
        return self._last_outcome

    def evaluate_frame(
        self,
        frame: np.ndarray,
        now_sec: Optional[float] = None,
        jewel_instances_override: Optional[List[RawSegInstance]] = None,
        hand_mask_override: Optional[np.ndarray] = None,
        packet_mask_override: Optional[np.ndarray] = None,
    ) -> Stage2FrameEvaluation:
        """Process one camera frame through the Stage 2 counting pipeline."""
        ts = time.monotonic() if now_sec is None else float(now_sec)
        roi = self.roi_manager.get_roi(frame)

        det_res = self.detector.detect(
            frame,
            roi,
            jewel_instances_override=jewel_instances_override,
            hand_mask_override=hand_mask_override,
            packet_mask_override=packet_mask_override,
        )

        # Phase A: Verify testbed starts empty before accepting jewellery
        empty_res: Optional[EmptyTestbedResult] = None
        if not self._empty_verified:
            empty_res = self.roi_manager.check_empty(
                frame,
                jewel_count_in_roi=len(det_res.detections),
                hand_present_in_roi=det_res.hand_present,
                packet_present_in_roi=det_res.packet_present,
            )
            if empty_res.confirmed_empty:
                self._empty_verified = True
                phase = "WAIT_FOR_JEWELLERY"
                count_status = "WAITING_JEWELLERY"
                reason = "Empty testbed confirmed. Place jewellery articles on green testbed."
            else:
                phase = "TESTBED_EMPTY_CHECK"
                count_status = "WAITING_EMPTY"
                reason = empty_res.reason

            empty_sep = SeparationResult(articles=[], is_uncertain=False)
            empty_trk = TrackingSnapshot(active_tracks=[], confirmed_tracks=[], live_count=0)
            outcome = JewelleryCountingOutcome(
                count=None,
                live_count=0,
                count_status=count_status,
                confidence=1.0 if self._empty_verified else 0.0,
                reason=reason,
                stable_frames=empty_res.streak,
                required_stable_frames=empty_res.required_frames,
                timestamp=utc_now_iso(),
                diagnostics={"empty_check": empty_res.diagnostics, "detection": det_res.diagnostics},
            )
            self._last_outcome = outcome
            vis = self._render_overlay(frame, roi, det_res, empty_trk, outcome, phase)
            return Stage2FrameEvaluation(
                phase=phase,
                outcome=outcome,
                empty_result=empty_res,
                detection_result=det_res,
                separation_result=empty_sep,
                tracking_snapshot=empty_trk,
                annotated_frame=vis,
            )

        # Phase B: Detect -> Separate -> Track
        sep_res = self.separator.separate(frame, det_res.detections)
        trk_snap = self.tracker.update(sep_res.articles)

        article_summaries = [
            TrackedArticleSummary(
                article_id=t.track_id,
                bbox=list(t.bbox),
                centroid=[round(t.centroid[0], 1), round(t.centroid[1], 1)],
                area_px=t.area,
                confidence=round(t.confidence, 4),
                separated_from_cluster=t.separated_from_cluster,
                uncertain=t.uncertain,
            )
            for t in trk_snap.confirmed_tracks
        ]

        if not det_res.model_available:
            self._reset_stability()
            outcome = JewelleryCountingOutcome(
                count=None,
                live_count=0,
                count_status="COUNT_UNCERTAIN",
                confidence=0.0,
                reason="Jewellery segmentation model unavailable; cannot count jewellery.",
                stable_frames=0,
                required_stable_frames=self.stb_cfg.required_stable_frames,
                timestamp=utc_now_iso(),
                diagnostics={"detection": det_res.diagnostics},
            )
            self._last_outcome = outcome
            vis = self._render_overlay(frame, roi, det_res, trk_snap, outcome, "COUNT_UNCERTAIN")
            return Stage2FrameEvaluation(
                phase="COUNT_UNCERTAIN",
                outcome=outcome,
                empty_result=None,
                detection_result=det_res,
                separation_result=sep_res,
                tracking_snapshot=trk_snap,
                annotated_frame=vis,
            )

        # If appraiser already confirmed, retain confirmed state unless reset
        if self._confirmed:
            vis = self._render_overlay(frame, roi, det_res, trk_snap, self._last_outcome, "CONFIRMED")
            return Stage2FrameEvaluation(
                phase="CONFIRMED",
                outcome=self._last_outcome,
                empty_result=None,
                detection_result=det_res,
                separation_result=sep_res,
                tracking_snapshot=trk_snap,
                annotated_frame=vis,
            )

        # Check scene state for stability
        if det_res.hand_present:
            self._reset_stability()
            phase = "TRACKING"
            count_status = "TRACKING"
            reason = "Hand detected over testbed; waiting for placement to finish."
        elif det_res.packet_present:
            self._reset_stability()
            phase = "WAIT_FOR_JEWELLERY"
            count_status = "WAITING_JEWELLERY"
            reason = "Packet detected on testbed; remove packet and place jewellery only."
        elif trk_snap.live_count == 0 and not sep_res.is_uncertain:
            self._reset_stability()
            phase = "WAIT_FOR_JEWELLERY"
            count_status = "WAITING_JEWELLERY"
            reason = "Testbed is empty. Place jewellery articles inside green testbed."
        else:
            # Evaluate frame stability: same live count, small motion & area drift
            self._uncertain_window.append(sep_res.is_uncertain)

            tracks_stationary = all(
                (t.last_centroid_shift <= self.stb_cfg.max_centroid_shift_px)
                and (t.last_area_change_ratio <= self.stb_cfg.max_area_change_ratio)
                for t in trk_snap.confirmed_tracks
            )

            if (
                self._last_stable_count == trk_snap.live_count
                and tracks_stationary
                and not trk_snap.newly_added_ids
                and not trk_snap.removed_ids
            ):
                self._stable_frames += 1
                if self._stable_started_at is None:
                    self._stable_started_at = ts
            else:
                self._last_stable_count = trk_snap.live_count
                self._stable_frames = 1
                self._stable_started_at = ts

            elapsed_stable = (ts - self._stable_started_at) if self._stable_started_at is not None else 0.0
            uncertain_ratio = (
                float(sum(1 for u in self._uncertain_window if u)) / float(len(self._uncertain_window))
                if self._uncertain_window
                else 0.0
            )
            stability_met = (
                self._stable_frames >= self.stb_cfg.required_stable_frames
                and elapsed_stable >= self.stb_cfg.min_stable_seconds
            )

            if sep_res.is_uncertain or uncertain_ratio > self.stb_cfg.max_uncertain_ratio_in_window:
                phase = "COUNT_UNCERTAIN"
                count_status = "COUNT_UNCERTAIN"
                detail = "; ".join(sep_res.uncertainty_reasons[:2]) if sep_res.uncertainty_reasons else (
                    "Overlapping or ambiguous jewellery segmentation detected."
                )
                reason = f"COUNT_UNCERTAIN: {detail} Separate items clearly on testbed."
            elif stability_met and trk_snap.live_count > 0:
                phase = "COUNT_CONFIRMATION"
                count_status = "PENDING_CONFIRMATION"
                reason = (
                    f"Placement stable ({self._stable_frames} frames). "
                    f"Detected {trk_snap.live_count} article(s) — awaiting appraiser confirmation."
                )
            else:
                phase = "TRACKING"
                count_status = "TRACKING"
                reason = (
                    f"Tracking {trk_snap.live_count} article(s) — stabilizing "
                    f"({self._stable_frames}/{self.stb_cfg.required_stable_frames})."
                )

        mean_conf = (
            float(np.mean([t.confidence for t in trk_snap.confirmed_tracks]))
            if trk_snap.confirmed_tracks
            else 0.0
        )

        # Never invent a count when COUNT_UNCERTAIN
        reported_count: Optional[int] = (
            None if count_status == "COUNT_UNCERTAIN" else (
                trk_snap.live_count if count_status == "PENDING_CONFIRMATION" else None
            )
        )

        outcome = JewelleryCountingOutcome(
            count=reported_count,
            live_count=0 if count_status == "COUNT_UNCERTAIN" else trk_snap.live_count,
            count_status=count_status,
            appraiser_confirmed=False,
            confidence=0.0 if count_status == "COUNT_UNCERTAIN" else mean_conf,
            reason=reason,
            stable_frames=self._stable_frames,
            required_stable_frames=self.stb_cfg.required_stable_frames,
            articles=article_summaries,
            timestamp=utc_now_iso(),
            diagnostics={
                "detection": det_res.diagnostics,
                "separation": sep_res.diagnostics,
                "uncertainty_reasons": list(sep_res.uncertainty_reasons),
                "used_sam": sep_res.used_sam,
            },
        )
        self._last_outcome = outcome
        vis = self._render_overlay(frame, roi, det_res, trk_snap, outcome, phase)
        return Stage2FrameEvaluation(
            phase=phase,
            outcome=outcome,
            empty_result=None,
            detection_result=det_res,
            separation_result=sep_res,
            tracking_snapshot=trk_snap,
            annotated_frame=vis,
        )

    @staticmethod
    def _render_overlay(
        frame: np.ndarray,
        roi: Tuple[int, int, int, int],
        det_res: TestbedDetectionResult,
        trk_snap: TrackingSnapshot,
        outcome: JewelleryCountingOutcome,
        phase: str,
    ) -> np.ndarray:
        vis = frame.copy()
        x1, y1, x2, y2 = roi

        # Draw testbed ROI boundary
        roi_color = (50, 220, 120) if not det_res.hand_present else (60, 150, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), roi_color, 2)
        cv2.putText(
            vis,
            "TESTBED ROI",
            (x1 + 4, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            roi_color,
            2,
        )

        # Hand warning overlay
        if det_res.hand_present and np.any(det_res.hand_mask > 0):
            vis = draw_mask_overlay(vis, det_res.hand_mask, (60, 60, 240), alpha=0.30, draw_contour=True)

        # Draw individual tracked jewellery masks, bounding boxes, and unique IDs
        for idx, trk in enumerate(trk_snap.confirmed_tracks):
            color = (40, 130, 255) if trk.uncertain else distinct_bgr_color(idx)
            vis = draw_mask_overlay(vis, trk.mask, color, alpha=0.35, draw_contour=True)
            bx1, by1, bx2, by2 = trk.bbox
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
            label = f"{trk.track_id} ({trk.confidence:.2f})"
            if trk.uncertain:
                label = f"{trk.track_id} [UNCERTAIN]"
            elif trk.separated_from_cluster:
                label = f"{trk.track_id} [SPLIT]"
            cv2.putText(
                vis,
                label,
                (bx1, max(18, by1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
            )

        # Top HUD banner
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 56), (16, 24, 38), -1)
        if outcome.count_status == "COUNT_UNCERTAIN":
            banner_color = (60, 130, 255)
            header = "STAGE 2 COUNT: COUNT_UNCERTAIN (SEPARATE OVERLAPPING ITEMS)"
        elif outcome.count_status == "CONFIRMED":
            banner_color = (50, 230, 120)
            header = f"STAGE 2 COUNT CONFIRMED: {outcome.count} ARTICLE(S)"
        elif outcome.count_status == "PENDING_CONFIRMATION":
            banner_color = (240, 200, 50)
            header = f"STAGE 2 READY: {outcome.live_count} ARTICLE(S) — CONFIRM COUNT"
        else:
            banner_color = (230, 210, 80)
            header = f"STAGE 2 [{phase}] | LIVE COUNT: {outcome.live_count} ({outcome.stable_frames}/{outcome.required_stable_frames})"

        cv2.putText(vis, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.54, banner_color, 2)
        cv2.putText(vis, outcome.reason[:90], (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (215, 225, 238), 1)
        return vis

