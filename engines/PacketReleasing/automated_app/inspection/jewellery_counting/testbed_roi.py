"""Testbed ROI management and empty-testbed verification for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..utils.image_utils import clamp_roi
from ..utils.inspection_config import InspectionConfig
from ..utils.temporal_verification import TemporalVerifier


@dataclass
class EmptyTestbedResult:
    raw_is_empty: bool
    confirmed_empty: bool
    streak: int
    required_frames: int
    reason: str
    foreground_ratio: float = 0.0
    roi: Tuple[int, int, int, int] = (0, 0, 0, 0)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class TestbedROIManager:
    """Defines and validates the fixed green rectangular testbed ROI."""

    def __init__(self, config: InspectionConfig) -> None:
        self.config = config
        self.jwl_cfg = config.jewellery_counting
        self._calibrated_roi: Optional[Tuple[int, int, int, int]] = None
        self._empty_baseline_roi: Optional[np.ndarray] = None
        self._empty_verifier: TemporalVerifier[bool] = TemporalVerifier(
            required_frames=self.jwl_cfg.empty_testbed_confirm_frames,
            window_size=max(self.jwl_cfg.empty_testbed_confirm_frames + 2, 6),
            reset_values={False},
        )

    def reset(self) -> None:
        self._empty_verifier.reset()

    def set_roi(self, roi: Tuple[int, int, int, int]) -> None:
        self._calibrated_roi = tuple(int(v) for v in roi)  # type: ignore[assignment]
        self._empty_baseline_roi = None

    def get_roi(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        if self._calibrated_roi is not None:
            return clamp_roi(self._calibrated_roi, (h, w))

        if self.jwl_cfg.auto_detect_green_testbed:
            detected = self.detect_green_testbed_roi(frame)
            if detected is not None:
                self._calibrated_roi = detected
                return detected

        if self.jwl_cfg.use_normalized_roi:
            nx1, ny1, nx2, ny2 = self.jwl_cfg.testbed_roi_normalized
            roi = (int(round(nx1 * w)), int(round(ny1 * h)), int(round(nx2 * w)), int(round(ny2 * h)))
            return clamp_roi(roi, (h, w))

        return clamp_roi(self.jwl_cfg.testbed_roi, (h, w))

    def get_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.get_roi(frame)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask

    @staticmethod
    def detect_green_testbed_roi(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Optional helper to detect the green rectangular testbed markings on the mat."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Green border / green mat detection
        green_mask = cv2.inRange(hsv, np.array([35, 45, 35]), np.array([92, 255, 255]))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_roi = None
        best_area = 0.0
        min_box_area = float(h * w) * 0.04
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            box_area = float(bw * bh)
            if box_area < min_box_area:
                continue
            aspect = float(min(bw, bh)) / float(max(bw, bh, 1))
            if aspect < 0.25:
                continue
            if box_area > best_area:
                best_area = box_area
                best_roi = clamp_roi((x, y, x + bw, y + bh), (h, w))
        return best_roi

    def capture_empty_baseline(self, frame: np.ndarray) -> None:
        x1, y1, x2, y2 = self.get_roi(frame)
        self._empty_baseline_roi = frame[y1:y2, x1:x2].copy()

    def foreground_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Changes from the selected empty surface, including its fixed markings.

        Compensate for a global lighting shift before comparing colours. Keep
        local changes (packets, hands, jewellery) and remove isolated sensor noise.
        """
        x1, y1, x2, y2 = self.get_roi(frame)
        crop = frame[y1:y2, x1:x2]
        baseline = self._empty_baseline_roi
        if baseline is None or baseline.shape != crop.shape:
            return None
        current = cv2.GaussianBlur(crop, (5, 5), 0).astype(np.float32)
        reference = cv2.GaussianBlur(baseline, (5, 5), 0).astype(np.float32)
        difference = current - reference
        # Bound compensation so a large newly placed object cannot become
        # the estimated illumination of the whole surface.
        lighting_shift = np.clip(np.median(difference, axis=(0, 1)), -15.0, 15.0)
        changed = np.max(np.abs(difference - lighting_shift), axis=2) > 24.0
        mask = changed.astype(np.uint8) * 255
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask
        return full_mask

    def check_empty(
        self,
        frame: np.ndarray,
        jewel_count_in_roi: int,
        hand_present_in_roi: bool = False,
        packet_present_in_roi: bool = False,
    ) -> EmptyTestbedResult:
        """Verify that the testbed ROI is empty across consecutive stable frames."""
        roi = self.get_roi(frame)
        x1, y1, x2, y2 = roi
        roi_crop = frame[y1:y2, x1:x2]

        fg_ratio = 0.0
        foreground = self.foreground_mask(frame)
        if foreground is not None:
            fg_ratio = float(np.count_nonzero(foreground)) / float(max(roi_crop.shape[0] * roi_crop.shape[1], 1))
        else:
            # Without a pre-captured baseline, verify absence of strong warm/dark foreign objects
            # on the bright/green testbed background via saturation + local deviation.
            hsv = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2HSV)
            # Exclude green testbed markings (H in 35..95) and bright neutral mat (S < 50, V > 110)
            is_green = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 40)
            is_neutral_mat = (hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 90)
            foreign = (~is_green) & (~is_neutral_mat)
            # Ignore a thin 4px border margin around the green rectangle edge
            if foreign.shape[0] > 10 and foreign.shape[1] > 10:
                foreign[:4, :] = False
                foreign[-4:, :] = False
                foreign[:, :4] = False
                foreign[:, -4:] = False
            fg_ratio = float(np.count_nonzero(foreign)) / float(max(foreign.size, 1))

        if hand_present_in_roi:
            raw_empty = False
            reason = "Hand detected over testbed ROI; clear hands to verify empty testbed."
        elif packet_present_in_roi:
            raw_empty = False
            reason = "Packet detected on testbed ROI; remove packet from testbed."
        elif jewel_count_in_roi > 0:
            raw_empty = False
            reason = f"Jewellery already detected on testbed ({jewel_count_in_roi} item(s)); clear testbed first."
        elif fg_ratio > self.jwl_cfg.empty_max_foreground_ratio:
            raw_empty = False
            reason = f"Foreign object detected on testbed ROI (occupancy {fg_ratio:.1%})."
        else:
            raw_empty = True
            reason = "Testbed ROI is clear."

        temporal = self._empty_verifier.update(raw_empty)
        if temporal.is_confirmed and self._empty_baseline_roi is None:
            self.capture_empty_baseline(frame)

        return EmptyTestbedResult(
            raw_is_empty=raw_empty,
            confirmed_empty=bool(temporal.is_confirmed),
            streak=temporal.streak,
            required_frames=temporal.required_frames,
            reason=reason,
            foreground_ratio=fg_ratio,
            roi=roi,
            diagnostics={
                "jewel_count_in_roi": jewel_count_in_roi,
                "hand_present_in_roi": hand_present_in_roi,
                "packet_present_in_roi": packet_present_in_roi,
                "foreground_ratio": round(fg_ratio, 4),
            },
        )

