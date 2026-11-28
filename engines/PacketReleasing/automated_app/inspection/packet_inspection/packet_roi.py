"""Packet localization, ROI extraction, and view-validity checks for Stage 1.

Uses the existing `packet-seg.onnx` (`bag` class) and optional `hand-seg.onnx`
models. Explicitly never uses the TopStrip (`strip-seg.onnx`) model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..utils.image_utils import (
    FrameQualityMetrics,
    clamp_roi,
    evaluate_frame_quality,
    extract_yolo_instances,
    mask_centroid_and_bbox,
    union_masks,
)
from ..utils.inspection_config import InspectionConfig
from ..utils.model_loader import InspectionModelRegistry


@dataclass
class PacketROIResult:
    detected: bool
    view_valid: bool
    reason: str
    packet_mask: Optional[np.ndarray] = None
    hand_mask: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    padded_roi: Optional[Tuple[int, int, int, int]] = None
    roi_crop: Optional[np.ndarray] = None
    confidence: float = 0.0
    area_px: int = 0
    solidity: float = 0.0
    aspect_ratio: float = 0.0
    tilt_deg: float = 0.0
    hand_occlusion_ratio: float = 0.0
    quality: Optional[FrameQualityMetrics] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class PacketROILocator:
    """Locates the vertically held packet and validates view quality."""

    def __init__(
        self,
        config: InspectionConfig,
        model_registry: Optional[InspectionModelRegistry] = None,
    ) -> None:
        self.config = config
        self.pkt_cfg = config.packet_inspection
        self.registry = model_registry

    def _run_packet_model(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float, Optional[str]]:
        if self.registry is None or not self.registry.has("packet"):
            return None, 0.0, "Packet segmentation model is not loaded."

        model = self.registry.get("packet")
        result = model.predict(
            source=frame,
            imgsz=self.config.models.imgsz,
            conf=self.pkt_cfg.packet_conf,
            device=self.registry.device,
            verbose=False,
        )[0]
        instances = extract_yolo_instances(
            result,
            frame.shape[:2],
            selected_ids=self.registry.get_class_ids("packet"),
            min_conf=self.pkt_cfg.packet_conf,
            min_area_px=200,
        )
        if not instances:
            return None, 0.0, None

        # Pick the primary packet instance (largest area weighted by confidence).
        best = max(instances, key=lambda inst: inst.area * max(inst.confidence, 0.1))
        return best.mask, best.confidence, None

    def _run_hand_model(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.registry is None or not self.registry.has("hand"):
            return np.zeros((h, w), dtype=np.uint8)

        model = self.registry.get("hand")
        result = model.predict(
            source=frame,
            imgsz=self.config.models.imgsz,
            conf=self.pkt_cfg.hand_conf,
            device=self.registry.device,
            verbose=False,
        )[0]
        instances = extract_yolo_instances(
            result,
            frame.shape[:2],
            selected_ids=self.registry.get_class_ids("hand"),
            min_conf=self.pkt_cfg.hand_conf,
            min_area_px=150,
        )
        return union_masks([inst.mask for inst in instances], (h, w))

    @staticmethod
    def _geometry_metrics(packet_mask: np.ndarray, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        h, w = frame_shape[:2]
        binary = (packet_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "area": 0,
                "solidity": 0.0,
                "aspect_ratio": 0.0,
                "tilt_deg": 0.0,
                "border_edges": 0,
                "minor_dim": 0.0,
            }

        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        hull = cv2.convexHull(largest)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0

        rect = cv2.minAreaRect(largest)
        (_, _), (rw, rh), angle = rect
        major = max(rw, rh, 1.0)
        minor = max(min(rw, rh), 1.0)
        aspect_ratio = minor / major

        # Compute deviation from axis-aligned (0 or 90 degrees).
        # OpenCV minAreaRect angle is in [0, 90] or [-90, 0).
        norm_angle = abs(float(angle)) % 90.0
        tilt_deg = min(norm_angle, 90.0 - norm_angle)

        x, y, bw, bh = cv2.boundingRect(largest)
        margin = 4
        border_edges = int(x <= margin) + int(y <= margin) + int((x + bw) >= (w - margin)) + int((y + bh) >= (h - margin))

        return {
            "area": int(np.count_nonzero(binary)),
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "tilt_deg": tilt_deg,
            "border_edges": border_edges,
            "minor_dim": minor,
        }

    def locate(
        self,
        frame: np.ndarray,
        packet_mask_override: Optional[np.ndarray] = None,
        hand_mask_override: Optional[np.ndarray] = None,
        packet_conf_override: Optional[float] = None,
    ) -> PacketROIResult:
        """Locate packet ROI and validate lighting, focus, occlusion, and orientation."""
        if frame is None or frame.size == 0:
            return PacketROIResult(
                detected=False,
                view_valid=False,
                reason="Empty camera frame.",
            )

        h, w = frame.shape[:2]

        # 1. Global lighting and focus quality check
        global_quality = evaluate_frame_quality(
            frame,
            min_blur_variance=self.pkt_cfg.min_blur_variance,
            min_mean_brightness=self.pkt_cfg.min_mean_brightness,
            max_mean_brightness=self.pkt_cfg.max_mean_brightness,
        )
        if global_quality.is_too_dark:
            return PacketROIResult(
                detected=False,
                view_valid=False,
                reason=f"Poor lighting: frame too dark (mean brightness {global_quality.mean_brightness:.1f}).",
                quality=global_quality,
            )
        if global_quality.is_overexposed:
            return PacketROIResult(
                detected=False,
                view_valid=False,
                reason=f"Poor lighting: frame overexposed (mean brightness {global_quality.mean_brightness:.1f}).",
                quality=global_quality,
            )

        # 2. Packet segmentation
        if packet_mask_override is not None:
            packet_mask = (packet_mask_override > 0).astype(np.uint8) * 255
            pkt_conf = float(packet_conf_override) if packet_conf_override is not None else 0.90
            model_err = None
        else:
            packet_mask, pkt_conf, model_err = self._run_packet_model(frame)

        if model_err is not None:
            return PacketROIResult(
                detected=False,
                view_valid=False,
                reason=model_err,
                quality=global_quality,
            )

        if packet_mask is None or not np.any(packet_mask > 0):
            return PacketROIResult(
                detected=False,
                view_valid=False,
                reason="Packet not detected under camera.",
                quality=global_quality,
            )

        # 3. Hand segmentation for occlusion detection
        if hand_mask_override is not None:
            hand_mask = (hand_mask_override > 0).astype(np.uint8) * 255
        else:
            hand_mask = self._run_hand_model(frame)

        geom = self._geometry_metrics(packet_mask, (h, w))
        area_px = int(geom["area"])
        solidity = float(geom["solidity"])
        aspect_ratio = float(geom["aspect_ratio"])
        tilt_deg = float(geom["tilt_deg"])
        border_edges = int(geom["border_edges"])

        _, bbox, _ = mask_centroid_and_bbox(packet_mask)
        x1, y1, x2, y2 = bbox
        pad_x = int(round((x2 - x1) * self.pkt_cfg.roi_crop_padding_ratio))
        pad_y = int(round((y2 - y1) * self.pkt_cfg.roi_crop_padding_ratio))
        padded_roi = clamp_roi((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), (h, w))
        px1, py1, px2, py2 = padded_roi
        roi_crop = frame[py1:py2, px1:px2].copy()

        # Check ROI focus & lighting specifically on the packet
        roi_quality = evaluate_frame_quality(
            frame,
            min_blur_variance=self.pkt_cfg.min_blur_variance,
            min_mean_brightness=self.pkt_cfg.min_mean_brightness,
            max_mean_brightness=self.pkt_cfg.max_mean_brightness,
            roi_mask=packet_mask,
        )

        # Compute hand overlap over packet
        overlap_px = int(np.count_nonzero((packet_mask > 0) & (hand_mask > 0)))
        hand_occlusion_ratio = float(overlap_px) / float(max(area_px, 1))

        diagnostics = {
            "packet_area_px": area_px,
            "solidity": round(solidity, 4),
            "aspect_ratio": round(aspect_ratio, 4),
            "tilt_deg": round(tilt_deg, 2),
            "border_edges": border_edges,
            "hand_occlusion_ratio": round(hand_occlusion_ratio, 4),
            "blur_variance": round(roi_quality.blur_variance, 2),
            "mean_brightness": round(roi_quality.mean_brightness, 2),
        }

        # 4. Strict view-validity checks (never classify occluded/bad views as CLOSED)
        view_valid = True
        reason = "Valid packet view."

        if area_px < self.pkt_cfg.min_packet_area_px:
            view_valid = False
            reason = f"Packet ROI too small ({area_px}px < {self.pkt_cfg.min_packet_area_px}px)."
        elif roi_quality.is_too_dark:
            view_valid = False
            reason = f"Poor lighting on packet ROI (brightness {roi_quality.mean_brightness:.1f})."
        elif roi_quality.is_overexposed:
            view_valid = False
            reason = f"Overexposed packet ROI (brightness {roi_quality.mean_brightness:.1f})."
        elif roi_quality.is_blurry:
            view_valid = False
            reason = f"Packet view is blurry (Laplacian variance {roi_quality.blur_variance:.1f})."
        elif hand_occlusion_ratio > self.pkt_cfg.max_hand_occlusion_ratio:
            view_valid = False
            reason = (
                f"Packet partially occluded by hand ({hand_occlusion_ratio:.0%} > "
                f"{self.pkt_cfg.max_hand_occlusion_ratio:.0%})."
            )
        elif solidity < self.pkt_cfg.min_packet_solidity:
            view_valid = False
            reason = (
                f"Packet contour heavily occluded or distorted (solidity {solidity:.2f} < "
                f"{self.pkt_cfg.min_packet_solidity:.2f})."
            )
        elif border_edges > self.pkt_cfg.max_border_touch_edges:
            view_valid = False
            reason = f"Packet clipped by camera frame borders ({border_edges} edges)."
        elif aspect_ratio < self.pkt_cfg.min_aspect_ratio:
            view_valid = False
            reason = f"Packet appears edge-on or excessively narrow (aspect {aspect_ratio:.2f})."
        elif aspect_ratio < 0.78 and tilt_deg > self.pkt_cfg.max_tilt_deviation_deg:
            view_valid = False
            reason = (
                f"Packet rotated out of valid top-down alignment "
                f"(tilt {tilt_deg:.1f} deg > {self.pkt_cfg.max_tilt_deviation_deg:.1f} deg)."
            )

        return PacketROIResult(
            detected=True,
            view_valid=view_valid,
            reason=reason,
            packet_mask=packet_mask,
            hand_mask=hand_mask,
            bbox=bbox,
            padded_roi=padded_roi,
            roi_crop=roi_crop,
            confidence=pkt_conf,
            area_px=area_px,
            solidity=solidity,
            aspect_ratio=aspect_ratio,
            tilt_deg=tilt_deg,
            hand_occlusion_ratio=hand_occlusion_ratio,
            quality=roi_quality,
            diagnostics=diagnostics,
        )

