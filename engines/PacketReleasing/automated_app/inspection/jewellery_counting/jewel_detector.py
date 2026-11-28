"""Jewellery detection strictly within the Stage 2 testbed ROI.

Reuses the existing `jeweltrack.onnx` segmentation model, strictly enforces
that jewellery inside a packet or outside the testbed ROI is never counted,
and reports hand presence over the testbed so stability waits until hands clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils.image_utils import (
    RawSegInstance,
    extract_yolo_instances,
    mask_centroid_and_bbox,
    union_masks,
)
from ..utils.inspection_config import InspectionConfig
from ..utils.model_loader import InspectionModelRegistry


@dataclass
class TestbedDetectionResult:
    detections: List[RawSegInstance]
    hand_present: bool
    packet_present: bool
    hand_mask: np.ndarray
    packet_mask: np.ndarray
    roi: Tuple[int, int, int, int]
    model_available: bool
    excluded_outside_roi: int = 0
    excluded_inside_packet: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class JewelDetector:
    """Detects jewellery on the green rectangular testbed ROI only."""

    def __init__(
        self,
        config: InspectionConfig,
        model_registry: Optional[InspectionModelRegistry] = None,
    ) -> None:
        self.config = config
        self.jwl_cfg = config.jewellery_counting
        self.registry = model_registry

    def _predict_instances(
        self,
        model_key: str,
        frame: np.ndarray,
        conf: float,
        min_area_px: int,
    ) -> Tuple[List[RawSegInstance], bool]:
        if self.registry is None or not self.registry.has(model_key):
            return [], False
        model = self.registry.get(model_key)
        res = model.predict(
            source=frame,
            imgsz=self.config.models.imgsz,
            conf=conf,
            device=self.registry.device,
            verbose=False,
        )[0]
        instances = extract_yolo_instances(
            res,
            frame.shape[:2],
            selected_ids=self.registry.get_class_ids(model_key),
            min_conf=conf,
            min_area_px=min_area_px,
        )
        return instances, True

    def detect(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int],
        jewel_instances_override: Optional[List[RawSegInstance]] = None,
        hand_mask_override: Optional[np.ndarray] = None,
        packet_mask_override: Optional[np.ndarray] = None,
    ) -> TestbedDetectionResult:
        """Run jewellery detection restricted to `roi`, excluding packet interiors."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = roi
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 255
        roi_bool = roi_mask > 0

        # 1. Hand detection inside testbed ROI
        if hand_mask_override is not None:
            hand_mask = (hand_mask_override > 0).astype(np.uint8) * 255
        else:
            hand_insts, _ = self._predict_instances("hand", frame, self.jwl_cfg.hand_conf, min_area_px=150)
            hand_mask = union_masks([i.mask for i in hand_insts], (h, w))

        hand_in_roi_px = int(np.count_nonzero((hand_mask > 0) & roi_bool))
        hand_present = hand_in_roi_px >= 250

        # 2. Packet detection (so jewellery still inside a packet is NEVER counted)
        if packet_mask_override is not None:
            packet_mask = (packet_mask_override > 0).astype(np.uint8) * 255
        else:
            pkt_insts, _ = self._predict_instances("packet", frame, self.jwl_cfg.packet_conf, min_area_px=1500)
            packet_mask = union_masks([i.mask for i in pkt_insts], (h, w))

        pkt_in_roi_px = int(np.count_nonzero((packet_mask > 0) & roi_bool))
        roi_area = max(1, (x2 - x1) * (y2 - y1))
        packet_present = (float(pkt_in_roi_px) / float(roi_area)) >= 0.08

        # 3. Jewellery segmentation on full frame (and cropped testbed ROI when model available)
        if jewel_instances_override is not None:
            raw_jewels = list(jewel_instances_override)
            model_available = True
        else:
            raw_jewels, model_available = self._predict_instances(
                "jewel",
                frame,
                self.jwl_cfg.jewel_conf,
                min_area_px=max(40, self.jwl_cfg.min_article_area_px // 2),
            )

        # 4. Filter strictly to testbed ROI and exclude detections inside any packet mask
        filtered: List[RawSegInstance] = []
        excluded_outside_roi = 0
        excluded_inside_packet = 0
        pkt_bool = packet_mask > 0

        for inst in raw_jewels:
            inst_bool = inst.mask > 0
            inst_area = int(np.count_nonzero(inst_bool))
            if inst_area < self.jwl_cfg.min_article_area_px:
                continue

            # Exclude jewellery sitting inside a packet!
            if np.any(pkt_bool):
                in_pkt = int(np.count_nonzero(inst_bool & pkt_bool))
                if float(in_pkt) / float(inst_area) >= 0.45:
                    excluded_inside_packet += 1
                    continue

            # Must be predominantly inside the testbed ROI
            in_roi_bool = inst_bool & roi_bool
            in_roi_area = int(np.count_nonzero(in_roi_bool))
            overlap_ratio = float(in_roi_area) / float(inst_area)
            if overlap_ratio < self.jwl_cfg.min_roi_overlap_ratio or in_roi_area < self.jwl_cfg.min_article_area_px:
                excluded_outside_roi += 1
                continue

            clipped_mask = np.zeros((h, w), dtype=np.uint8)
            clipped_mask[in_roi_bool] = 255
            centroid, bbox, area = mask_centroid_and_bbox(clipped_mask)
            filtered.append(
                RawSegInstance(
                    mask=clipped_mask,
                    bbox=bbox,
                    confidence=inst.confidence,
                    class_id=inst.class_id,
                    area=area,
                    centroid=centroid,
                )
            )

        return TestbedDetectionResult(
            detections=filtered,
            hand_present=hand_present,
            packet_present=packet_present,
            hand_mask=hand_mask,
            packet_mask=packet_mask,
            roi=roi,
            model_available=model_available,
            excluded_outside_roi=excluded_outside_roi,
            excluded_inside_packet=excluded_inside_packet,
            diagnostics={
                "raw_jewels": len(raw_jewels),
                "kept_jewels": len(filtered),
                "excluded_outside_roi": excluded_outside_roi,
                "excluded_inside_packet": excluded_inside_packet,
                "hand_in_roi_px": hand_in_roi_px,
                "packet_in_roi_px": pkt_in_roi_px,
            },
        )

