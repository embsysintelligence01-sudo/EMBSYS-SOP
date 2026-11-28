"""Interior/depth region extractor for Stage 1 packet inspection.

Isolates the visible interior cavity inside the packet while excluding:
- Packet outer borders and heat-sealed rims
- Hands holding the packet
- Bright low-saturation printed exterior labels / table gridlines
- External table background
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..utils.inspection_config import InspectionConfig


@dataclass
class InteriorExtractionResult:
    interior_valid: bool
    reason: str
    interior_mask: np.ndarray
    inset_mask: np.ndarray
    excluded_label_mask: np.ndarray
    interior_area_px: int = 0
    interior_area_ratio: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class InteriorExtractor:
    """Extracts the inspectable packet interior depth region."""

    def __init__(self, config: InspectionConfig) -> None:
        self.config = config
        self.pkt_cfg = config.packet_inspection

    def extract(
        self,
        frame: np.ndarray,
        packet_mask: np.ndarray,
        hand_mask: Optional[np.ndarray] = None,
    ) -> InteriorExtractionResult:
        h, w = frame.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)
        if packet_mask is None or not np.any(packet_mask > 0):
            return InteriorExtractionResult(
                interior_valid=False,
                reason="No packet mask available for interior extraction.",
                interior_mask=empty,
                inset_mask=empty,
                excluded_label_mask=empty,
            )

        pkt_bin = (packet_mask > 0).astype(np.uint8) * 255
        packet_area = int(np.count_nonzero(pkt_bin))

        # 1. Inset inward using distance transform to strip borders & heat-sealed edges cleanly.
        dist = cv2.distanceTransform(pkt_bin, cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        inset_px = max(
            float(self.pkt_cfg.min_border_inset_px),
            max_dist * 2.0 * float(self.pkt_cfg.border_inset_ratio),
        )
        # Prevent completely erasing narrow packet openings while still excluding rims.
        inset_px = min(inset_px, max(2.0, max_dist * 0.45))

        inset_mask = np.zeros((h, w), dtype=np.uint8)
        inset_mask[dist >= inset_px] = 255

        # 2. Strictly subtract any hand pixels (with small safety dilation around fingers).
        if hand_mask is not None and np.any(hand_mask > 0):
            hand_bin = (hand_mask > 0).astype(np.uint8) * 255
            hand_dilated = cv2.dilate(hand_bin, np.ones((7, 7), np.uint8), iterations=1)
            inset_mask[hand_dilated > 0] = 0

        inset_area = int(np.count_nonzero(inset_mask))
        if inset_area < self.pkt_cfg.min_interior_area_px:
            return InteriorExtractionResult(
                interior_valid=False,
                reason=(
                    f"Usable packet interior too small after border/hand exclusion "
                    f"({inset_area}px < {self.pkt_cfg.min_interior_area_px}px)."
                ),
                interior_mask=empty,
                inset_mask=inset_mask,
                excluded_label_mask=empty,
                interior_area_px=inset_area,
                interior_area_ratio=float(inset_area) / float(max(packet_area, 1)),
            )

        # 3. Identify and exclude bright monochrome printed exterior label/form regions
        # (bright white paper + dark printed grid lines / text) while preserving
        # shadowed depth cavities and any warm metallic candidates inside the opening.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Local monochrome edge density catches printed form grids & black text on white plastic.
        edges = cv2.Canny(gray, 55, 145)
        edge_density = cv2.boxFilter(
            (edges > 0).astype(np.float32),
            ddepth=-1,
            ksize=(15, 15),
            normalize=True,
        )
        local_val_mean = cv2.boxFilter(val.astype(np.float32), ddepth=-1, ksize=(15, 15), normalize=True)
        local_sat_mean = cv2.boxFilter(sat.astype(np.float32), ddepth=-1, ksize=(15, 15), normalize=True)

        # Printed label regions are bright white/light-gray locally with low color saturation
        # and printed dark table/text lines.
        printed_label = (
            (inset_mask > 0)
            & (local_val_mean >= 175.0)
            & (local_sat_mean <= 28.0)
            & (edge_density >= 0.08)
        )
        excluded_label_mask = np.zeros((h, w), dtype=np.uint8)
        excluded_label_mask[printed_label] = 255
        if np.any(excluded_label_mask > 0):
            excluded_label_mask = cv2.morphologyEx(
                excluded_label_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
            )

        interior_mask = inset_mask.copy()
        interior_mask[excluded_label_mask > 0] = 0

        # Morphological cleanup to remove isolated speckles while preserving interior opening.
        kernel = np.ones((5, 5), np.uint8)
        interior_mask = cv2.morphologyEx(interior_mask, cv2.MORPH_OPEN, kernel)
        interior_mask = cv2.morphologyEx(interior_mask, cv2.MORPH_CLOSE, kernel)

        # Ensure strict subset of inset_mask (never leaks outside packet or onto hands).
        interior_mask[inset_mask == 0] = 0

        interior_area = int(np.count_nonzero(interior_mask))
        interior_ratio = float(interior_area) / float(max(packet_area, 1))

        # Check if printed form covers almost the entire packet (e.g., flat exterior packet face, not open cavity).
        label_ratio = float(np.count_nonzero(excluded_label_mask)) / float(max(inset_area, 1))

        diagnostics = {
            "packet_area_px": packet_area,
            "inset_px": round(inset_px, 2),
            "inset_area_px": inset_area,
            "excluded_label_ratio": round(label_ratio, 4),
            "interior_area_px": interior_area,
            "interior_area_ratio": round(interior_ratio, 4),
        }

        if interior_area < self.pkt_cfg.min_interior_area_px:
            return InteriorExtractionResult(
                interior_valid=False,
                reason=(
                    f"Packet interior cavity not clearly visible ({interior_area}px; "
                    f"printed label coverage {label_ratio:.0%})."
                ),
                interior_mask=interior_mask,
                inset_mask=inset_mask,
                excluded_label_mask=excluded_label_mask,
                interior_area_px=interior_area,
                interior_area_ratio=interior_ratio,
                diagnostics=diagnostics,
            )

        if interior_ratio < self.pkt_cfg.min_interior_area_ratio:
            return InteriorExtractionResult(
                interior_valid=False,
                reason=(
                    f"Visible interior ratio too low ({interior_ratio:.0%} < "
                    f"{self.pkt_cfg.min_interior_area_ratio:.0%})."
                ),
                interior_mask=interior_mask,
                inset_mask=inset_mask,
                excluded_label_mask=excluded_label_mask,
                interior_area_px=interior_area,
                interior_area_ratio=interior_ratio,
                diagnostics=diagnostics,
            )

        return InteriorExtractionResult(
            interior_valid=True,
            reason="Packet interior region extracted.",
            interior_mask=interior_mask,
            inset_mask=inset_mask,
            excluded_label_mask=excluded_label_mask,
            interior_area_px=interior_area,
            interior_area_ratio=interior_ratio,
            diagnostics=diagnostics,
        )

