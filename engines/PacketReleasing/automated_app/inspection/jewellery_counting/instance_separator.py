"""Instance separation for touching and overlapping jewellery articles (Stage 2).

Supports:
1. Optional SAM2.1 / MobileSAM prompt-based instance refinement when weights
   are configured and available on disk.
2. Deterministic OpenCV geometric separation (distance-transform peak analysis,
   hole-filled bangle/ring separation, outer convexity defect & saddle/neck
   analysis, and marker-controlled watershed).
3. Explicit uncertainty detection: whenever articles overlap or entangle such
   that individual separation cannot be guaranteed, flags `is_uncertain=True`
   so the system reports `COUNT_UNCERTAIN` instead of inventing a count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils.image_utils import RawSegInstance, bbox_iou, mask_centroid_and_bbox, mask_iou
from ..utils.inspection_config import InspectionConfig
from ..utils.model_loader import InspectionModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class SeparatedArticle:
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: int
    confidence: float
    separated_from_cluster: bool = False
    separation_method: str = "single"  # single | connected_component | watershed | sam | uncertain_cluster
    uncertain: bool = False
    uncertainty_reason: str = ""


@dataclass
class SeparationResult:
    articles: List[SeparatedArticle]
    is_uncertain: bool
    uncertainty_reasons: List[str] = field(default_factory=list)
    used_sam: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class InstanceSeparator:
    """Separates touching jewellery articles and flags ambiguous overlaps."""

    def __init__(
        self,
        config: InspectionConfig,
        model_registry: Optional[InspectionModelRegistry] = None,
    ) -> None:
        self.config = config
        self.jwl_cfg = config.jewellery_counting
        self.sep_cfg = config.jewellery_counting.separation
        self.registry = model_registry

    @staticmethod
    def _fill_internal_holes(mask: np.ndarray) -> Tuple[np.ndarray, int, List[int]]:
        """Fill enclosed holes inside a binary mask and count significant holes."""
        binary = (mask > 0).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        filled = binary.copy()
        hole_areas: List[int] = []

        if hierarchy is not None and len(contours) > 0:
            hier = hierarchy[0]
            for idx, cnt in enumerate(contours):
                # In RETR_CCOMP, parent >= 0 indicates an internal hole contour.
                if hier[idx][3] >= 0:
                    area = int(round(cv2.contourArea(cnt)))
                    if area >= 35:
                        hole_areas.append(area)
                        cv2.drawContours(filled, [cnt], -1, 255, thickness=-1)
        return filled, len(hole_areas), hole_areas

    @staticmethod
    def _outer_contour_metrics(mask: np.ndarray, deep_defect_px: float) -> Dict[str, Any]:
        binary = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"solidity": 1.0, "deep_defects": 0, "max_defect_px": 0.0, "perimeter": 0.0}

        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 1.0
        perimeter = float(cv2.arcLength(cnt, True))

        deep_defects = 0
        max_defect_px = 0.0
        if len(cnt) >= 5:
            hull_idx = cv2.convexHull(cnt, returnPoints=False)
            if hull_idx is not None and len(hull_idx) >= 3:
                try:
                    defects = cv2.convexityDefects(cnt, hull_idx)
                    if defects is not None:
                        for d in defects[:, 0, :]:
                            depth_px = float(d[3]) / 256.0
                            if depth_px > max_defect_px:
                                max_defect_px = depth_px
                            if depth_px >= deep_defect_px:
                                deep_defects += 1
                except cv2.error:
                    pass

        return {
            "solidity": solidity,
            "deep_defects": deep_defects,
            "max_defect_px": max_defect_px,
            "perimeter": perimeter,
        }

    def _find_distance_peaks(self, binary_mask: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
        """Find well-separated local maxima in the Euclidean distance transform."""
        dist = cv2.distanceTransform((binary_mask > 0).astype(np.uint8) * 255, cv2.DIST_L2, 5)
        max_val = float(dist.max())
        if max_val <= 2.0:
            return dist, []

        dist_smooth = cv2.GaussianBlur(dist, (5, 5), 0)
        ksize = max(3, int(self.sep_cfg.min_peak_distance_px) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.dilate(dist_smooth, kernel)

        thresh = max(3.5, max_val * float(self.sep_cfg.peak_rel_threshold))
        is_peak = (dist_smooth >= dilated - 1e-5) & (dist_smooth >= thresh) & (binary_mask > 0)

        num_lbl, _, _, centroids = cv2.connectedComponentsWithStats(is_peak.astype(np.uint8) * 255)
        raw_peaks: List[Tuple[int, int, float]] = []
        for lbl in range(1, num_lbl):
            cx, cy = int(round(centroids[lbl][0])), int(round(centroids[lbl][1]))
            cx = max(0, min(cx, binary_mask.shape[1] - 1))
            cy = max(0, min(cy, binary_mask.shape[0] - 1))
            raw_peaks.append((cx, cy, float(dist[cy, cx])))

        # Non-maximum suppression by peak height & minimum Euclidean distance
        raw_peaks.sort(key=lambda p: p[2], reverse=True)
        peaks: List[Tuple[int, int, float]] = []
        min_d = float(self.sep_cfg.min_peak_distance_px)
        for px, py, pv in raw_peaks:
            if all(np.hypot(px - qx, py - qy) >= min_d for qx, qy, _ in peaks):
                peaks.append((px, py, pv))
        return dist, peaks

    @staticmethod
    def _saddle_neck_ratio(
        dist_map: np.ndarray,
        p1: Tuple[int, int, float],
        p2: Tuple[int, int, float],
    ) -> float:
        """Compute minimum distance-transform value along the segment connecting two peaks,
        normalized by the smaller of the two peak heights.

        Low neck ratio (~0.0..0.55) => sharp pinch / touching articles.
        Intermediate neck ratio (~0.58..0.85) => thick overlap / ambiguous waist.
        High neck ratio (>0.85) => uniform elongated single body (e.g. single bar/chain).
        """
        x1, y1, v1 = p1
        x2, y2, v2 = p2
        steps = max(12, int(round(np.hypot(x2 - x1, y2 - y1) * 1.5)))
        xs = np.clip(np.round(np.linspace(x1, x2, steps)).astype(int), 0, dist_map.shape[1] - 1)
        ys = np.clip(np.round(np.linspace(y1, y2, steps)).astype(int), 0, dist_map.shape[0] - 1)
        line_vals = dist_map[ys, xs]
        min_val = float(np.min(line_vals))
        ref_height = max(min(v1, v2), 1e-5)
        return min_val / ref_height

    def _watershed_split(
        self,
        original_mask: np.ndarray,
        working_mask: np.ndarray,
        dist_map: np.ndarray,
        peaks: List[Tuple[int, int, float]],
    ) -> List[np.ndarray]:
        """Split a binary mask using true topographic watershed on `-dist_map`.

        Uses `skimage.segmentation.watershed` when available (exact flooding on
        inverted Euclidean distance topography) and falls back to geodesic
        multi-seed region growing inside `working_mask` using pure OpenCV.
        """
        h, w = original_mask.shape[:2]
        work_bool = working_mask > 0
        orig_bool = original_mask > 0

        markers = np.zeros((h, w), dtype=np.int32)
        for idx, (px, py, pv) in enumerate(peaks, start=1):
            seed_r = max(2, int(round(pv * 0.25)))
            cv2.circle(markers, (px, py), seed_r, idx, -1)
        markers[~work_bool] = 0

        try:
            from skimage.segmentation import watershed  # type: ignore

            labels = watershed(-dist_map, markers, mask=work_bool)
        except Exception:
            # Pure-OpenCV fallback: assign each pixel in working_mask to nearest peak
            # normalized by peak radius (power diagram / Voronoi partition).
            ys, xs = np.where(work_bool)
            labels = np.zeros((h, w), dtype=np.int32)
            if xs.size > 0:
                best_score = np.full(xs.size, np.inf, dtype=np.float32)
                best_lbl = np.zeros(xs.size, dtype=np.int32)
                for idx, (px, py, pv) in enumerate(peaks, start=1):
                    d2 = ((xs - px).astype(np.float32) ** 2 + (ys - py).astype(np.float32) ** 2) / max(pv * pv, 1.0)
                    better = d2 < best_score
                    best_score[better] = d2[better]
                    best_lbl[better] = idx
                labels[ys, xs] = best_lbl

        sub_masks: List[np.ndarray] = []
        for idx in range(1, len(peaks) + 1):
            region = (labels == idx) & orig_bool
            area = int(np.count_nonzero(region))
            if area >= self.jwl_cfg.min_article_area_px:
                m = np.zeros((h, w), dtype=np.uint8)
                m[region] = 255
                sub_masks.append(m)
        return sub_masks

    def _try_sam_separation(
        self,
        frame: np.ndarray,
        component_mask: np.ndarray,
        peaks: List[Tuple[int, int, float]],
    ) -> Optional[List[np.ndarray]]:
        """Optional prompt-based separation using SAM2.1 or MobileSAM if installed & configured."""
        if not self.sep_cfg.enable_sam or self.registry is None or not self.registry.has("sam"):
            return None
        if len(peaks) < 2:
            return None

        sam_model = self.registry.get("sam")
        h, w = frame.shape[:2]
        comp_bool = component_mask > 0
        sam_masks: List[np.ndarray] = []

        try:
            for px, py, _ in peaks:
                res = sam_model.predict(
                    source=frame,
                    points=[[px, py]],
                    labels=[1],
                    verbose=False,
                )[0]
                if res.masks is None or res.masks.data is None or len(res.masks.data) == 0:
                    continue
                raw = res.masks.data[0].detach().cpu().numpy()
                if raw.shape[:2] != (h, w):
                    raw = cv2.resize(raw.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                cand = (raw >= 0.5) & comp_bool
                if int(np.count_nonzero(cand)) >= self.jwl_cfg.min_article_area_px:
                    m = np.zeros((h, w), dtype=np.uint8)
                    m[cand] = 255
                    sam_masks.append(m)
        except Exception as exc:
            logger.warning("SAM separation fallback triggered due to error: %s", exc)
            return None

        if len(sam_masks) < 2:
            return None

        # Verify SAM masks are reasonably distinct (not duplicates of the same merged blob)
        distinct: List[np.ndarray] = []
        for cand in sam_masks:
            if all(mask_iou(cand, existing) < 0.65 for existing in distinct):
                distinct.append(cand)
        return distinct if len(distinct) >= 2 else None

    def _analyze_and_split_component(
        self,
        frame: np.ndarray,
        comp_mask: np.ndarray,
        confidence: float,
    ) -> Tuple[List[SeparatedArticle], bool, Optional[str], bool]:
        """Inspect one connected component: return single, cleanly split, or uncertain."""
        centroid, bbox, area = mask_centroid_and_bbox(comp_mask)

        filled_mask, hole_count, _ = self._fill_internal_holes(comp_mask)
        outer = self._outer_contour_metrics(filled_mask, self.sep_cfg.deep_defect_depth_px)
        solidity = float(outer["solidity"])
        deep_defects = int(outer["deep_defects"])

        # Case A: Overlapping rings/entangled loops create >= 3 enclosed holes inside one blob -> UNCERTAIN
        if hole_count >= 3:
            reason = (
                f"Overlapping/entangled jewellery cluster detected "
                f"({hole_count} enclosed loops in single mask); cannot reliably separate."
            )
            return (
                [
                    SeparatedArticle(
                        mask=comp_mask,
                        bbox=bbox,
                        centroid=centroid,
                        area=area,
                        confidence=confidence,
                        separated_from_cluster=False,
                        separation_method="uncertain_cluster",
                        uncertain=True,
                        uncertainty_reason=reason,
                    )
                ],
                True,
                reason,
                False,
            )

        # Case B: Choose working geometry — use hole-filled mask for rings/bangles (hole_count >= 1)
        # so touching bangles form clear lobe peaks at each ring center!
        working_mask = filled_mask if hole_count >= 1 else comp_mask
        dist_map, peaks = self._find_distance_peaks(working_mask)

        # Evaluate neck ratios only for peak pairs whose direct segment stays inside the mask
        # (pairs separated by exterior empty space are non-adjacent arms, e.g. in an X-crossing).
        neck_ratios: List[float] = []
        exterior_separated_pairs = 0
        if len(peaks) >= 2:
            for i in range(len(peaks)):
                for j in range(i + 1, len(peaks)):
                    nr = self._saddle_neck_ratio(dist_map, peaks[i], peaks[j])
                    if nr <= 1e-4:
                        exterior_separated_pairs += 1
                    else:
                        neck_ratios.append(nr)

        min_neck = min(neck_ratios) if neck_ratios else 1.0

        # Case C: Complex crossing/overlapping geometry (e.g., X-crossing chains or tangled pile)
        if deep_defects >= 3 and (solidity < 0.65 or min_neck > self.sep_cfg.clean_neck_ratio_max):
            reason = (
                f"Overlapping/crossing jewellery detected ({deep_defects} deep concavities, "
                f"solidity {solidity:.2f}, neck ratio {min_neck:.2f}); manual separation required."
            )
            return (
                [
                    SeparatedArticle(
                        mask=comp_mask,
                        bbox=bbox,
                        centroid=centroid,
                        area=area,
                        confidence=confidence,
                        separated_from_cluster=False,
                        separation_method="uncertain_cluster",
                        uncertain=True,
                        uncertainty_reason=reason,
                    )
                ],
                True,
                reason,
                False,
            )

        # Case D: Multi-peak candidate (touching OR overlapping)
        if len(peaks) >= 2:
            # Clean tangent touch: narrow neck AND pinched outer contour (or 2 distinct holes)
            has_pinch_geometry = (
                min_neck <= self.sep_cfg.clean_neck_ratio_max
                and (deep_defects >= 1 or solidity <= self.sep_cfg.touching_solidity_max or hole_count == 2)
            )
            if has_pinch_geometry:
                # Try optional SAM prompt separation first if configured & loaded
                sam_splits = self._try_sam_separation(frame, comp_mask, peaks)
                if sam_splits is not None:
                    articles = []
                    for sm in sam_splits:
                        c, b, a = mask_centroid_and_bbox(sm)
                        articles.append(
                            SeparatedArticle(
                                mask=sm,
                                bbox=b,
                                centroid=c,
                                area=a,
                                confidence=confidence,
                                separated_from_cluster=True,
                                separation_method="sam",
                                uncertain=False,
                            )
                        )
                    return articles, False, None, True

                # Deterministic OpenCV Marker-Controlled Watershed separation
                ws_splits = self._watershed_split(comp_mask, working_mask, dist_map, peaks)
                if len(ws_splits) >= 2:
                    articles = []
                    for wm in ws_splits:
                        c, b, a = mask_centroid_and_bbox(wm)
                        articles.append(
                            SeparatedArticle(
                                mask=wm,
                                bbox=b,
                                centroid=c,
                                area=a,
                                confidence=confidence,
                                separated_from_cluster=True,
                                separation_method="watershed",
                                uncertain=False,
                            )
                        )
                    return articles, False, None, False

            # Ambiguous overlap zone: multiple peaks with wide saddle (not a clean point-touch)
            # AND pinched/non-convex outer shape => overlapping articles -> COUNT_UNCERTAIN!
            is_ambiguous_overlap = (
                min_neck <= self.sep_cfg.ambiguous_neck_ratio_max
                and (deep_defects >= 2 or solidity <= self.sep_cfg.touching_solidity_max or hole_count >= 2)
            )
            if is_ambiguous_overlap:
                reason = (
                    f"Overlapping jewellery articles detected ({len(peaks)} peaks, "
                    f"thick overlap bridge {min_neck:.2f}, solidity {solidity:.2f}); "
                    "cannot separate unambiguously."
                )
                return (
                    [
                        SeparatedArticle(
                            mask=comp_mask,
                            bbox=bbox,
                            centroid=centroid,
                            area=area,
                            confidence=confidence,
                            separated_from_cluster=False,
                            separation_method="uncertain_cluster",
                            uncertain=True,
                            uncertainty_reason=reason,
                        )
                    ],
                    True,
                    reason,
                    False,
                )

        # Case E: Clean single article
        return (
            [
                SeparatedArticle(
                    mask=comp_mask,
                    bbox=bbox,
                    centroid=centroid,
                    area=area,
                    confidence=confidence,
                    separated_from_cluster=False,
                    separation_method="single",
                    uncertain=False,
                )
            ],
            False,
            None,
            False,
        )

    def separate(
        self,
        frame: np.ndarray,
        detections: List[RawSegInstance],
    ) -> SeparationResult:
        """Separate raw testbed jewellery detections into individual articles."""
        if not detections:
            return SeparationResult(articles=[], is_uncertain=False)

        h, w = frame.shape[:2]
        uncertainty_reasons: List[str] = []
        used_sam = False

        # Check for overlapping raw YOLO detections (e.g., stacked/overlapping jewellery masks)
        overlapping_indices: set[int] = set()
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                m_iou = mask_iou(detections[i].mask, detections[j].mask)
                b_iou = bbox_iou(detections[i].bbox, detections[j].bbox)
                inter_px = int(np.count_nonzero((detections[i].mask > 0) & (detections[j].mask > 0)))
                min_area = max(1, min(detections[i].area, detections[j].area))
                containment = float(inter_px) / float(min_area)
                if m_iou >= 0.18 or containment >= 0.25 or (b_iou >= self.sep_cfg.overlap_box_iou_threshold and inter_px > 40):
                    overlapping_indices.add(i)
                    overlapping_indices.add(j)
                    uncertainty_reasons.append(
                        f"Overlapping detector masks between detections #{i + 1} and #{j + 1} "
                        f"(mask IoU {m_iou:.2f}, overlap {containment:.0%})."
                    )

        articles: List[SeparatedArticle] = []
        for idx, det in enumerate(detections):
            # Split raw detection into connected components first
            binary = (det.mask > 0).astype(np.uint8) * 255
            num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            valid_comps: List[np.ndarray] = []
            for lbl in range(1, num_lbl):
                if int(stats[lbl, cv2.CC_STAT_AREA]) >= self.jwl_cfg.min_article_area_px:
                    cm = np.zeros((h, w), dtype=np.uint8)
                    cm[labels == lbl] = 255
                    valid_comps.append(cm)

            for cm in valid_comps:
                comp_articles, comp_uncertain, comp_reason, comp_sam = self._analyze_and_split_component(
                    frame, cm, det.confidence
                )
                used_sam = used_sam or comp_sam
                if idx in overlapping_indices:
                    for art in comp_articles:
                        art.uncertain = True
                        if not art.uncertainty_reason:
                            art.uncertainty_reason = "Overlaps with another detected jewellery mask."
                if comp_uncertain and comp_reason:
                    uncertainty_reasons.append(comp_reason)
                articles.extend(comp_articles)

        is_uncertain = bool(uncertainty_reasons) or any(a.uncertain for a in articles)
        return SeparationResult(
            articles=articles,
            is_uncertain=is_uncertain,
            uncertainty_reasons=uncertainty_reasons,
            used_sam=used_sam,
            diagnostics={
                "input_detections": len(detections),
                "output_articles": len(articles),
                "split_clusters": sum(1 for a in articles if a.separated_from_cluster),
                "uncertain_articles": sum(1 for a in articles if a.uncertain),
            },
        )
