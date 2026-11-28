"""Gold visibility detection inside a packet interior (Stage 1).

Combines:
1. Primary signal: existing `jeweltrack.onnx` gold segmentation model run on
   the full frame and/or padded packet ROI crop, strictly masked to the packet
   interior depth region.
2. Supporting OpenCV analysis: multi-color-space (HSV + CIE LAB) verification,
   specular glare rejection, and metallic texture/gradient verification so
   bright plastic reflections or yellow lighting casts are NOT misclassified
   as gold jewellery.

Never relies solely on simple HSV thresholding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils.image_utils import RawSegInstance, extract_yolo_instances, union_masks
from ..utils.inspection_config import InspectionConfig
from ..utils.model_loader import InspectionModelRegistry


@dataclass
class GoldVisibilityResult:
    decision: str  # GOLD_VISIBLE | NO_GOLD | UNCERTAIN
    gold_visible: Optional[bool]
    confidence: float
    reason: str
    gold_mask: np.ndarray
    glare_mask: np.ndarray
    gold_area_px: int = 0
    model_confidence: float = 0.0
    model_gold_area_px: int = 0
    cv_support_area_px: int = 0
    glare_rejected: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class GoldVisibilityDetector:
    """Detects whether gold jewellery is visible inside the extracted packet interior."""

    def __init__(
        self,
        config: InspectionConfig,
        model_registry: Optional[InspectionModelRegistry] = None,
    ) -> None:
        self.config = config
        self.pkt_cfg = config.packet_inspection
        self.registry = model_registry

    def _run_jewel_model(
        self,
        frame: np.ndarray,
        padded_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[List[RawSegInstance], bool]:
        """Run `jeweltrack.onnx` on full frame and optionally zoomed packet ROI crop."""
        if self.registry is None or not self.registry.has("jewel"):
            return [], False

        model = self.registry.get("jewel")
        class_ids = self.registry.get_class_ids("jewel")
        min_conf = min(self.pkt_cfg.jewel_conf, self.pkt_cfg.jewel_borderline_conf)

        instances: List[RawSegInstance] = []
        h, w = frame.shape[:2]

        # Full-frame inference
        res_full = model.predict(
            source=frame,
            imgsz=self.config.models.imgsz,
            conf=min_conf,
            device=self.registry.device,
            verbose=False,
        )[0]
        instances.extend(
            extract_yolo_instances(
                res_full,
                (h, w),
                selected_ids=class_ids,
                min_conf=min_conf,
                min_area_px=30,
            )
        )

        # Zoomed packet ROI crop inference (improves recall on small gold inside deep packet)
        if padded_roi is not None:
            px1, py1, px2, py2 = padded_roi
            crop = frame[py1:py2, px1:px2]
            if crop.size > 0 and crop.shape[0] >= 32 and crop.shape[1] >= 32:
                res_crop = model.predict(
                    source=crop,
                    imgsz=self.config.models.imgsz,
                    conf=min_conf,
                    device=self.registry.device,
                    verbose=False,
                )[0]
                instances.extend(
                    extract_yolo_instances(
                        res_crop,
                        crop.shape[:2],
                        selected_ids=class_ids,
                        min_conf=min_conf,
                        min_area_px=25,
                        offset_xy=(px1, py1),
                        full_shape=(h, w),
                    )
                )

        return instances, True

    def _analyze_cv_gold_and_glare(
        self,
        frame: np.ndarray,
        interior_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Multi-space HSV+LAB chromatic, specular-glare, and metallic-texture analysis."""
        h, w = frame.shape[:2]
        interior_bool = interior_mask > 0
        if not np.any(interior_bool):
            empty = np.zeros((h, w), dtype=np.uint8)
            return {
                "cv_gold_mask": empty,
                "glare_mask": empty,
                "cv_textured_area_px": 0,
                "cv_raw_warm_area_px": 0,
                "glare_area_px": 0,
                "max_component_texture_std": 0.0,
                "max_component_grad_mean": 0.0,
                "reflection_only": False,
            }

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        L_ch, A_ch, B_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Specular white / blown-out plastic glare mask
        glare_bool = interior_bool & ((V >= 242) & (S <= 48))
        glare_mask = np.zeros((h, w), dtype=np.uint8)
        glare_mask[glare_bool] = 255

        # Multi-space warm metallic gold chroma candidate (excluding specular washout)
        warm_hsv = (H >= 12) & (H <= 38) & (S >= 65) & (V >= 45) & (V <= 245)
        warm_lab = (B_ch >= 148) & (A_ch >= 118) & (A_ch <= 156) & (L_ch >= 42)
        warm_candidate = interior_bool & warm_hsv & warm_lab & (~glare_bool)

        raw_warm_mask = np.zeros((h, w), dtype=np.uint8)
        raw_warm_mask[warm_candidate] = 255
        raw_warm_mask = cv2.morphologyEx(raw_warm_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        raw_warm_mask = cv2.morphologyEx(raw_warm_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        raw_warm_mask[~interior_bool] = 0

        # Compute gradient magnitude for metallic texture verification (rejects smooth yellow wall/lamp cast)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_warm_mask, connectivity=8)
        textured_gold_mask = np.zeros((h, w), dtype=np.uint8)
        max_tex_std = 0.0
        max_grad_mean = 0.0
        rejected_flat_area = 0

        for lbl in range(1, num_labels):
            comp_area = int(stats[lbl, cv2.CC_STAT_AREA])
            if comp_area < 40:
                continue
            comp_bool = labels == lbl
            l_std = float(np.std(L_ch[comp_bool]))
            g_mean = float(np.mean(grad_mag[comp_bool]))
            max_tex_std = max(max_tex_std, l_std)
            max_grad_mean = max(max_grad_mean, g_mean)

            # Real gold ornaments have specular-to-shadow luminance variation & edge structure.
            # Smooth uniform yellow reflections on plastic have very low std and gradient.
            if l_std >= self.pkt_cfg.min_gold_texture_std and g_mean >= 14.0:
                textured_gold_mask[comp_bool] = 255
            else:
                rejected_flat_area += comp_area

        raw_warm_area = int(np.count_nonzero(raw_warm_mask))
        textured_area = int(np.count_nonzero(textured_gold_mask))
        glare_area = int(np.count_nonzero(glare_mask))
        reflection_only = (textured_area == 0) and (rejected_flat_area >= 60 or glare_area >= 120)

        return {
            "cv_gold_mask": textured_gold_mask,
            "glare_mask": glare_mask,
            "cv_textured_area_px": textured_area,
            "cv_raw_warm_area_px": raw_warm_area,
            "glare_area_px": glare_area,
            "rejected_flat_area_px": rejected_flat_area,
            "max_component_texture_std": round(max_tex_std, 2),
            "max_component_grad_mean": round(max_grad_mean, 2),
            "reflection_only": reflection_only,
        }

    def _check_candidate_glare_and_texture(
        self,
        frame: np.ndarray,
        candidate_mask: np.ndarray,
        glare_mask: np.ndarray,
    ) -> Tuple[bool, float, float]:
        """Verify a model-detected candidate mask is not purely specular glare or flat reflection."""
        cand_bool = candidate_mask > 0
        cand_area = int(np.count_nonzero(cand_bool))
        if cand_area == 0:
            return False, 1.0, 0.0

        glare_overlap = int(np.count_nonzero(cand_bool & (glare_mask > 0)))
        glare_ratio = float(glare_overlap) / float(cand_area)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_std = float(np.std(lab[:, :, 0][cand_bool]))

        is_glare_or_flat = (
            glare_ratio >= self.pkt_cfg.max_specular_glare_ratio
            or (l_std < (self.pkt_cfg.min_gold_texture_std * 0.5) and glare_ratio > 0.35)
        )
        return (not is_glare_or_flat), glare_ratio, l_std

    def detect(
        self,
        frame: np.ndarray,
        interior_mask: np.ndarray,
        padded_roi: Optional[Tuple[int, int, int, int]] = None,
        jewel_instances_override: Optional[List[RawSegInstance]] = None,
    ) -> GoldVisibilityResult:
        """Evaluate whether gold jewellery is visible inside `interior_mask`."""
        h, w = frame.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)

        if interior_mask is None or not np.any(interior_mask > 0):
            return GoldVisibilityResult(
                decision="UNCERTAIN",
                gold_visible=None,
                confidence=0.0,
                reason="Invalid interior mask.",
                gold_mask=empty,
                glare_mask=empty,
            )

        cv_info = self._analyze_cv_gold_and_glare(frame, interior_mask)
        glare_mask = cv_info["glare_mask"]

        if jewel_instances_override is not None:
            raw_instances = jewel_instances_override
            model_available = True
        else:
            raw_instances, model_available = self._run_jewel_model(frame, padded_roi=padded_roi)

        # Intersect model instances strictly with interior_mask.
        interior_bool = interior_mask > 0
        valid_model_masks: List[np.ndarray] = []
        best_model_conf = 0.0
        best_overlap_ratio = 0.0
        glare_rejected = False

        for inst in raw_instances:
            inside_mask = np.zeros((h, w), dtype=np.uint8)
            inside_bool = (inst.mask > 0) & interior_bool
            inside_area = int(np.count_nonzero(inside_bool))
            if inside_area < self.pkt_cfg.min_gold_area_px:
                continue
            overlap_ratio = float(inside_area) / float(max(inst.area, 1))
            if overlap_ratio < self.pkt_cfg.min_gold_interior_overlap:
                continue

            inside_mask[inside_bool] = 255
            passes_check, g_ratio, _ = self._check_candidate_glare_and_texture(frame, inside_mask, glare_mask)
            if not passes_check:
                glare_rejected = True
                continue

            valid_model_masks.append(inside_mask)
            if inst.confidence > best_model_conf:
                best_model_conf = float(inst.confidence)
                best_overlap_ratio = overlap_ratio

        model_gold_mask = union_masks(valid_model_masks, (h, w))
        model_gold_area = int(np.count_nonzero(model_gold_mask))
        cv_gold_area = int(cv_info["cv_textured_area_px"])

        diagnostics = {
            "model_available": model_available,
            "raw_model_instances": len(raw_instances),
            "model_gold_area_px": model_gold_area,
            "best_model_conf": round(best_model_conf, 4),
            "best_overlap_ratio": round(best_overlap_ratio, 4),
            "cv_textured_area_px": cv_gold_area,
            "cv_raw_warm_area_px": int(cv_info["cv_raw_warm_area_px"]),
            "glare_area_px": int(cv_info["glare_area_px"]),
            "reflection_only": bool(cv_info["reflection_only"]),
            "glare_rejected": glare_rejected,
            "max_texture_std": cv_info["max_component_texture_std"],
            "max_grad_mean": cv_info["max_component_grad_mean"],
        }

        # Case 1: Strong deep-learning gold detection inside valid packet interior
        if model_gold_area >= self.pkt_cfg.min_gold_area_px and best_model_conf >= self.pkt_cfg.jewel_conf:
            return GoldVisibilityResult(
                decision="GOLD_VISIBLE",
                gold_visible=True,
                confidence=best_model_conf,
                reason=(
                    f"Gold jewellery detected inside packet interior "
                    f"(model conf {best_model_conf:.2f}, area {model_gold_area}px)."
                ),
                gold_mask=model_gold_mask,
                glare_mask=glare_mask,
                gold_area_px=model_gold_area,
                model_confidence=best_model_conf,
                model_gold_area_px=model_gold_area,
                cv_support_area_px=cv_gold_area,
                glare_rejected=glare_rejected,
                diagnostics=diagnostics,
            )

        # Case 2: Borderline model confidence corroborated by textured non-glare CV gold evidence
        if (
            model_gold_area >= self.pkt_cfg.min_gold_area_px
            and best_model_conf >= self.pkt_cfg.jewel_borderline_conf
            and cv_gold_area >= self.pkt_cfg.min_gold_area_px
        ):
            combined_mask = union_masks([model_gold_mask, cv_info["cv_gold_mask"]], (h, w))
            fused_conf = min(0.85, best_model_conf + 0.15)
            return GoldVisibilityResult(
                decision="GOLD_VISIBLE",
                gold_visible=True,
                confidence=fused_conf,
                reason=(
                    f"Gold jewellery corroborated by model ({best_model_conf:.2f}) "
                    f"and metallic texture analysis ({cv_gold_area}px)."
                ),
                gold_mask=combined_mask,
                glare_mask=glare_mask,
                gold_area_px=int(np.count_nonzero(combined_mask)),
                model_confidence=best_model_conf,
                model_gold_area_px=model_gold_area,
                cv_support_area_px=cv_gold_area,
                glare_rejected=glare_rejected,
                diagnostics=diagnostics,
            )

        # Case 3: Model unavailable -> never pretend high certainty from HSV alone; flag UNCERTAIN
        if not model_available:
            return GoldVisibilityResult(
                decision="UNCERTAIN",
                gold_visible=None,
                confidence=0.0,
                reason="Jewellery segmentation model unavailable; cannot confirm gold visibility reliably.",
                gold_mask=cv_info["cv_gold_mask"],
                glare_mask=glare_mask,
                gold_area_px=cv_gold_area,
                model_confidence=0.0,
                model_gold_area_px=0,
                cv_support_area_px=cv_gold_area,
                glare_rejected=glare_rejected,
                diagnostics=diagnostics,
            )

        # Case 4: Model saw nothing, but strong textured metallic-warm signal exists -> UNCERTAIN
        # (protects against false CLOSED when gold is partially shadowed or low-conf)
        if cv_gold_area >= self.pkt_cfg.cv_gold_uncertainty_area_px:
            return GoldVisibilityResult(
                decision="UNCERTAIN",
                gold_visible=None,
                confidence=0.35,
                reason=(
                    f"Ambiguous metallic-warm region inside packet ({cv_gold_area}px) "
                    "not confirmed by jewellery model; hold steady or adjust angle."
                ),
                gold_mask=cv_info["cv_gold_mask"],
                glare_mask=glare_mask,
                gold_area_px=cv_gold_area,
                model_confidence=best_model_conf,
                model_gold_area_px=model_gold_area,
                cv_support_area_px=cv_gold_area,
                glare_rejected=glare_rejected,
                diagnostics=diagnostics,
            )

        # Case 5: Clean negative (reflection-only glare rejected, no gold detected inside valid interior)
        reason = "No gold jewellery visible inside packet interior."
        if glare_rejected or cv_info["reflection_only"]:
            reason = "No gold jewellery visible (surface glare/reflection rejected)."

        return GoldVisibilityResult(
            decision="NO_GOLD",
            gold_visible=False,
            confidence=0.85 if not (glare_rejected or cv_info["reflection_only"]) else 0.75,
            reason=reason,
            gold_mask=empty,
            glare_mask=glare_mask,
            gold_area_px=0,
            model_confidence=best_model_conf,
            model_gold_area_px=0,
            cv_support_area_px=cv_gold_area,
            glare_rejected=glare_rejected or bool(cv_info["reflection_only"]),
            diagnostics=diagnostics,
        )

