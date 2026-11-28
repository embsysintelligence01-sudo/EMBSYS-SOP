"""Image, mask, lighting, blur, and visualization utilities for Post-Cutting Inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class FrameQualityMetrics:
    blur_variance: float
    mean_brightness: float
    std_brightness: float
    is_blurry: bool
    is_too_dark: bool
    is_overexposed: bool

    @property
    def is_acceptable(self) -> bool:
        return not (self.is_blurry or self.is_too_dark or self.is_overexposed)


@dataclass
class RawSegInstance:
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    area: int
    centroid: Tuple[float, float]


def unpad_letterbox_mask(
    m_raw: np.ndarray,
    orig_shape: Tuple[int, int],
    input_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Invert YOLO letterbox padding and map raw mask back to (orig_h, orig_w)."""
    orig_h, orig_w = orig_shape[:2]
    inp_h, inp_w = m_raw.shape[:2] if input_shape is None else input_shape[:2]

    ratio = min(float(inp_h) / float(max(orig_h, 1)), float(inp_w) / float(max(orig_w, 1)))
    new_unpad_w = int(round(orig_w * ratio))
    new_unpad_h = int(round(orig_h * ratio))

    dw = (inp_w - new_unpad_w) / 2.0
    dh = (inp_h - new_unpad_h) / 2.0

    top = max(0, int(round(dh - 0.1)))
    bottom = max(0, int(round(dh + 0.1)))
    left = max(0, int(round(dw - 0.1)))
    right = max(0, int(round(dw + 0.1)))

    crop_h_end = max(top + 1, inp_h - bottom)
    crop_w_end = max(left + 1, inp_w - right)
    m_cropped = m_raw[top:crop_h_end, left:crop_w_end]
    return cv2.resize(
        m_cropped.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )


def evaluate_frame_quality(
    frame: np.ndarray,
    min_blur_variance: float = 25.0,
    min_mean_brightness: float = 38.0,
    max_mean_brightness: float = 242.0,
    roi_mask: Optional[np.ndarray] = None,
) -> FrameQualityMetrics:
    """Measure focus sharpness and lighting quality on full frame or within ROI."""
    if frame is None or frame.size == 0:
        return FrameQualityMetrics(0.0, 0.0, 0.0, True, True, False)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv_v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]

    if roi_mask is not None and np.any(roi_mask > 0):
        pixels = hsv_v[roi_mask > 0]
        mean_v = float(np.mean(pixels))
        std_v = float(np.std(pixels))
        ys, xs = np.where(roi_mask > 0)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        lap_region = gray[y1:y2, x1:x2]
        blur_var = float(cv2.Laplacian(lap_region, cv2.CV_64F).var()) if lap_region.size > 0 else 0.0
    else:
        mean_v = float(np.mean(hsv_v))
        std_v = float(np.std(hsv_v))
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return FrameQualityMetrics(
        blur_variance=blur_var,
        mean_brightness=mean_v,
        std_brightness=std_v,
        is_blurry=blur_var < min_blur_variance,
        is_too_dark=mean_v < min_mean_brightness,
        is_overexposed=(mean_v > max_mean_brightness and std_v < 28.0),
    )


def mask_centroid_and_bbox(mask: np.ndarray) -> Tuple[Tuple[float, float], Tuple[int, int, int, int], int]:
    """Compute (cx, cy), (x1, y1, x2, y2), and pixel area for a binary mask."""
    ys, xs = np.where(mask > 0)
    area = int(xs.size)
    if area == 0:
        return (0.0, 0.0), (0, 0, 0, 0), 0
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return (cx, cy), bbox, area


def extract_yolo_instances(
    result: Any,
    frame_shape: Tuple[int, int],
    selected_ids: Optional[Sequence[int]] = None,
    min_conf: float = 0.0,
    min_area_px: int = 1,
    offset_xy: Tuple[int, int] = (0, 0),
    full_shape: Optional[Tuple[int, int]] = None,
) -> List[RawSegInstance]:
    """Extract individual segmentation instances from an Ultralytics result.

    Supports mapping crop-space predictions back into `full_shape` via `offset_xy`.
    """
    crop_h, crop_w = frame_shape[:2]
    out_h, out_w = full_shape[:2] if full_shape is not None else (crop_h, crop_w)
    ox, oy = offset_xy

    if result is None or getattr(result, "boxes", None) is None or len(result.boxes) == 0:
        return []

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = (
        result.boxes.cls.detach().cpu().numpy().astype(int)
        if result.boxes.cls is not None
        else np.zeros(len(boxes_xyxy), dtype=int)
    )

    masks_obj = getattr(result, "masks", None)
    has_polys = masks_obj is not None and getattr(masks_obj, "xy", None) is not None
    has_data = masks_obj is not None and getattr(masks_obj, "data", None) is not None
    mask_data = masks_obj.data.detach().cpu().numpy() if has_data else None

    instances: List[RawSegInstance] = []
    for i, (box, conf, cls_id) in enumerate(zip(boxes_xyxy, confs, classes)):
        if float(conf) < min_conf:
            continue
        if selected_ids is not None and int(cls_id) not in selected_ids:
            continue

        crop_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        if has_polys and i < len(masks_obj.xy) and len(masks_obj.xy[i]) >= 3:
            poly = np.round(masks_obj.xy[i]).astype(np.int32)
            cv2.fillPoly(crop_mask, [poly], 255)
        elif mask_data is not None and i < len(mask_data):
            m_unpadded = unpad_letterbox_mask(mask_data[i], (crop_h, crop_w))
            crop_mask[m_unpadded >= 0.5] = 255
        else:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(crop_w, x2), min(crop_h, y2)
            if x2 > x1 and y2 > y1:
                crop_mask[y1:y2, x1:x2] = 255

        if full_shape is not None and (ox != 0 or oy != 0 or (out_h, out_w) != (crop_h, crop_w)):
            full_mask = np.zeros((out_h, out_w), dtype=np.uint8)
            y_end = min(out_h, oy + crop_h)
            x_end = min(out_w, ox + crop_w)
            sub_h = max(0, y_end - oy)
            sub_w = max(0, x_end - ox)
            if sub_h > 0 and sub_w > 0:
                full_mask[oy:y_end, ox:x_end] = crop_mask[:sub_h, :sub_w]
            final_mask = full_mask
        else:
            final_mask = crop_mask

        centroid, bbox, area = mask_centroid_and_bbox(final_mask)
        if area < min_area_px:
            continue

        instances.append(
            RawSegInstance(
                mask=final_mask,
                bbox=bbox,
                confidence=float(conf),
                class_id=int(cls_id),
                area=area,
                centroid=centroid,
            )
        )

    return instances


def union_masks(masks: Sequence[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        if m is not None and m.shape[:2] == (h, w):
            out[m > 0] = 255
    return out


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = float(area_a + area_b - inter)
    return float(inter) / denom if denom > 0 else 0.0


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = int(np.count_nonzero(a & b))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return float(inter) / float(union) if union > 0 else 0.0


def clamp_roi(roi: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = roi
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return (x1, y1, x2, y2)


def draw_mask_overlay(
    frame: np.ndarray,
    mask: Optional[np.ndarray],
    color: Tuple[int, int, int],
    alpha: float = 0.30,
    draw_contour: bool = True,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Alpha-blend a binary mask onto a copy of `frame`."""
    out = frame.copy()
    if mask is None or not np.any(mask > 0):
        return out
    overlay = out.copy()
    overlay[mask > 0] = color
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)
    if draw_contour:
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(out, contours, -1, color, contour_thickness)
    return out


def distinct_bgr_color(index: int) -> Tuple[int, int, int]:
    palette = [
        (40, 220, 120),
        (255, 190, 50),
        (80, 180, 255),
        (220, 110, 255),
        (60, 230, 230),
        (110, 140, 255),
        (170, 240, 80),
        (255, 130, 170),
    ]
    return palette[index % len(palette)]

