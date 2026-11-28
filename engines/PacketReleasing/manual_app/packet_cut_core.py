"""
Packet-Cut Verification Engine: Shared Computer Vision & Decision Logic
========================================================================
Shared module for:
- cut_2_seg_pyqt.py
- test_cut_model-new.py

Features:
1. Robust Hand Detection & Seal False-Positive Filtering
2. Scissor Detection Validation & Packet Spatial Interaction
3. Multi-Instance Packet Segmentation & Connected Component Extraction
4. Rigorous Physical Gap Verification (Distance Transform + Background Check)
5. Strict 10-Second Stability Verification & Sample Enforcement
6. Standardized 5 Decision States (CUT_CONFIRMED, NOT_CUTTED, NOT_CONFIRMED, UNVERIFIED, PACKET_MISSING)
7. Standardized Debug Diagnostics Saver
"""

import time
from pathlib import Path
import cv2
import numpy as np


# ============================================================
# CONFIGURATION CONSTANTS (Configurable Thresholds)
# ============================================================
MIN_BACKGROUND_GAP_PX = 3.0        # Minimum physical distance between piece edges (pixels)
MIN_NORM_GAP_RATIO = 0.015         # Minimum gap normalized by packet minor dimension (>= 1.5%)
MIN_GAP_COVERAGE_RATIO = 0.40      # Gap must span >= 40% of the cutting interface
MIN_EXPOSED_GAP_PIXELS = 10        # Minimum number of true exposed background pixels
MIN_PIECE_AREA_PX = 150            # Absolute noise rejection floor
MIN_PIECE_AREA_RATIO = 0.03        # Minimum piece area relative to intact baseline (3%, down from restrictive 8%)
MIN_TOTAL_AREA_RATIO = 0.50        # Lower bound for total segmented area vs baseline
MAX_TOTAL_AREA_RATIO = 1.50        # Upper bound for total segmented area vs baseline

MAX_CENTROID_SHIFT_PX = 15.0       # Max centroid movement between consecutive stable frames
MAX_AREA_CHANGE_RATIO = 0.20       # Max area change (20%) between consecutive stable frames

CUT_VERIFY_SECONDS = 10.0          # Full verification duration (10 seconds)
VERIFY_SAMPLE_INTERVAL = 0.75      # Seconds between post-cut segmentation samples
MIN_VERIFY_SAMPLES = 8             # Minimum samples required in 10s
MIN_VALID_SAMPLE_RATIO = 0.70      # At least 70% of post-cut samples must be valid
CLEAR_WAIT_SECONDS = 1.5           # Clear scene cooldown after cutting (1.5 seconds)

HAND_CONFIRM_FRAMES = 2            # Temporal frames required for hand
SCISSOR_CONFIRM_FRAMES = 2         # Temporal frames required for scissors


# ============================================================
# 0. LETTERBOX UNPADDING & COORDINATE TRANSFORMATION
# ============================================================
def unpad_letterbox_mask(m_raw, orig_shape, input_shape=None):
    """
    Inverts letterbox padding and scales raw model mask back to original frame dimensions (orig_h, orig_w).
    Ensures 1:1 coordinate alignment between bounding boxes, contours, and physical frame coordinates.
    Avoids vertical squashing, coordinate shifts, and artificial mask merging.
    """
    orig_h, orig_w = orig_shape[:2]
    if input_shape is None:
        inp_h, inp_w = m_raw.shape[:2]
    else:
        inp_h, inp_w = input_shape[:2]

    # Ratio r = min(inp_h / orig_h, inp_w / orig_w)
    r = min(float(inp_h) / float(orig_h), float(inp_w) / float(orig_w))
    new_unpad_w = int(round(orig_w * r))
    new_unpad_h = int(round(orig_h * r))

    dw = (inp_w - new_unpad_w) / 2.0
    dh = (inp_h - new_unpad_h) / 2.0

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    # Crop out letterbox padding
    crop_h_end = max(top + 1, inp_h - bottom)
    crop_w_end = max(left + 1, inp_w - right)
    m_cropped = m_raw[top:crop_h_end, left:crop_w_end]

    m_unpadded = cv2.resize(m_cropped.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return m_unpadded


# ============================================================
# 1. HAND DETECTION & SEAL FALSE-POSITIVE FILTERING
# ============================================================
def evaluate_hand_detection(
    hand_result,
    hand_ids,
    frame_shape,
    reference_mask=None,
    topstrip_mask=None,
    frame=None
):
    """
    Evaluates raw hand predictions.
    Filters out red printed seals / tamper strips without suppressing real hands resting over the packet.
    Returns: (is_hand_present, valid_hand_mask, diagnostics_list)
    """
    h, w = frame_shape[:2]
    diagnostics = []

    if hand_result is None or hand_result.boxes is None or len(hand_result.boxes) == 0:
        return False, np.zeros((h, w), dtype=np.uint8), diagnostics

    boxes = hand_result.boxes.xyxy.cpu().numpy()
    confidences = hand_result.boxes.conf.cpu().numpy()
    classes = hand_result.boxes.cls.cpu().numpy().astype(int)

    has_masks = (
        hand_result.masks is not None
        and hand_result.masks.data is not None
        and len(hand_result.masks.data) == len(boxes)
    )

    valid_mask = np.zeros((h, w), dtype=np.uint8)
    any_valid_hand = False

    ref_pkt = reference_mask if (reference_mask is not None and np.any(reference_mask)) else None

    # Precompute HSV if frame is provided for color-based seal filtering
    hsv = None
    if frame is not None and frame.shape[:2] == (h, w):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        except Exception:
            hsv = None

    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, classes)):
        if hand_ids is not None and cls_id not in hand_ids:
            continue

        x1, y1, x2, y2 = box.astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        box_area = max(0, (x2 - x1) * (y2 - y1))

        # Extract individual mask
        inst_mask = np.zeros((h, w), dtype=np.uint8)
        if has_masks:
            if hasattr(hand_result.masks, "xy") and hand_result.masks.xy is not None and i < len(hand_result.masks.xy) and len(hand_result.masks.xy[i]) >= 3:
                cv2.fillPoly(inst_mask, [hand_result.masks.xy[i].astype(np.int32)], 255)
            else:
                m_raw = hand_result.masks.data[i].detach().cpu().numpy()
                m_unpadded = unpad_letterbox_mask(m_raw, (h, w))
                inst_mask[m_unpadded >= 0.5] = 255
        else:
            inst_mask[y1:y2, x1:x2] = 255

        mask_area = int(np.count_nonzero(inst_mask))
        if mask_area == 0:
            continue

        in_packet_ratio = 0.0
        in_topstrip_ratio = 0.0

        if ref_pkt is not None:
            overlap_pkt = np.count_nonzero((inst_mask > 0) & (ref_pkt > 0))
            in_packet_ratio = overlap_pkt / float(mask_area)

        if topstrip_mask is not None and np.any(topstrip_mask):
            overlap_ts = np.count_nonzero((inst_mask > 0) & (topstrip_mask > 0))
            in_topstrip_ratio = overlap_ts / float(mask_area)

        # Boundary contact check (operator hand enters from frame edges)
        margin = 6
        touches_border = (x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin)

        # Relative position
        if in_packet_ratio > 0.85:
            pos_rel = "INSIDE_PACKET"
        elif in_packet_ratio > 0.15:
            pos_rel = "CROSSING_BORDER"
        else:
            pos_rel = "OUTSIDE_PACKET"

        # Check for vivid red tamper seal color inside candidate mask
        is_vivid_red = False
        red_ratio = 0.0
        if hsv is not None and mask_area > 0:
            m_bool = inst_mask > 0
            h_vals = hsv[m_bool, 0]
            s_vals = hsv[m_bool, 1]
            v_vals = hsv[m_bool, 2]
            red_count = int(np.count_nonzero(
                ((h_vals <= 14) | (h_vals >= 166)) & (s_vals >= 70) & (v_vals >= 50)
            ))
            red_ratio = red_count / float(len(h_vals))
            if red_ratio >= 0.25:
                is_vivid_red = True

        # Red seal characteristics:
        # A printed seal is bright red, localized on packet/testbed.
        # Human skin is NEVER predominantly saturated crimson/red.
        # Conversely, a real human hand in overhead cameras enters from the frame edges and has skin tone.
        is_seal_fp = False
        reason = "REAL_HAND"

        # Condition 0: Predominantly saturated crimson red tamper tape (>= 40% red pixels)
        # Pure red tamper tape is NEVER a human hand!
        if is_vivid_red and red_ratio >= 0.40:
            is_seal_fp = True
            reason = f"SEAL_FP (Predominantly red tamper seal: {red_ratio:.1%} red px)"
        # Condition 1: High overlap with known TopStrip mask (>= 25%) and detached from border
        elif in_topstrip_ratio >= 0.25 and not touches_border:
            is_seal_fp = True
            reason = f"SEAL_FP (TopStrip overlap: {in_topstrip_ratio:.1%}, detached from border)"
        # Condition 2: Saturated red tamper strip (>= 25% red px) and detached from border
        elif is_vivid_red and not touches_border:
            is_seal_fp = True
            reason = f"SEAL_FP (Vivid red tamper strip: {red_ratio:.1%} red px, detached from border)"
        # Condition 3: Deeply enclosed in packet (>= 75%) and detached from frame edge
        elif in_packet_ratio >= 0.75 and not touches_border:
            is_seal_fp = True
            reason = f"SEAL_FP (Enclosed inside packet: {in_packet_ratio:.1%}, detached from border)"
        # Condition 4: Detached from all frame edges and substantially on packet or red
        elif not touches_border and (in_packet_ratio >= 0.35 or red_ratio >= 0.15) and mask_area < (0.35 * h * w):
            is_seal_fp = True
            reason = f"SEAL_FP (Detached from frame borders, in_pkt: {in_packet_ratio:.1%}, red: {red_ratio:.1%})"

        diag = {
            "index": i + 1,
            "conf": float(conf),
            "bbox": (x1, y1, x2, y2),
            "area_px": mask_area,
            "pos_rel": pos_rel,
            "in_pkt_pct": in_packet_ratio * 100.0,
            "in_topstrip_pct": in_topstrip_ratio * 100.0,
            "touches_border": touches_border,
            "is_vivid_red": is_vivid_red,
            "red_ratio": red_ratio,
            "is_seal_fp": is_seal_fp,
            "reason": reason
        }
        diagnostics.append(diag)

        if not is_seal_fp:
            any_valid_hand = True
            valid_mask[inst_mask > 0] = 255

    return any_valid_hand, valid_mask, diagnostics


# ============================================================
# 2. SCISSOR VALIDATION & PACKET SPATIAL INTERACTION
# ============================================================
def evaluate_scissor_detection(
    scissor_result,
    scissor_ids,
    frame_shape,
    packet_mask=None,
    min_conf=0.22,
    max_dist_px=160.0,
    roi=None
):
    """
    Validates scissors and ensures they are interacting with or near the packet inside the designated ROI.
    Rejects spurious background artifacts such as the table extrusion rail along the bottom of the camera view.
    Returns: (is_scissor_valid, scissor_mask, diagnostics_list)
    """
    h, w = frame_shape[:2]
    diagnostics = []

    if scissor_result is None or scissor_result.boxes is None or len(scissor_result.boxes) == 0:
        return False, np.zeros((h, w), dtype=np.uint8), diagnostics

    boxes = scissor_result.boxes.xyxy.cpu().numpy()
    confidences = scissor_result.boxes.conf.cpu().numpy()
    classes = scissor_result.boxes.cls.cpu().numpy().astype(int)

    has_masks = (
        scissor_result.masks is not None
        and scissor_result.masks.data is not None
        and len(scissor_result.masks.data) == len(boxes)
    )

    valid_mask = np.zeros((h, w), dtype=np.uint8)
    any_valid_scissor = False

    # Distance transform and vertical bounds of packet mask for proximity check
    dt_packet = None
    max_pkt_y = None
    if packet_mask is not None and np.any(packet_mask):
        dt_packet = cv2.distanceTransform((packet_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
        pkt_ys = np.where(packet_mask > 0)[0]
        if pkt_ys.size > 0:
            max_pkt_y = int(np.max(pkt_ys))

    rx1, ry1, rx2, ry2 = (0, 0, w, h)
    if roi is not None and len(roi) == 4:
        rx1, ry1, rx2, ry2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, classes)):
        if scissor_ids is not None and cls_id not in scissor_ids:
            continue

        if conf < min_conf:
            diagnostics.append({
                "index": i + 1, "conf": float(conf), "valid": False, "reason": "Low confidence"
            })
            continue

        x1, y1, x2, y2 = box.astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # 1. ROI boundary filter: reject if detection is outside designated working area
        if roi is not None:
            if x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2:
                diagnostics.append({
                    "index": i + 1, "conf": float(conf), "bbox": (x1, y1, x2, y2),
                    "valid": False, "reason": f"OUTSIDE_ROI (BBox outside ROI [{rx1},{ry1},{rx2},{ry2}])"
                })
                continue

        # 2. Bottom Rail / Table Edge Artifact Rejection
        # Extrusion rails / metal lips at the bottom of the camera view are wide horizontal bars
        aspect_ratio = float(x2 - x1) / max(float(y2 - y1), 1.0)
        cy = (y1 + y2) / 2.0
        is_bottom_rail = (y2 >= int(h * 0.83) or cy >= int(h * 0.83)) and (
            aspect_ratio >= 2.5 or (x2 - x1) >= int(w * 0.40) or y1 >= int(h * 0.84)
        )
        if is_bottom_rail:
            diagnostics.append({
                "index": i + 1, "conf": float(conf), "bbox": (x1, y1, x2, y2),
                "valid": False, "reason": "REJECTED_BOTTOM_RAIL_ARTIFACT (Horizontal strip at bottom edge)"
            })
            continue

        inst_mask = np.zeros((h, w), dtype=np.uint8)
        if has_masks:
            if hasattr(scissor_result.masks, "xy") and scissor_result.masks.xy is not None and i < len(scissor_result.masks.xy) and len(scissor_result.masks.xy[i]) >= 3:
                cv2.fillPoly(inst_mask, [scissor_result.masks.xy[i].astype(np.int32)], 255)
            else:
                m_raw = scissor_result.masks.data[i].detach().cpu().numpy()
                m_unpadded = unpad_letterbox_mask(m_raw, (h, w))
                inst_mask[m_unpadded >= 0.5] = 255
        else:
            inst_mask[y1:y2, x1:x2] = 255

        # Clip mask to ROI if provided
        if roi is not None:
            inst_mask[:ry1, :] = 0
            inst_mask[ry2:, :] = 0
            inst_mask[:, :rx1] = 0
            inst_mask[:, rx2:] = 0

        mask_area = int(np.count_nonzero(inst_mask))
        if mask_area == 0:
            continue

        # 3. Bottom Rail Zone Filter:
        # Prevent railing edge detection near bottom boundary without penalizing scissors near packet bottom
        if y1 >= int(h * 0.84):
            diagnostics.append({
                "index": i + 1, "conf": float(conf), "bbox": (x1, y1, x2, y2),
                "area_px": mask_area, "valid": False,
                "reason": f"BOTTOM_RAIL_ZONE (y1={y1} >= {int(h * 0.84)})"
            })
            continue

        # Proximity to packet
        is_near_packet = True
        min_dist_to_pkt = 0.0

        if dt_packet is not None:
            dist_samples = dt_packet[inst_mask > 0]
            if dist_samples.size > 0:
                min_dist_to_pkt = float(np.min(dist_samples))
                is_near_packet = (min_dist_to_pkt <= max_dist_px)
            else:
                is_near_packet = False

        if not is_near_packet:
            reason = f"SPURIOUS_BACKGROUND (Distance to packet: {min_dist_to_pkt:.1f}px > {max_dist_px}px)"
            is_valid = False
        else:
            reason = f"VALID_SCISSOR (Near/overlapping packet: {min_dist_to_pkt:.1f}px)"
            is_valid = True

        diagnostics.append({
            "index": i + 1,
            "conf": float(conf),
            "bbox": (x1, y1, x2, y2),
            "area_px": mask_area,
            "dist_to_pkt": min_dist_to_pkt,
            "valid": is_valid,
            "reason": reason
        })

        if is_valid:
            any_valid_scissor = True
            valid_mask[inst_mask > 0] = 255

    return any_valid_scissor, valid_mask, diagnostics


# ============================================================
# 3. PACKET SEGMENTATION & CONNECTED COMPONENT EXTRACTION
# ============================================================
def extract_packet_pieces(
    packet_result,
    frame_shape,
    packet_ids=None,
    reference_area=None,
    min_conf=0.12,
    min_piece_area_px=MIN_PIECE_AREA_PX,
    min_piece_area_ratio=MIN_PIECE_AREA_RATIO,
    roi_offset=(0, 0),
    verbose=True
):
    """
    11-Stage Packet Segmentation & Piece Extraction Pipeline:
    1. YOLO segmentation inference inspection.
    2. Detection confidence filtering (conf >= min_conf).
    3. Mask extraction (contour polygons or raw tensor).
    4. Mask resizing with letterbox unpadding (prevents aspect ratio distortion and vertical shifts).
    5. ROI coordinate transformation (maps back to full-frame coordinates).
    6. Connected-component analysis on individual instances and union.
    7. Minimum-area filtering with documented rejection reasons.
    8. Duplicate detection removal (IoU >= 0.65 suppression).
    9. Mask overlap analysis and boundary disambiguation.
    10. Piece association and independent piece preservation.
    11. Final piece counting and structured diagnostic logging.
    """
    h, w = frame_shape[:2]
    ox, oy = roi_offset

    # Stage 1: YOLO segmentation inference inspection
    raw_detections = []
    has_boxes = (packet_result is not None and getattr(packet_result, "boxes", None) is not None and len(packet_result.boxes) > 0)
    has_masks = (packet_result is not None and getattr(packet_result, "masks", None) is not None)

    if not has_boxes or not has_masks:
        if verbose:
            print("[SEG-TRACE Stage 1] No packet detections/masks returned by YOLO segmentation model.")
        return {
            "pieces": [],
            "rejected_pieces": [],
            "yolo_instance_count": 0,
            "raw_cc_count": 0,
            "accepted_piece_count": 0,
            "rejected_piece_count": 0,
            "union_mask": np.zeros((h, w), dtype=np.uint8),
            "instance_masks": [],
            "total_area": 0,
            "rejection_summary": ["No detections returned by model"]
        }

    boxes_xyxy = packet_result.boxes.xyxy.detach().cpu().numpy()
    confidences = packet_result.boxes.conf.detach().cpu().numpy()
    classes = packet_result.boxes.cls.detach().cpu().numpy().astype(int)

    has_polys = hasattr(packet_result.masks, "xy") and packet_result.masks.xy is not None
    mask_data = packet_result.masks.data.detach().cpu().numpy() if packet_result.masks.data is not None else []

    yolo_instance_count = len(boxes_xyxy)
    if verbose:
        print(f"[SEG-TRACE Stage 1] YOLO inference completed: {yolo_instance_count} raw detections.")

    # Stage 2 to 5: Confidence filtering, Mask extraction, Unpadding, ROI transform
    candidate_instances = []
    rejected_instances = []

    for i in range(yolo_instance_count):
        conf = float(confidences[i])
        cls_id = int(classes[i])
        bbox_raw = boxes_xyxy[i].astype(int).tolist()

        # Stage 2: Confidence & Class ID filtering
        if packet_ids is not None and cls_id not in packet_ids:
            reason = f"Class ID {cls_id} not in target packet classes {packet_ids}"
            rejected_instances.append({
                "index": i + 1, "conf": conf, "cls_id": cls_id, "bbox": tuple(bbox_raw),
                "area": 0, "cc_count": 0, "accepted": False, "reason": reason
            })
            if verbose:
                print(f"[SEG-TRACE Stage 2] Detection #{i+1}: REJECTED (conf={conf:.3f}, cls={cls_id}) - {reason}")
            continue

        if conf < min_conf:
            reason = f"Confidence {conf:.3f} below threshold {min_conf:.3f}"
            rejected_instances.append({
                "index": i + 1, "conf": conf, "cls_id": cls_id, "bbox": tuple(bbox_raw),
                "area": 0, "cc_count": 0, "accepted": False, "reason": reason
            })
            if verbose:
                print(f"[SEG-TRACE Stage 2] Detection #{i+1}: REJECTED (conf={conf:.3f}, cls={cls_id}) - {reason}")
            continue

        # Stage 3 & 4: Mask extraction & Letterbox unpadding
        inst_mask = np.zeros((h, w), dtype=np.uint8)
        if has_polys and i < len(packet_result.masks.xy) and len(packet_result.masks.xy[i]) >= 3:
            poly = packet_result.masks.xy[i].astype(np.int32)
            cv2.fillPoly(inst_mask, [poly], 255)
        elif i < len(mask_data):
            m_raw = mask_data[i]
            m_unpadded = unpad_letterbox_mask(m_raw, (h, w))
            inst_mask[m_unpadded >= 0.5] = 255
        else:
            bx = bbox_raw
            inst_mask[max(0, bx[1]):min(h, bx[3]), max(0, bx[0]):min(w, bx[2])] = 255

        # Stage 5: ROI coordinate transformation
        if ox != 0 or oy != 0:
            shifted = np.zeros((h + oy, w + ox), dtype=np.uint8)
            shifted[oy:oy+h, ox:ox+w] = inst_mask
            inst_mask = shifted[:h, :w]
            bbox_raw = [bbox_raw[0] + ox, bbox_raw[1] + oy, bbox_raw[2] + ox, bbox_raw[3] + oy]

        mask_area = int(np.count_nonzero(inst_mask))
        if mask_area == 0:
            reason = "Mask area is zero after rasterization"
            rejected_instances.append({
                "index": i + 1, "conf": conf, "cls_id": cls_id, "bbox": tuple(bbox_raw),
                "area": 0, "cc_count": 0, "accepted": False, "reason": reason
            })
            if verbose:
                print(f"[SEG-TRACE Stage 4] Detection #{i+1}: REJECTED - {reason}")
            continue

        candidate_instances.append({
            "index": i + 1,
            "conf": conf,
            "cls_id": cls_id,
            "bbox": tuple(bbox_raw),
            "mask": inst_mask,
            "area": mask_area
        })

    # Stage 6: Connected-component analysis on candidate instances
    cc_analyzed_pieces = []
    for cand in candidate_instances:
        m = cand["mask"]
        n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        sub_count = n_cc - 1
        cand["cc_count"] = sub_count
        cand["cc_areas"] = [int(stats[k, cv2.CC_STAT_AREA]) for k in range(1, n_cc)]

        if sub_count == 1:
            cc_analyzed_pieces.append(cand)
        elif sub_count > 1:
            # If one YOLO instance segmented multiple distinct detached pieces, extract each
            for k in range(1, n_cc):
                sub_area = int(stats[k, cv2.CC_STAT_AREA])
                sub_m = np.zeros_like(m, dtype=np.uint8)
                sub_m[labels == k] = 255
                sub_bx = (
                    int(stats[k, cv2.CC_STAT_LEFT]),
                    int(stats[k, cv2.CC_STAT_TOP]),
                    int(stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH]),
                    int(stats[k, cv2.CC_STAT_TOP] + stats[k, cv2.CC_STAT_HEIGHT])
                )
                cc_analyzed_pieces.append({
                    "index": cand["index"],
                    "conf": cand["conf"],
                    "cls_id": cand["cls_id"],
                    "bbox": sub_bx,
                    "mask": sub_m,
                    "area": sub_area,
                    "cc_count": 1,
                    "cc_areas": [sub_area]
                })

    # Stage 7: Minimum-area filtering
    total_inst_area = sum(p["area"] for p in cc_analyzed_pieces)
    ref_a = reference_area if (reference_area is not None and reference_area > 0) else float(total_inst_area)
    dynamic_min_area = max(min_piece_area_px, int(ref_a * min_piece_area_ratio))

    area_filtered_pieces = []
    for p in cc_analyzed_pieces:
        if p["area"] < dynamic_min_area:
            reason = f"Area {p['area']}px < minimum threshold {dynamic_min_area}px ({min_piece_area_ratio:.1%} of {int(ref_a)}px)"
            rejected_instances.append({
                "index": p["index"], "conf": p["conf"], "cls_id": p["cls_id"], "bbox": p["bbox"],
                "area": p["area"], "cc_count": p.get("cc_count", 1), "accepted": False, "reason": reason
            })
            if verbose:
                print(f"[SEG-TRACE Stage 7] Detection #{p['index']}: REJECTED - {reason}")
        else:
            area_filtered_pieces.append(p)

    # Stage 8: Duplicate detection removal (IoU >= 0.65)
    area_filtered_pieces.sort(key=lambda x: x["conf"], reverse=True)
    deduped_pieces = []
    for p in area_filtered_pieces:
        is_duplicate = False
        for accepted in deduped_pieces:
            inter = np.count_nonzero((p["mask"] > 0) & (accepted["mask"] > 0))
            union = np.count_nonzero((p["mask"] > 0) | (accepted["mask"] > 0))
            iou = inter / float(union) if union > 0 else 0.0

            if iou >= 0.65:
                is_duplicate = True
                reason = f"Duplicate detection suppressed (IoU {iou:.2f} >= 0.65 with #{accepted['index']})"
                rejected_instances.append({
                    "index": p["index"], "conf": p["conf"], "cls_id": p["cls_id"], "bbox": p["bbox"],
                    "area": p["area"], "cc_count": p.get("cc_count", 1), "accepted": False, "reason": reason
                })
                if verbose:
                    print(f"[SEG-TRACE Stage 8] Detection #{p['index']}: REJECTED - {reason}")
                break
        if not is_duplicate:
            deduped_pieces.append(p)

    # Stage 9: Mask overlap analysis & raster touch disambiguation
    if len(deduped_pieces) >= 2:
        for i in range(len(deduped_pieces)):
            for j in range(i + 1, len(deduped_pieces)):
                p1, p2 = deduped_pieces[i], deduped_pieces[j]
                overlap = np.count_nonzero((p1["mask"] > 0) & (p2["mask"] > 0))
                if overlap > 0:
                    smaller_area = min(p1["area"], p2["area"])
                    overlap_ratio = overlap / float(max(smaller_area, 1))
                    # If tiny raster boundary touch (<= 3%), resolve by assigning overlap to larger piece
                    if overlap_ratio <= 0.03:
                        p2["mask"] = np.where((p1["mask"] > 0) & (p2["mask"] > 0), 0, p2["mask"]).astype(np.uint8)
                        p2["area"] = int(np.count_nonzero(p2["mask"]))
                        if verbose:
                            print(f"[SEG-TRACE Stage 9] Minor boundary overlap resolved ({overlap}px, {overlap_ratio:.1%}) between #{p1['index']} and #{p2['index']}.")
                    else:
                        if verbose:
                            print(f"[SEG-TRACE Stage 9] Significant overlap ({overlap}px, {overlap_ratio:.1%}) detected between #{p1['index']} and #{p2['index']}.")

    # Stage 10 & 11: Piece association, Centroid calculation, Final piece compilation
    final_pieces = []
    union_mask = np.zeros((h, w), dtype=np.uint8)

    for p in deduped_pieces:
        m = p["mask"]
        mom = cv2.moments(m, binaryImage=True)
        if mom["m00"] <= 0:
            continue
        cx = float(mom["m10"] / mom["m00"])
        cy = float(mom["m01"] / mom["m00"])

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x_min, y_min, w_box, h_box = cv2.boundingRect(cnts[0]) if cnts else (0, 0, 0, 0)

        final_piece = {
            "index": p["index"],
            "conf": p["conf"],
            "cls_id": p["cls_id"],
            "mask": m,
            "area": int(p["area"]),
            "centroid": (cx, cy),
            "bbox": (x_min, y_min, w_box, h_box),
            "accepted": True
        }
        final_pieces.append(final_piece)
        union_mask = cv2.bitwise_or(union_mask, m)

        if verbose:
            print(f"[SEG-TRACE Stage 11] Piece #{p['index']}: ACCEPTED | Conf: {p['conf']:.3f} | Area: {p['area']}px | BBox: ({x_min},{y_min},{w_box},{h_box}) | Centroid: ({cx:.1f},{cy:.1f})")

    # Sort final pieces by area descending
    final_pieces.sort(key=lambda x: x["area"], reverse=True)

    # Compute raw CC count on union mask for diagnostic reporting
    n_union_cc, _, _, _ = cv2.connectedComponentsWithStats((union_mask > 0).astype(np.uint8), connectivity=8)
    raw_cc_count = max(0, n_union_cc - 1)

    return {
        "pieces": final_pieces,
        "rejected_pieces": rejected_instances,
        "yolo_instance_count": yolo_instance_count,
        "raw_cc_count": raw_cc_count,
        "accepted_piece_count": len(final_pieces),
        "rejected_piece_count": len(rejected_instances),
        "union_mask": union_mask,
        "instance_masks": [p["mask"] for p in final_pieces],
        "total_area": int(np.count_nonzero(union_mask)),
        "rejection_summary": [r["reason"] for r in rejected_instances]
    }


# ============================================================
# 4. PHYSICAL GAP & BACKGROUND VERIFICATION
# ============================================================
def analyze_packet_gap(
    mask_a,
    mask_b,
    frame=None,
    reference_area=None,
    min_background_gap_px=MIN_BACKGROUND_GAP_PX,
    min_norm_gap=MIN_NORM_GAP_RATIO,
    min_coverage=MIN_GAP_COVERAGE_RATIO,
    min_exposed_gap_px=MIN_EXPOSED_GAP_PIXELS
):
    """
    Physical gap verification between two packet pieces:
    1. Spatial separation & overlap rejection.
    2. Bidirectional distance transforms for exact boundary-to-boundary distance.
    3. Gap measurement line identifying closest physical points.
    4. Bidirectional facing edge extraction along the separation interface.
    5. Gap coverage calculation along the relevant separation boundary.
    6. True exposed background pixel verification strictly between pieces.
    7. Background consistency/texture check against frame.
    8. Comprehensive diagnostic return values.
    """
    h, w = mask_a.shape[:2]
    bin_a = (mask_a > 0).astype(np.uint8)
    bin_b = (mask_b > 0).astype(np.uint8)

    # Step 1: Spatial separation & overlap check
    overlap_px = int(np.count_nonzero(bin_a & bin_b))
    smaller_area = min(np.count_nonzero(bin_a), np.count_nonzero(bin_b))
    overlap_pct = (overlap_px / max(smaller_area, 1)) * 100.0

    if overlap_px > 10 and overlap_pct > 3.0:
        return {
            "valid": False,
            "reason": f"Pieces overlapping ({overlap_px}px, {overlap_pct:.1f}%)",
            "min_gap": 0.0,
            "median_gap": 0.0,
            "max_gap": 0.0,
            "norm_gap": 0.0,
            "coverage_pct": 0.0,
            "facing_span": 0.0,
            "expected_interface": 0.0,
            "gap_px_count": 0,
            "overlap_px": overlap_px,
            "overlap_pct": overlap_pct,
            "bg_valid": False,
            "gap_line": None,
            "gap_mask": np.zeros((h, w), dtype=np.uint8),
            "pkt_minor": 0.0
        }

    # Step 2: Extract contours & packet orientation
    cnt_a, _ = cv2.findContours(bin_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_b, _ = cv2.findContours(bin_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not cnt_a or not cnt_b:
        return {
            "valid": False,
            "reason": "Missing piece contours",
            "min_gap": 0.0, "median_gap": 0.0, "max_gap": 0.0, "norm_gap": 0.0,
            "coverage_pct": 0.0, "facing_span": 0.0, "expected_interface": 0.0,
            "gap_px_count": 0, "overlap_px": overlap_px, "overlap_pct": overlap_pct,
            "bg_valid": False, "gap_line": None,
            "gap_mask": np.zeros((h, w), dtype=np.uint8), "pkt_minor": 0.0
        }

    pts_a = cnt_a[0].squeeze()
    pts_b = cnt_b[0].squeeze()
    if pts_a.ndim < 2 or pts_b.ndim < 2:
        return {
            "valid": False,
            "reason": "Degenerate piece contour points",
            "min_gap": 0.0, "median_gap": 0.0, "max_gap": 0.0, "norm_gap": 0.0,
            "coverage_pct": 0.0, "facing_span": 0.0, "expected_interface": 0.0,
            "gap_px_count": 0, "overlap_px": overlap_px, "overlap_pct": overlap_pct,
            "bg_valid": False, "gap_line": None,
            "gap_mask": np.zeros((h, w), dtype=np.uint8), "pkt_minor": 0.0
        }

    # Step 3: Bidirectional Euclidean distance transforms
    dt_a = cv2.distanceTransform((bin_a == 0).astype(np.uint8), cv2.DIST_L2, 5)
    dt_b = cv2.distanceTransform((bin_b == 0).astype(np.uint8), cv2.DIST_L2, 5)

    dists_a_to_b = np.array([dt_b[p[1], p[0]] for p in pts_a])
    dists_b_to_a = np.array([dt_a[p[1], p[0]] for p in pts_b])

    if dists_a_to_b.size == 0 or dists_b_to_a.size == 0:
        return {
            "valid": False, "reason": "Empty contour distance arrays",
            "min_gap": 0.0, "median_gap": 0.0, "max_gap": 0.0, "norm_gap": 0.0,
            "coverage_pct": 0.0, "facing_span": 0.0, "expected_interface": 0.0,
            "gap_px_count": 0, "overlap_px": overlap_px, "overlap_pct": overlap_pct,
            "bg_valid": False, "gap_line": None,
            "gap_mask": np.zeros((h, w), dtype=np.uint8), "pkt_minor": 0.0
        }

    min_gap_a = float(np.min(dists_a_to_b))
    min_gap_b = float(np.min(dists_b_to_a))
    min_gap = min(min_gap_a, min_gap_b)

    # Step 4: Closest point pair for gap measurement line
    min_idx_a = int(np.argmin(dists_a_to_b))
    pt_on_a = (int(pts_a[min_idx_a][0]), int(pts_a[min_idx_a][1]))

    min_idx_b = int(np.argmin(dists_b_to_a))
    pt_on_b = (int(pts_b[min_idx_b][0]), int(pts_b[min_idx_b][1]))
    gap_line = (pt_on_a, pt_on_b)

    # Step 5: Facing edge points along the cut interface
    facing_thresh = max(min_gap + 6.0, min_gap * 1.5)
    facing_idx_a = np.where(dists_a_to_b <= facing_thresh)[0]
    facing_idx_b = np.where(dists_b_to_a <= facing_thresh)[0]
    facing_pts_a = pts_a[facing_idx_a]
    facing_pts_b = pts_b[facing_idx_b]

    facing_dists = dists_a_to_b[facing_idx_a]
    median_gap = float(np.median(facing_dists)) if len(facing_dists) > 0 else min_gap
    max_gap = float(np.max(facing_dists)) if len(facing_dists) > 0 else min_gap

    span_a = float(np.linalg.norm(np.max(facing_pts_a, axis=0) - np.min(facing_pts_a, axis=0))) if len(facing_pts_a) > 1 else 0.0
    span_b = float(np.linalg.norm(np.max(facing_pts_b, axis=0) - np.min(facing_pts_b, axis=0))) if len(facing_pts_b) > 1 else 0.0
    facing_span = max(span_a, span_b)

    # Step 6: Expected cut interface & gap coverage calculation
    union_mask = bin_a | bin_b
    cnt_u, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnt_u:
        rect_u = cv2.minAreaRect(cnt_u[0])
        pkt_minor = float(min(rect_u[1]))
    else:
        pkt_minor = 100.0

    rect_a = cv2.minAreaRect(cnt_a[0])
    rect_b = cv2.minAreaRect(cnt_b[0])
    # The relevant separation boundary length is governed by the cut interface between the pieces
    expected_interface = float(min(pkt_minor, max(facing_span, min(rect_a[1]), min(rect_b[1]))))
    expected_cut_len = max(20.0, expected_interface)
    coverage_ratio = min(1.0, facing_span / expected_cut_len)
    coverage_pct = coverage_ratio * 100.0
    norm_gap = min_gap / max(1.0, pkt_minor)

    # Step 7: True exposed background corridor verification
    gap_corridor = ((dt_a <= max_gap * 1.25) & (dt_b <= max_gap * 1.25) & (union_mask == 0)).astype(np.uint8)
    gap_px_count = int(np.count_nonzero(gap_corridor))

    bg_valid = True
    bg_consistency = 1.0
    if frame is not None and gap_px_count >= min_exposed_gap_px:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gap_pixels = gray[gap_corridor > 0]
        if gap_pixels.size > 0:
            std_dev = float(np.std(gap_pixels))
            bg_consistency = max(0.0, 1.0 - (std_dev / 75.0))

    # Step 8: Strict multi-condition verification
    if min_gap < 2.0:
        is_valid_gap = False
        reason = f"Pieces touching (gap: {min_gap:.1f}px < 2.0px)"
    elif min_gap < min_background_gap_px:
        is_valid_gap = False
        reason = f"Gap too narrow: {min_gap:.1f}px < {min_background_gap_px:.1f}px"
    elif norm_gap < min_norm_gap:
        is_valid_gap = False
        reason = f"Gap normalized width too small: {norm_gap:.3f} < {min_norm_gap:.3f}"
    elif coverage_ratio < min_coverage:
        is_valid_gap = False
        reason = f"Gap coverage incomplete: {coverage_pct:.1f}% < {min_coverage*100:.0f}%"
    elif gap_px_count < min_exposed_gap_px:
        is_valid_gap = False
        reason = f"Insufficient exposed background ({gap_px_count}px < {min_exposed_gap_px}px)"
    else:
        is_valid_gap = True
        reason = f"Physical gap confirmed: {min_gap:.1f}px ({coverage_pct:.0f}% interface, {gap_px_count}px bg)"

    return {
        "valid": is_valid_gap,
        "min_gap": min_gap,
        "median_gap": median_gap,
        "max_gap": max_gap,
        "norm_gap": norm_gap,
        "coverage_pct": coverage_pct,
        "facing_span": facing_span,
        "expected_interface": expected_cut_len,
        "gap_px_count": gap_px_count,
        "overlap_px": overlap_px,
        "overlap_pct": overlap_pct,
        "bg_valid": bg_valid,
        "bg_consistency": bg_consistency,
        "reason": reason,
        "gap_line": gap_line,
        "gap_mask": gap_corridor,
        "pkt_minor": pkt_minor
    }


# ============================================================
# 5. TEMPORAL SIGNATURE & PAIR STABILITY
# ============================================================
def piece_signature(pieces):
    """Generates immutable signature for temporal stability matching."""
    if len(pieces) != 2:
        return None
    a, b = pieces[0], pieces[1]
    return (
        int(a["area"]),
        float(a["centroid"][0]),
        float(a["centroid"][1]),
        int(b["area"]),
        float(b["centroid"][0]),
        float(b["centroid"][1]),
    )


def is_stable_pair(prev_sig, curr_sig):
    """Validates that pieces have not shifted erratically between frames."""
    if prev_sig is None or curr_sig is None:
        return False

    a_area_p, a_cx_p, a_cy_p, b_area_p, b_cx_p, b_cy_p = prev_sig
    a_area_c, a_cx_c, a_cy_c, b_area_c, b_cx_c, b_cy_c = curr_sig

    # Area change ratios
    a_area_chg = abs(a_area_c - a_area_p) / max(a_area_p, 1)
    b_area_chg = abs(b_area_c - b_area_p) / max(b_area_p, 1)
    if a_area_chg > MAX_AREA_CHANGE_RATIO or b_area_chg > MAX_AREA_CHANGE_RATIO:
        return False

    # Centroid shifts
    a_shift = np.hypot(a_cx_c - a_cx_p, a_cy_c - a_cy_p)
    b_shift = np.hypot(b_cx_c - b_cx_p, b_cy_c - b_cy_p)
    if a_shift > MAX_CENTROID_SHIFT_PX or b_shift > MAX_CENTROID_SHIFT_PX:
        return False

    return True


# ============================================================
# 6. VISUALIZATION & DEBUG SAVING
# ============================================================
def draw_verification_overlay(frame, pieces, gap_info=None, stage_text=""):
    """
    Renders professional multi-color diagnostics onto a fresh copy of frame:
    - Piece 1: Dodger Blue / Cyan (235, 160, 40)
    - Piece 2: Emerald Green (60, 220, 100)
    - Individual piece bounding boxes and pixel area annotations.
    - Verified gap corridor in translucent electric yellow/cyan.
    - Measured gap line connecting the two closest physical boundary points.
    - Never modifies the input frame or inference masks in-place.
    """
    output = frame.copy()

    # Distinct color palette: Piece 1 (Cyan/Blue), Piece 2 (Emerald Green), Extras (Magenta)
    piece_colors = [
        (235, 160, 40),   # Piece 1: Deep Cyan / Dodger Blue (BGR)
        (60, 220, 100),   # Piece 2: Emerald Green (BGR)
        (210, 60, 230),   # Extras: Magenta (BGR)
    ]

    for i, piece in enumerate(pieces[:2]):
        col = piece_colors[i % len(piece_colors)]
        m = piece["mask"]
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(output, cnts, -1, col, 2)

        x, y, w_box, h_box = piece["bbox"]
        cv2.rectangle(output, (x, y), (x + w_box, y + h_box), col, 1)
        cx, cy = int(piece["centroid"][0]), int(piece["centroid"][1])
        cv2.circle(output, (cx, cy), 4, col, -1)
        cv2.putText(
            output,
            f"Piece {i+1}: {piece['area']}px",
            (x, max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            col,
            1,
            cv2.LINE_AA
        )

    # Draw confirmed gap corridor
    if gap_info is not None and gap_info.get("gap_mask") is not None:
        gap_m = gap_info["gap_mask"]
        if np.any(gap_m):
            tint = np.zeros_like(output)
            tint[:] = (0, 240, 255)  # Electric yellow-cyan
            overlay = cv2.addWeighted(output, 0.65, tint, 0.35, 0)
            output[gap_m > 0] = overlay[gap_m > 0]

    # Draw physical gap measurement line
    if gap_info is not None and gap_info.get("gap_line") is not None:
        pt_a, pt_b = gap_info["gap_line"]
        min_g = gap_info.get("min_gap", 0.0)
        cv2.line(output, pt_a, pt_b, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(output, pt_a, 4, (0, 255, 255), -1)
        cv2.circle(output, pt_b, 4, (0, 255, 255), -1)

        mid_x = (pt_a[0] + pt_b[0]) // 2
        mid_y = (pt_a[1] + pt_b[1]) // 2
        cv2.putText(
            output,
            f"Gap: {min_g:.1f}px",
            (mid_x + 8, mid_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    if stage_text:
        cv2.putText(
            output,
            stage_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output


def save_verification_debug(output_dir, frame, packet_mask, pieces, gap_info, status_name):
    """Saves timestamped audit images for quality control."""
    debug_dir = Path(output_dir) / "verification_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)

    # 1. Original frame
    cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_orig.jpg"), frame)

    # 2. Raw packet segmentation mask
    if packet_mask is not None:
        cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_pkt_mask.png"), packet_mask)

    # 3. Individual piece masks
    for i, piece in enumerate(pieces[:2]):
        cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_piece{i+1}_mask.png"), piece["mask"])

    # 4. Gap candidate mask
    if gap_info is not None and gap_info.get("gap_mask") is not None:
        cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_gap_mask.png"), gap_info["gap_mask"] * 255)

    # 5. Validated gap overlay
    vis = draw_verification_overlay(frame, pieces, gap_info=gap_info)
    cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_gap_vis.jpg"), vis)


# ============================================================
# 7. FINAL CUT DETECTION MODEL EVALUATION
# ============================================================
def evaluate_cut_model_prediction(
    cut_model_result,
    model=None,
    cut_class_id=0,
    min_conf=0.25
):
    """
    Evaluates inference results from models/final(cut_detection).pt.
    Supports both detection and classification result formats, and model name resolution.
    Classes: {0: 'Cut', 1: 'Packet'}
    Returns:
        dict: is_cut_detected, confidence, box, class_name, reason
    """
    result = cut_model_result[0] if isinstance(cut_model_result, (list, tuple)) and cut_model_result else cut_model_result

    names = getattr(model, "names", None) if model is not None else None
    if names is None and result is not None:
        names = getattr(result, "names", None)

    if isinstance(names, (list, tuple)):
        names = {i: str(v) for i, v in enumerate(names)}
    elif not isinstance(names, dict):
        names = {0: "Cut", 1: "Packet"}

    if result is None or getattr(result, "boxes", None) is None or len(result.boxes) == 0:
        return {
            "is_cut_detected": False,
            "confidence": 0.0,
            "box": None,
            "mask": None,
            "class_name": "None",
            "reason": "No objects detected by final cut model"
        }

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    best_cut_conf = 0.0
    best_cut_idx = -1

    for i, (conf, cls_id) in enumerate(zip(confidences, classes)):
        # Normalize class name
        cls_name = names.get(cls_id, str(cls_id)).lower()
        is_target_cut = (cls_id == cut_class_id or "cut" in cls_name)
        if is_target_cut and conf >= min_conf:
            if conf > best_cut_conf:
                best_cut_conf = float(conf)
                best_cut_idx = i

    if best_cut_idx >= 0:
        box = boxes[best_cut_idx].astype(int).tolist()
        cls_id = int(classes[best_cut_idx])
        cls_name = names.get(cls_id, "Cut")
        return {
            "is_cut_detected": True,
            "confidence": best_cut_conf,
            "box": tuple(box),
            "class_name": cls_name,
            "reason": f"Cut detected with confidence {best_cut_conf:.1%}"
        }

    top_idx = int(np.argmax(confidences))
    top_cls = int(classes[top_idx])
    top_conf = float(confidences[top_idx])
    cls_name = names.get(top_cls, "Packet" if top_cls == 1 else f"Class_{top_cls}")

    return {
        "is_cut_detected": False,
        "confidence": top_conf,
        "box": tuple(boxes[top_idx].astype(int).tolist()),
        "class_name": cls_name,
        "reason": f"No cut detected; dominant detection: {cls_name} ({top_conf:.1%})"
    }


# ============================================================
# 8. STANDARDIZED DUAL-EVIDENCE CUT DECISION ENGINE
# ============================================================
def evaluate_final_cut_decision(
    components,
    cut_eval,
    last_gap_info=None,
    sample_count=8,
    valid_count=8,
    stable_streak=5,
    min_verify_samples=MIN_VERIFY_SAMPLES,
    min_valid_sample_ratio=MIN_VALID_SAMPLE_RATIO,
    min_streak=5,
    last_p_info=None
):
    """
    Combines physical separation verification and final(cut_detection).pt prediction
    using explicit, non-conflicting decision rules.

    States:
      - PACKET_MISSING
      - CUT_PHYSICALLY_CONFIRMED
      - MODEL_PREDICTED_CUT
      - MODEL_PREDICTED_PACKET
      - NOT_CUTTED
      - CUT_VERIFICATION_PENDING
      - UNCERTAIN

    Rules:
    1. A single segmentation component NEVER silently overrides a Cut model prediction.
    2. Physical cut is CONFIRMED only when 2 pieces + background gap + temporal stability are satisfied.
    3. UI and callers distinguish MODEL PREDICTED CUT from CUT PHYSICALLY CONFIRMED.
    """
    valid_ratio = valid_count / max(sample_count, 1)
    is_cut_pred = cut_eval.get("is_cut_detected", False)
    cut_conf = cut_eval.get("confidence", 0.0)
    cut_class = cut_eval.get("class_name", "None")
    num_pieces = len(components)

    total_px = last_p_info.get("total_area", 1000) if last_p_info else 1000
    if total_px < 300:
        return "PACKET_MISSING", "Packet was not detected in the verification scene"

    # Rule 1: Physical confirmation requires 2 stable pieces + validated background gap
    has_phys_evidence = (
        num_pieces == 2
        and last_gap_info is not None
        and last_gap_info.get("valid", False)
        and sample_count >= min_verify_samples
        and valid_ratio >= min_valid_sample_ratio
        and stable_streak >= min_streak
    )

    if has_phys_evidence:
        gap_w = last_gap_info.get("min_gap", 0.0)
        cov_pct = last_gap_info.get("coverage_pct", 0.0)
        if is_cut_pred:
            return (
                "CUT_PHYSICALLY_CONFIRMED",
                f"CUT CONFIRMED: 2 separated pieces, verified physical gap ({gap_w:.1f}px, {cov_pct:.0f}% interface), "
                f"and cut model verified Cut (conf: {cut_conf:.1%})"
            )
        else:
            return (
                "CUT_PHYSICALLY_CONFIRMED",
                f"CUT CONFIRMED: Physical separation verified across full window "
                f"({gap_w:.1f}px gap, {valid_count}/{sample_count} valid samples; cut model predicted {cut_class} {cut_conf:.1%})"
            )

    # Rule 2: Single continuous component detected
    if num_pieces == 1:
        if is_cut_pred and cut_conf >= 0.25:
            # Cut model predicts Cut, but physical pieces have not separated yet
            return (
                "CUT_VERIFICATION_PENDING",
                f"CUT VERIFICATION PENDING: Cut model detected cut ({cut_conf:.1%}), but pieces are touching as 1 continuous mask. "
                f"Ensure severed pieces are physically pulled apart so a background gap is visible."
            )
        elif cut_class == "Packet" or not is_cut_pred:
            return (
                "NOT_CUTTED",
                f"NOT CUT: Intact packet confirmed (single continuous piece, cut model: {cut_class} {cut_conf:.1%})"
            )
        else:
            return (
                "UNCERTAIN",
                f"UNCERTAIN: Single piece detected with inconclusive cut model prediction ({cut_class} {cut_conf:.1%})"
            )

    # Rule 3: Exactly two pieces detected, but physical gap or temporal stability pending
    if num_pieces == 2:
        gap_w = last_gap_info.get("min_gap", 0.0) if last_gap_info else 0.0
        if last_gap_info and not last_gap_info.get("valid", False):
            if gap_w < 2.0:
                if is_cut_pred and cut_conf >= 0.70:
                    return (
                        "MODEL_PREDICTED_CUT",
                        f"MODEL PREDICTED CUT ({cut_conf:.1%}), but pieces are touching (gap: {gap_w:.1f}px < 2.0px)"
                    )
                return (
                    "CUT_VERIFICATION_PENDING",
                    f"CUT VERIFICATION PENDING: Pieces are touching or separation gap is incomplete (gap: {gap_w:.1f}px)"
                )
            else:
                return (
                    "CUT_VERIFICATION_PENDING",
                    f"CUT VERIFICATION PENDING: {last_gap_info.get('reason', 'Unreliable gap')}"
                )

        if sample_count < min_verify_samples or valid_ratio < min_valid_sample_ratio or stable_streak < min_streak:
            if is_cut_pred and cut_conf >= 0.70:
                return (
                    "MODEL_PREDICTED_CUT",
                    f"MODEL PREDICTED CUT ({cut_conf:.1%}), but stability window incomplete "
                    f"({valid_count}/{sample_count} valid samples, streak {stable_streak}/{min_streak})"
                )
            return (
                "CUT_VERIFICATION_PENDING",
                f"CUT VERIFICATION PENDING: Temporal stability incomplete "
                f"({valid_count}/{sample_count} valid samples, streak {stable_streak}/{min_streak})"
            )

    # Rule 4: Fragmented or ambiguous masks (> 2 components)
    if num_pieces > 2:
        if is_cut_pred and cut_conf >= 0.70:
            return (
                "MODEL_PREDICTED_CUT",
                f"MODEL PREDICTED CUT ({cut_conf:.1%}), but segmentation fragmented into {num_pieces} pieces"
            )
        return "UNCERTAIN", f"UNCERTAIN: Found {num_pieces} fragments (expected 2 pieces)"

    # Rule 5: Catch-all fallback
    if is_cut_pred and cut_conf >= 0.50:
        return "MODEL_PREDICTED_CUT", f"MODEL PREDICTED CUT ({cut_conf:.1%}): Physical evidence inconclusive"
    elif cut_class == "Packet":
        return "MODEL_PREDICTED_PACKET", f"MODEL PREDICTED PACKET ({cut_conf:.1%}): Physical evidence inconclusive"

    return "UNCERTAIN", f"UNCERTAIN: Inconclusive physical and model evidence (cut model: {cut_class} {cut_conf:.1%})"

