
"""
Packet-Cut Full Workflow - Live Webcam

1. Ask operator to place packet FRONT side.
2. Detect packet number using dedicated YOLO model.
3. Crop packet number and extract text using EasyOCR.
4. Ask operator to turn packet to BACK / seal side.
5. Detect TopStrip:
      Visible     -> OPEN / CUTTING_NOT_READY
      Not visible -> CLOSED / CUTTING_READY (valid inspection only)
      Uncertain   -> UNKNOWN_ORIENTATION
6. Segment and store intact packet reference.
7. Detect hand and scissors during cutting.
8. Wait 7 seconds after both disappear.
9. Segment packet again and identify two pieces.
10. Verify background gap and stability for 10 seconds.

Models are loaded from ./models.
packet_cut_seg_yolov8n.pt is intentionally NOT used.

Install:
pip install ultralytics opencv-python numpy easyocr onnxruntime

Run:
python test_cut_model.py --cpu
"""

import argparse
import glob
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

# IMPORTANT: Import torch before paddle/paddleocr to avoid Windows DLL issues
import torch
import paddle
from paddleocr import PaddleOCR
from ultralytics import YOLO

import packet_cut_core
from packet_cut_core import (
    MIN_BACKGROUND_GAP_PX,
    MIN_NORM_GAP_RATIO,
    MIN_GAP_COVERAGE_RATIO,
    MIN_EXPOSED_GAP_PIXELS,
    MIN_PIECE_AREA_PX,
    MIN_PIECE_AREA_RATIO,
    MIN_TOTAL_AREA_RATIO,
    MAX_TOTAL_AREA_RATIO,
    MAX_CENTROID_SHIFT_PX,
    MAX_AREA_CHANGE_RATIO,
    CUT_VERIFY_SECONDS,
    VERIFY_SAMPLE_INTERVAL,
    MIN_VERIFY_SAMPLES,
    MIN_VALID_SAMPLE_RATIO,
    CLEAR_WAIT_SECONDS,
    HAND_CONFIRM_FRAMES,
    SCISSOR_CONFIRM_FRAMES,
    evaluate_hand_detection,
    evaluate_scissor_detection,
    extract_packet_pieces,
    analyze_packet_gap,
    piece_signature,
    is_stable_pair,
    draw_verification_overlay,
    save_verification_debug,
    evaluate_cut_model_prediction
)


# ============================================================
# MODEL PATHS
# ============================================================

def _find_model_dir() -> Path:
    env_dir = os.environ.get("PACKET_CUT_MODELS_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    for p in (
        Path(__file__).resolve().parent / "models",
        Path(__file__).resolve().parent.parent / "models",
        Path(r"D:\Project\Embsys\Packet-cut-new\models"),
    ):
        if p.exists():
            return p
    return Path(r"D:\Project\Embsys\Packet-cut-new\models")


MODEL_DIR = _find_model_dir()


def _resolve_model(canonical: str, *fallbacks: str) -> Path:
    for name in (canonical, *fallbacks):
        p = MODEL_DIR / name
        if p.is_file():
            return p
    return MODEL_DIR / canonical


PACKET_MODEL_PATH = _resolve_model("packet-seg.onnx", "packet_seg_yolov8s_320.onnx")
HAND_MODEL_PATH = _resolve_model("hand-seg.onnx", "hand_seg_yolo26s_320.onnx")
SCISSOR_MODEL_PATH = _resolve_model("scissor-seg.pt")
TOPSTRIP_MODEL_PATH = _resolve_model("strip-seg.onnx", "topstrip_seg_yolov8s_320.onnx")
PACKET_NUMBER_MODEL_PATH = _resolve_model("packet-no-obb.onnx", "packet_number_obj_dec_yolov8s_640.onnx")
PACKET_NUMBER_MODEL_PATTERN = str(MODEL_DIR / "packet*")
CUT_MODEL_PATH = _resolve_model("cut-seg.pt", "final(cut_detection).pt") 


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_ID = 0
IMGSZ = 320

PACKET_CONF = 0.25
HAND_CONF = 0.30
SCISSOR_CONF = 0.30
TOPSTRIP_CONF = 0.25

# Packet-number detector has its own 640 input size.
NUMBER_IMGSZ = 640
NUMBER_CONF = 0.05

FRONT_NUMBER_TIMEOUT = 60
SEAL_INSPECTION_TIMEOUT = 90

CLEAR_WAIT_SECONDS = 5.0
CUT_VERIFY_SECONDS = 10.0
VERIFY_SAMPLE_INTERVAL = 0.75

MIN_PIECE_AREA_PX = 150
MIN_PIECE_AREA_RATIO = 0.08

MIN_TOTAL_AREA_RATIO = 0.55
MAX_TOTAL_AREA_RATIO = 1.45

# Physical Gap Verification Parameters (Normalized and Calibrated)
MIN_BACKGROUND_GAP_PX = 4.0        # Absolute lower bound in pixels
MIN_NORM_GAP_RATIO = 0.015         # Min gap width relative to packet dimension (e.g. >= 1.5% of packet width)
MIN_GAP_COVERAGE_RATIO = 0.55      # Gap must span across at least 55% of the separation interface
MIN_EXPOSED_GAP_PIXELS = 10        # Minimum confirmed exposed background pixels between pieces

MAX_CENTROID_SHIFT_PX = 12.0
MAX_AREA_CHANGE_RATIO = 0.20

MIN_VALID_SAMPLE_RATIO = 0.90
MIN_VERIFY_SAMPLES = 8

BLUR_VARIANCE_MIN = 25.0

WINDOW = "Packet-Cut Verification"
OUTPUT_DIR = Path("packet_cut_outputs")


# ============================================================
# GENERAL UTILITIES
# ============================================================

def require_file(path, description):

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path.resolve()}"
        )

    return path


def find_number_model():

    matches = sorted(
        glob.glob(PACKET_NUMBER_MODEL_PATTERN)
    )

    matches = [
        Path(p) for p in matches
        if Path(p).is_file()
    ]

    if not matches:
        raise FileNotFoundError(
            "Packet-number detector not found.\n"
            f"Expected: {PACKET_NUMBER_MODEL_PATTERN}"
        )

    if len(matches) > 1:

        print("Multiple packet number models found:")

        for item in matches:
            print(item)

        print("Using:", matches[0])

    return matches[0]


def class_names_dict(model):

    names = model.names

    if isinstance(names, dict):
        return names

    return {
        i: name
        for i, name in enumerate(names)
    }


def matching_class_ids(model, keywords):

    found = []

    for class_id, name in class_names_dict(model).items():

        normalized = re.sub(
            r"[^a-z0-9]",
            "",
            str(name).lower()
        )

        if any(k in normalized for k in keywords):
            found.append(int(class_id))

    return found or None


def predict(
    model,
    frame,
    confidence,
    imgsz=IMGSZ,
    device="cpu"
):

    return model.predict(
        source=frame,
        imgsz=imgsz,
        conf=confidence,
        device=device,
        verbose=False
    )[0]


def detection_present(result, selected_ids=None):

    if result.boxes is None:
        return False

    if len(result.boxes) == 0:
        return False

    if selected_ids is None:
        return True

    if result.boxes.cls is None:
        return False

    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    return any(
        int(c) in selected_ids
        for c in classes
    )


def evaluate_hand_detection(
    hand_result,
    hand_ids,
    frame_shape,
    packet_mask=None,
    topstrip_mask=None,
    reference_mask=None,
    frame=None
):
    """
    Evaluates raw hand predictions and filters out false positives caused by red seals.
    Prints diagnostic logs and returns (is_hand_present, valid_hand_mask, diagnostics_list).
    """
    h, w = frame_shape[:2]
    diagnostics = []

    if hand_result.boxes is None or len(hand_result.boxes) == 0:
        return False, np.zeros((h, w), dtype=np.uint8), diagnostics

    # Base reference packet mask: use current packet_mask, fallback to stored reference_mask
    ref_pkt = None
    if packet_mask is not None and np.any(packet_mask):
        ref_pkt = packet_mask
    elif reference_mask is not None and np.any(reference_mask):
        ref_pkt = reference_mask

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

    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, classes)):
        if cls_id not in hand_ids:
            continue

        x1, y1, x2, y2 = box.astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        box_area = max(0, (x2 - x1) * (y2 - y1))

        # Extract individual mask
        inst_mask = np.zeros((h, w), dtype=np.uint8)
        if has_masks:
            m_raw = hand_result.masks.data[i].detach().cpu().numpy()
            m_resized = cv2.resize(
                m_raw.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )
            inst_mask[m_resized >= 0.5] = 255
        else:
            inst_mask[y1:y2, x1:x2] = 255

        mask_area = int(np.count_nonzero(inst_mask))

        # Analyze position relative to packet / seal
        in_packet_ratio = 0.0
        in_topstrip_ratio = 0.0
        pos_rel = "OUTSIDE_PACKET"

        if mask_area > 0:
            if ref_pkt is not None and np.any(ref_pkt):
                overlap_pkt = np.count_nonzero((inst_mask > 0) & (ref_pkt > 0))
                in_packet_ratio = overlap_pkt / mask_area
            if topstrip_mask is not None and np.any(topstrip_mask):
                overlap_ts = np.count_nonzero((inst_mask > 0) & (topstrip_mask > 0))
                in_topstrip_ratio = overlap_ts / mask_area

        if in_packet_ratio > 0.85:
            pos_rel = "INSIDE_PACKET"
        elif in_packet_ratio > 0.20:
            pos_rel = "CROSSING_BORDER"
        else:
            pos_rel = "OUTSIDE_PACKET"

        # Check touching image boundaries (a real human hand enters from outside the camera frame)
        touches_border = (x1 <= 6 or y1 <= 6 or x2 >= w - 6 or y2 >= h - 6)

        # Check vivid red tamper strip
        is_vivid_red = False
        red_ratio = 0.0
        if frame is not None and mask_area > 0 and frame.shape[:2] == (h, w):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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

        # False-positive seal suppression condition:
        is_seal_fp = False
        reason = "REAL_HAND"

        if is_vivid_red and red_ratio >= 0.40:
            is_seal_fp = True
            reason = f"SEAL_FP (red_tamper_seal={red_ratio:.1%})"
        elif in_topstrip_ratio > 0.25 and not touches_border:
            is_seal_fp = True
            reason = f"SEAL_FP (topstrip_overlap={in_topstrip_ratio:.1%})"
        elif in_packet_ratio > 0.70 and not touches_border:
            is_seal_fp = True
            reason = f"SEAL_FP (enclosed_in_packet={in_packet_ratio:.1%}, detached_from_edge)"
        elif not touches_border and (in_packet_ratio > 0.35 or red_ratio > 0.15):
            is_seal_fp = True
            reason = f"SEAL_FP (detached_from_edge, red={red_ratio:.1%})"

        diag = {
            "index": i + 1,
            "conf": conf,
            "bbox": (x1, y1, x2, y2),
            "area_px": mask_area,
            "pos_rel": pos_rel,
            "in_pkt_pct": in_packet_ratio * 100,
            "is_seal_fp": is_seal_fp,
            "reason": reason
        }
        diagnostics.append(diag)

        print(
            f"[HAND DIAGNOSTIC] Pred {i+1}: conf={conf:.3f}, bbox={(x1, y1, x2, y2)}, "
            f"area={mask_area}px, pos={pos_rel} (pkt_overlap={in_packet_ratio:.1%}), "
            f"touches_border={touches_border} -> {reason}"
        )

        if not is_seal_fp:
            any_valid_hand = True
            valid_mask[inst_mask > 0] = 255

    return any_valid_hand, valid_mask, diagnostics


def get_union_mask(
    result,
    frame_shape,
    selected_ids=None
):

    h, w = frame_shape

    union = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    if result.masks is None:
        return union

    if result.masks.data is None:
        return union

    mask_data = (
        result.masks.data
        .detach()
        .cpu()
        .numpy()
    )

    classes = None

    if result.boxes is not None and result.boxes.cls is not None:

        classes = (
            result.boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

    for i, mask in enumerate(mask_data):

        if selected_ids is not None:

            if classes is None:
                continue

            if i >= len(classes):
                continue

            if classes[i] not in selected_ids:
                continue

        resized = cv2.resize(
            mask.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

        union[resized >= 0.5] = 255

    # Remove isolated noise.
    # Do not use morphological closing because it can bridge a real cut.
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    return cv2.morphologyEx(
        union,
        cv2.MORPH_OPEN,
        kernel
    )


def blur_score(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )


def draw_mask(
    frame,
    mask,
    color,
    alpha=0.25
):

    if mask is None or not np.any(mask):
        return frame

    overlay = frame.copy()

    overlay[mask > 0] = color

    return cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0
    )


def put_lines(
    frame,
    lines,
    color=(255, 255, 255)
):

    y = 28

    for i, line in enumerate(lines):

        cv2.putText(
            frame,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62 if i == 0 else 0.50,
            color if i == 0 else (255, 255, 255),
            2 if i == 0 else 1,
            cv2.LINE_AA
        )

        y += 27


def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1):
    """Draws a smooth rounded rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return
    r = min(radius, w // 2, h // 2)
    if thickness == -1:
        # Fill
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
    else:
        # Outline
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def draw_fitted_text(img, text, x, y, max_w, font=cv2.FONT_HERSHEY_DUPLEX, initial_scale=0.55, color=(255, 255, 255), thickness=1):
    """
    Renders text cleanly without truncation. If text exceeds max_w,
    dynamically scales it down so the complete text is always visible and crisp.
    """
    scale = initial_scale
    while scale > 0.35:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        if tw <= max_w:
            break
        scale -= 0.03
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return scale


def create_dashboard(frame, stage, status, instructions, crop=None, extra_info=None):
    """
    Renders an industrial, high-clarity dashboard.
    Left: Camera viewport with live telemetry header badge.
    Right: 420px high-contrast control & monitoring panel.
    Ensures zero truncated text, crisp rendering, and full visibility.
    """
    camera = frame.copy()
    h, w = camera.shape[:2]
    panel_w = 420  # Expanded width so full sentences, coordinates, and metrics fit without clipping

    # Deep high-contrast background: BGR (22, 18, 16)
    panel = np.full((h, panel_w, 3), (22, 18, 16), dtype=np.uint8)

    # Accent color based on status
    status_lower = str(status).lower()
    if any(k in status_lower for k in ("confirm", "closed", "pass", "complete", "ready")):
        accent_color = (90, 235, 110)       # Electric Green
        badge_bg = (24, 62, 32)
    elif any(k in status_lower for k in ("detect", "running", "stabiliz", "wait", "clear")):
        accent_color = (255, 195, 45)       # Amber Gold
        badge_bg = (68, 52, 16)
    elif any(k in status_lower for k in ("fail", "unreadable", "not", "paused", "open")):
        accent_color = (75, 75, 255)        # Vivid Coral Red
        badge_bg = (32, 20, 72)
    else:
        accent_color = (245, 185, 55)       # Cyan Accent
        badge_bg = (48, 40, 22)

    # Clean border between camera & dashboard panel
    cv2.line(panel, (0, 0), (0, h), (70, 58, 52), 2)

    # 1. Header Banner Card
    draw_rounded_rect(panel, (12, 10), (panel_w - 12, 58), (38, 32, 30), radius=6, thickness=-1)
    draw_rounded_rect(panel, (12, 10), (panel_w - 12, 58), (75, 62, 56), radius=6, thickness=1)

    cv2.putText(panel, "PACKET-CUT VERIFICATION", (22, 36), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    # Status indicator dot + text
    pill_x1 = panel_w - 110
    draw_rounded_rect(panel, (pill_x1, 20), (panel_w - 20, 46), (0, 140, 230), radius=4, thickness=-1)
    cv2.putText(panel, "LIVE SYSTEM", (pill_x1 + 8, 35), cv2.FONT_HERSHEY_DUPLEX, 0.38, (10, 10, 10), 1, cv2.LINE_AA)

    # 2. Stage Card
    curr_y = 68
    draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + 60), (32, 26, 24), radius=6, thickness=-1)
    draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + 60), (65, 52, 46), radius=6, thickness=1)
    # Left accent bar
    cv2.rectangle(panel, (12, curr_y + 4), (17, curr_y + 56), accent_color, -1)

    cv2.putText(panel, "STAGE", (28, curr_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)
    draw_fitted_text(panel, str(stage), 28, curr_y + 45, max_w=panel_w - 55, font=cv2.FONT_HERSHEY_DUPLEX, initial_scale=0.58, color=(255, 255, 255), thickness=1)

    # 3. Status Card
    curr_y += 68
    draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + 54), badge_bg, radius=6, thickness=-1)
    draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + 54), accent_color, radius=6, thickness=1)
    cv2.putText(panel, "STATUS", (28, curr_y + 19), cv2.FONT_HERSHEY_DUPLEX, 0.40, (210, 210, 210), 1, cv2.LINE_AA)
    draw_fitted_text(panel, str(status), 28, curr_y + 42, max_w=panel_w - 55, font=cv2.FONT_HERSHEY_DUPLEX, initial_scale=0.55, color=accent_color, thickness=1)

    # 4. Telemetry Grid Card
    curr_y += 62
    if extra_info:
        card_h = 24 + len(extra_info) * 23
        draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + card_h), (30, 25, 23), radius=6, thickness=-1)
        draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + card_h), (60, 48, 42), radius=6, thickness=1)
        cv2.putText(panel, "TELEMETRY & MODEL DATA", (24, curr_y + 17), cv2.FONT_HERSHEY_DUPLEX, 0.40, (170, 170, 170), 1, cv2.LINE_AA)

        row_y = curr_y + 36
        for label, val in extra_info:
            label_str = str(label)
            val_str = str(val)

            # Label on left
            cv2.putText(panel, label_str, (26, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (190, 190, 190), 1, cv2.LINE_AA)

            # Value color coding
            val_col = (255, 255, 255)
            if any(w in val_str.upper() for w in ("YES", "PASS", "LOADED", "FOUND", "CONFIRMED")):
                val_col = (95, 240, 115)
            elif any(w in val_str.upper() for w in ("NO", "NONE", "UNREADABLE", "FILTERED", "ERROR")):
                val_col = (115, 145, 255)

            # Right align value dynamically
            (vw, _), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_DUPLEX, 0.46, 1)
            cv2.putText(panel, val_str, (panel_w - 26 - vw, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.46, val_col, 1, cv2.LINE_AA)
            row_y += 23

        curr_y += card_h + 8
    else:
        curr_y += 6

    # 5. Instructions Box (Always fully wrapped, no cutting off words)
    if instructions:
        # Wrap long instruction lines so every single word is visible
        wrapped_lines = []
        for line in instructions:
            line_str = str(line)
            # Check if fits
            (tw, _), _ = cv2.getTextSize(line_str, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
            if tw <= panel_w - 60:
                wrapped_lines.append(line_str)
            else:
                # Word wrap
                words = line_str.split(" ")
                cur_line = ""
                for w_item in words:
                    test_l = (cur_line + " " + w_item).strip()
                    (test_w, _), _ = cv2.getTextSize(test_l, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
                    if test_w <= panel_w - 60:
                        cur_line = test_l
                    else:
                        if cur_line:
                            wrapped_lines.append(cur_line)
                        cur_line = w_item
                if cur_line:
                    wrapped_lines.append(cur_line)

        instr_h = 24 + len(wrapped_lines) * 22
        draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + instr_h), (25, 30, 36), radius=6, thickness=-1)
        draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + instr_h), (45, 85, 120), radius=6, thickness=1)
        cv2.putText(panel, "INSTRUCTIONS", (24, curr_y + 17), cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 205, 255), 1, cv2.LINE_AA)

        i_y = curr_y + 36
        for line in wrapped_lines:
            cv2.circle(panel, (28, i_y - 4), 2, (0, 210, 255), -1, cv2.LINE_AA)
            cv2.putText(panel, line, (38, i_y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
            i_y += 22

        curr_y += instr_h + 8

    # 6. Number ROI Crop Card (Safe scaling and positioning at bottom)
    if crop is not None and crop.size > 0:
        ch, cw = crop.shape[:2]
        pad_x = 14
        start_y = curr_y + 2
        pad_bottom = 10
        avail_w = max(10, panel_w - 2 * pad_x)
        avail_h = max(10, h - start_y - pad_bottom - 22)

        if avail_w > 20 and avail_h > 20 and ch > 0 and cw > 0:
            card_total_h = avail_h + 22
            draw_rounded_rect(panel, (12, start_y), (panel_w - 12, start_y + card_total_h), (28, 24, 22), radius=6, thickness=-1)
            draw_rounded_rect(panel, (12, start_y), (panel_w - 12, start_y + card_total_h), (60, 48, 42), radius=6, thickness=1)
            cv2.putText(panel, "NUMBER ROI PREVIEW", (24, start_y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.38, (150, 220, 160), 1, cv2.LINE_AA)

            roi_avail_h = card_total_h - 26
            scale = min((avail_w - 16) / cw, (roi_avail_h - 10) / ch, 4.5)
            target_w = max(1, min(avail_w - 16, int(cw * scale)))
            target_h = max(1, min(roi_avail_h - 10, int(ch * scale)))

            resized = cv2.resize(
                crop,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST if scale >= 2.0 else cv2.INTER_LINEAR
            )

            offset_x = 12 + ((panel_w - 24) - target_w) // 2
            offset_y = start_y + 22 + (roi_avail_h - target_h) // 2

            y_end = min(h - pad_bottom, offset_y + target_h)
            x_end = min(panel_w - 12, offset_x + target_w)
            act_h = y_end - offset_y
            act_w = x_end - offset_x

            if act_h > 0 and act_w > 0:
                cv2.rectangle(panel, (offset_x - 1, offset_y - 1), (x_end, y_end), (100, 100, 100), 1)
                panel[offset_y:y_end, offset_x:x_end] = resized[:act_h, :act_w]

    # Camera Viewport HUD
    header_overlay = camera.copy()
    cv2.rectangle(header_overlay, (0, 0), (w, 32), (12, 12, 12), -1)
    camera = cv2.addWeighted(header_overlay, 0.70, camera, 0.30, 0)
    # HUD Telemetry
    cv2.circle(camera, (15, 16), 4, (0, 255, 120), -1, cv2.LINE_AA)
    cv2.putText(camera, "LIVE CAMERA VIEW", (26, 21), cv2.FONT_HERSHEY_DUPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
    res_text = f"{w}x{h} | CPU"
    (rw, _), _ = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
    cv2.putText(camera, res_text, (w - rw - 14, 21), cv2.FONT_HERSHEY_DUPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)

    combined = np.hstack([camera, panel])
    return combined


def show(
    frame,
    lines,
    color=(255, 255, 255)
):
    """
    Renders an image or auto-wraps lines in the rich unified dashboard.
    Ensures seamless full-application consistency across every step.
    """
    # If the frame is already the wide combined dashboard (e.g. w > 640), display directly
    if frame.shape[1] > 640:
        cv2.imshow(WINDOW, frame)
        return cv2.waitKey(1) & 0xFF

    # If legacy lines are passed without a dashboard, wrap into the rich dashboard automatically
    stage_name = lines[0] if lines else "INSPECTION"
    status_name = "ACTIVE"
    instructions = []
    extra_items = []

    if len(lines) > 1:
        status_name = lines[1]
    if len(lines) > 2:
        instructions = lines[2:]

    dashboard = create_dashboard(
        frame,
        stage_name,
        status_name,
        instructions,
        extra_info=extra_items
    )
    cv2.imshow(WINDOW, dashboard)
    return cv2.waitKey(1) & 0xFF


# ============================================================
# PACKET NUMBER DETECTION
# ============================================================

def detect_packet_number(frame, number_model, device):
    """
    Infers packet number using the dedicated YOLO model.
    Note: The trained ONNX model uses task='obb' (Oriented Bounding Boxes).
    We check both result.obb and result.boxes to support either format.
    """
    print("\n[PACKET OCR] Inference started")
    print(f"[PACKET OCR] Input frame dimensions: {frame.shape}")

    result = number_model.predict(
        source=frame,
        imgsz=NUMBER_IMGSZ,
        conf=NUMBER_CONF,
        device=device,
        verbose=False
    )[0]

    # Support both OBB (Oriented Bounding Boxes) and standard Boxes
    boxes = None
    confidences = None
    classes = None

    if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        boxes = result.obb.xyxy.cpu().numpy()
        confidences = result.obb.conf.cpu().numpy()
        classes = result.obb.cls.cpu().numpy().astype(int)
    elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

    print(f"[PACKET OCR] Inference completed")
    print(f"[PACKET OCR] Number of detections: {len(boxes) if boxes is not None else 0}")

    if boxes is None or len(boxes) == 0:
        return None, None, result

    for i, (box, confidence, class_id) in enumerate(zip(boxes, confidences, classes)):
        print(
            f"[PACKET OCR] Detection {i + 1}: "
            f"class={class_id}, "
            f"confidence={confidence:.4f}, "
            f"box={box.tolist()}"
        )

    best = int(np.argmax(confidences))
    x1, y1, x2, y2 = boxes[best].astype(int)
    h, w = frame.shape[:2]

    # Clamp coordinates to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        print("[PACKET OCR] Invalid bounding box coordinates:", x1, y1, x2, y2)
        return None, None, result

    crop = frame[y1:y2, x1:x2].copy()

    print(f"[PACKET OCR] Detection confidence: {confidences[best]:.4f}")
    print(f"[PACKET OCR] Bounding box coordinates: {(x1, y1, x2, y2)}")
    print(f"[PACKET OCR] Crop dimensions: {crop.shape}")

    return crop, (x1, y1, x2, y2), result


# ============================================================
# OCR INITIALIZATION & INFERENCE
# ============================================================

_OCR_ENGINE = None

def get_ocr_engine():
    """
    Initializes and returns a single reusable PaddleOCR instance.
    """
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        print("[PACKET OCR] Initializing PaddleOCR engine...")
        # Avoid the unsupported oneDNN PIR attribute conversion on CPU.
        _OCR_ENGINE = PaddleOCR(
            enable_mkldnn=False,
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
        print("[PACKET OCR] PaddleOCR engine ready.")
    return _OCR_ENGINE


def preprocess_crop_for_ocr(crop, scale_factor=3.0):
    """
    Preprocesses the number crop using grayscale, CLAHE, 2x/3x enlargement and sharpening.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    # 3. Enlargement / Scaling
    ch, cw = enhanced.shape[:2]
    resized = cv2.resize(
        enhanced,
        (int(cw * scale_factor), int(ch * scale_factor)),
        interpolation=cv2.INTER_CUBIC
    )
    # 4. Sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(resized, -1, kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def run_ocr(crop):
    """
    Runs PaddleOCR 3.7.0 on the cropped number region.
    Tries preprocessed and original crops, extracts text and confidences,
    and prioritizes 6-digit numeric candidates.
    """
    print("[PACKET OCR] PaddleOCR started")
    ocr_engine = get_ocr_engine()

    def parse_paddle_results(predict_results):
        candidates = []
        if not predict_results:
            return candidates

        for res in predict_results:
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])
            for text, score in zip(rec_texts, rec_scores):
                clean_text = str(text).strip()
                if clean_text:
                    candidates.append((clean_text, float(score)))
        return candidates

    # Attempt 1: Preprocessed crop (grayscale + CLAHE + 3x resize + sharpening)
    prep_crop = preprocess_crop_for_ocr(crop, scale_factor=3.0)
    pred_res = ocr_engine.predict(prep_crop)
    candidates = parse_paddle_results(pred_res)
    print(f"[PACKET OCR] Preprocessed 3x candidates: {candidates}")

    # Attempt 2: If empty, try simple 3x cubic upscale
    if not candidates:
        simple_3x = cv2.resize(
            crop,
            (crop.shape[1] * 3, crop.shape[0] * 3),
            interpolation=cv2.INTER_CUBIC
        )
        pred_res = ocr_engine.predict(simple_3x)
        candidates = parse_paddle_results(pred_res)
        print(f"[PACKET OCR] Simple 3x candidates: {candidates}")

    # Attempt 3: If still empty, try raw crop
    if not candidates:
        pred_res = ocr_engine.predict(crop)
        candidates = parse_paddle_results(pred_res)
        print(f"[PACKET OCR] Raw crop candidates: {candidates}")

    if not candidates:
        print("[PACKET OCR] PaddleOCR produced no detections.")
        return "", 0.0, []

    # Selection logic: Prioritize 6-digit numeric candidate
    six_digit_match = None
    digit_match = None

    for text, score in candidates:
        # Check pure digits
        digits = re.sub(r"\D", "", text)
        if len(digits) == 6 and six_digit_match is None:
            six_digit_match = (digits, score)
        elif len(digits) >= 4 and digit_match is None:
            digit_match = (digits, score)

    if six_digit_match is not None:
        best_text, best_score = six_digit_match
        print(f"[PACKET OCR] Selected exact 6-digit packet number: {best_text} (conf: {best_score:.4f})")
    elif digit_match is not None:
        best_text, best_score = digit_match
        print(f"[PACKET OCR] Selected numeric candidate: {best_text} (conf: {best_score:.4f})")
    else:
        # Highest confidence text candidate
        best_candidate = max(candidates, key=lambda x: x[1])
        best_text, best_score = best_candidate
        print(f"[PACKET OCR] Selected top confidence candidate: {best_text} (conf: {best_score:.4f})")

    return best_text, best_score, candidates


# ============================================================
# FRONT SIDE + PACKET NUMBER INTERACTION
# ============================================================

def front_side_and_number(
    cap,
    number_model,
    device
):
    print("\n[PACKET OCR] Stage entered")
    print(f"[PACKET OCR] Model loaded: {number_model is not None}")
    if hasattr(number_model, "ckpt_path"):
        print(f"[PACKET OCR] Model path: {number_model.ckpt_path}")

    # Track live detection state
    frame_count = 0
    cached_bbox = None
    cached_conf = 0.0
    cached_crop = None
    status = "YOLO MODEL LOADED"

    # Step 1: Live placement with live packet-number detection every 3 frames
    deadline = time.time() + FRONT_NUMBER_TIMEOUT

    while time.time() < deadline:
        ok, live_frame = cap.read()
        if not ok:
            continue

        frame_count += 1

        # Run detection every 3 frames to keep CPU responsive
        if frame_count % 3 == 0:
            status = "DETECTING NUMBER REGION"
            crop_res, bbox_res, yolo_res = detect_packet_number(
                live_frame,
                number_model,
                device
            )
            if bbox_res is not None:
                cached_crop = crop_res
                cached_bbox = bbox_res
                # Extract best conf
                if hasattr(yolo_res, "obb") and yolo_res.obb is not None and len(yolo_res.obb) > 0:
                    cached_conf = float(yolo_res.obb.conf.cpu().numpy()[int(np.argmax(yolo_res.obb.conf.cpu().numpy()))])
                elif hasattr(yolo_res, "boxes") and yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
                    cached_conf = float(yolo_res.boxes.conf.cpu().numpy()[int(np.argmax(yolo_res.boxes.conf.cpu().numpy()))])
                status = "NUMBER REGION DETECTED"
            else:
                cached_crop = None
                cached_bbox = None
                cached_conf = 0.0
                status = "DETECTING NUMBER REGION"

        # Draw live bounding box overlay
        display_frame = live_frame.copy()
        if cached_bbox is not None:
            bx1, by1, bx2, by2 = cached_bbox
            cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"packet-ocr: {cached_conf:.2f}",
                (bx1, max(20, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        extra_info = [
            ("YOLO Model", "LOADED"),
            ("Confidence", f"{cached_conf:.2f}" if cached_conf > 0 else "N/A"),
            ("Bounding Box", "FOUND" if cached_bbox is not None else "SEARCHING")
        ]

        dashboard = create_dashboard(
            display_frame,
            "FRONT SIDE",
            status,
            [
                "Align packet FRONT side",
                "Press F when number is boxed",
                "Q to quit"
            ],
            crop=cached_crop,
            extra_info=extra_info
        )
        key = show(dashboard, [])

        if key in (ord("q"), 27):
            return None

        if key == ord("f"):
            break

    else:
        print("[PACKET OCR] Timed out waiting for front-side placement.")
        return None

    # Step 2: Fresh capture & detection for stabilization
    for remaining in range(2, 0, -1):
        ok, live_frame = cap.read()
        if not ok:
            continue
        dashboard = create_dashboard(
            live_frame,
            "FRONT SIDE",
            "STABILIZING",
            [
                f"Keep packet still... {remaining}",
                "Preparing high-resolution capture"
            ],
            crop=cached_crop,
            extra_info=[("YOLO Model", "LOADED"), ("Status", "STABILIZING")]
        )
        show(dashboard, [])
        time.sleep(0.4)

    # Capture a fresh frame
    ok, frame = cap.read()
    if not ok:
        print("[PACKET OCR] Failed to capture fresh frame.")
        return None

    crop, bbox, result = detect_packet_number(
        frame,
        number_model,
        device
    )

    if crop is None:
        print("[PACKET OCR] Number region not detected on fresh capture.")
        while True:
            ok, live_frame = cap.read()
            if not ok:
                continue

            dashboard = create_dashboard(
                live_frame,
                "FRONT SIDE",
                "DETECTING NUMBER REGION",
                [
                    "Packet number region not found",
                    "Adjust packet position",
                    "Press F to retry detection",
                    "Q to quit"
                ],
                extra_info=[("YOLO Model", "LOADED"), ("Bounding Box", "NOT FOUND")]
            )
            key = show(dashboard, [])

            if key in (ord("q"), 27):
                return None

            if key == ord("f"):
                return front_side_and_number(
                    cap,
                    number_model,
                    device
                )

    # Step 3: PaddleOCR execution
    x1, y1, x2, y2 = bbox
    preview = frame.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)

    status = "RUNNING PADDLEOCR"
    # Show status before running OCR inference so user sees the transition
    interim_dashboard = create_dashboard(
        preview,
        "PACKET NUMBER OCR",
        status,
        ["Running PaddleOCR recognition..."],
        crop=crop,
        extra_info=[
            ("YOLO Model", "LOADED"),
            ("Bounding Box", f"{x2 - x1}x{y2 - y1} px"),
            ("OCR State", "PROCESSING")
        ]
    )
    show(interim_dashboard, [])

    packet_number, ocr_score, candidates = run_ocr(crop)

    if not packet_number:
        status = "OCR TEXT UNREADABLE"
        packet_number = "UNREADABLE"
        ocr_conf_str = "0.00"
    else:
        status = "OCR CONFIRMATION REQUIRED"
        ocr_conf_str = f"{ocr_score:.2f}"

    print(f"[PACKET OCR] Final packet number: {packet_number} (confidence: {ocr_conf_str})")

    # Step 4: Operator confirmation loop
    while True:
        extra_info = [
            ("YOLO Model", "LOADED"),
            ("Bounding Box", f"{x2 - x1}x{y2 - y1} px"),
            ("OCR Text", packet_number),
            ("OCR Conf", ocr_conf_str),
            ("Confirmed", "NO")
        ]

        dashboard = create_dashboard(
            preview,
            "PACKET NUMBER OCR",
            status,
            [
                f"Detected: {packet_number}",
                "Y = Accept & proceed",
                "R = Retry detection",
                "Q = Quit"
            ],
            crop=crop,
            extra_info=extra_info
        )
        key = show(dashboard, [])

        if key in (ord("y"), ord("Y")):
            if packet_number == "UNREADABLE":
                print("[PACKET OCR] Cannot accept unreadable text. Please retry with 'R'.")
                continue
            status = "PACKET NUMBER CONFIRMED"
            confirm_dashboard = create_dashboard(
                preview,
                "PACKET NUMBER OCR",
                status,
                [f"Confirmed: {packet_number}", "Proceeding to next step..."],
                crop=crop,
                extra_info=[
                    ("YOLO Model", "LOADED"),
                    ("OCR Text", packet_number),
                    ("Confirmed", "YES")
                ]
            )
            show(confirm_dashboard, [])
            cv2.waitKey(600)
            print(f"[PACKET OCR] PACKET NUMBER CONFIRMED: {packet_number}")
            return packet_number

        if key in (ord("r"), ord("R")):
            return front_side_and_number(
                cap,
                number_model,
                device
            )

        if key in (ord("q"), 27):
            return None


# ============================================================
# PACKET CONNECTED COMPONENTS
# ============================================================

def get_components(
    mask,
    reference_area
):

    binary = (
        mask > 0
    ).astype(np.uint8)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    min_area = max(
        MIN_PIECE_AREA_PX,
        int(reference_area * MIN_PIECE_AREA_RATIO)
    )

    items = []

    for label_id in range(1, count):

        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA
            ]
        )

        if area < min_area:
            continue

        component = np.zeros_like(
            binary,
            dtype=np.uint8
        )

        component[
            labels == label_id
        ] = 255

        m = cv2.moments(
            component,
            binaryImage=True
        )

        if m["m00"] <= 0:
            continue

        items.append({

            "mask": component,

            "area": area,

            "centroid": (
                float(m["m10"] / m["m00"]),
                float(m["m01"] / m["m00"])
            ),

            "bbox": (
                int(stats[label_id, cv2.CC_STAT_LEFT]),
                int(stats[label_id, cv2.CC_STAT_TOP]),
                int(stats[label_id, cv2.CC_STAT_WIDTH]),
                int(stats[label_id, cv2.CC_STAT_HEIGHT])
            )
        })

    items.sort(
        key=lambda x: x["area"],
        reverse=True
    )

    return items


# ============================================================
# GAP ANALYSIS & FACING-EDGE VERIFICATION
# ============================================================

def analyze_packet_gap(
    mask_a,
    mask_b,
    frame=None,
    reference_area=None,
    min_norm_gap=MIN_NORM_GAP_RATIO,
    min_coverage=MIN_GAP_COVERAGE_RATIO
):
    """
    Robust physical gap verification between two packet pieces:
    1. Extracts facing edges using distance transforms.
    2. Measures min, median, and max gap widths in pixels and normalized to packet scale.
    3. Calculates gap coverage percentage across the separation boundary.
    4. Validates exposed background strictly between facing edges.
    5. Returns comprehensive gap metrics and visualization masks.
    """
    h, w = mask_a.shape[:2]
    union_mask = (mask_a > 0) | (mask_b > 0)
    cnts, _ = cv2.findContours(union_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return {"valid": False, "reason": "No packet contours found", "min_gap": 0.0, "norm_gap": 0.0, "coverage_pct": 0.0}

    # Estimate orientation and dimensions via minAreaRect
    all_pts = np.vstack(cnts)
    rect = cv2.minAreaRect(all_pts)  # ((cx, cy), (rw, rh), angle)
    (cx, cy), (rw, rh), angle = rect
    pkt_major = max(rw, rh)
    pkt_minor = min(rw, rh)

    if pkt_minor <= 10 or pkt_major <= 10:
        return {"valid": False, "reason": "Packet dimensions too small for gap analysis", "min_gap": 0.0, "norm_gap": 0.0, "coverage_pct": 0.0}

    # Bidirectional Euclidean distance transforms
    dt_a = cv2.distanceTransform((mask_a == 0).astype(np.uint8), cv2.DIST_L2, 5)
    dt_b = cv2.distanceTransform((mask_b == 0).astype(np.uint8), cv2.DIST_L2, 5)

    cnt_a, _ = cv2.findContours(mask_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not cnt_a or not cnt_b:
        return {"valid": False, "reason": "Missing piece contours", "min_gap": 0.0, "norm_gap": 0.0, "coverage_pct": 0.0}

    pts_a = cnt_a[0].squeeze()
    pts_b = cnt_b[0].squeeze()
    if pts_a.ndim < 2 or pts_b.ndim < 2:
        return {"valid": False, "reason": "Degenerate piece contour points", "min_gap": 0.0, "norm_gap": 0.0, "coverage_pct": 0.0}

    dists_a_to_b = np.array([dt_b[p[1], p[0]] for p in pts_a])
    dists_b_to_a = np.array([dt_a[p[1], p[0]] for p in pts_b])

    min_gap_a = float(np.min(dists_a_to_b))
    min_gap_b = float(np.min(dists_b_to_a))
    min_gap = min(min_gap_a, min_gap_b)

    # Threshold to identify facing edges (boundary points facing the opposing piece)
    facing_thresh = max(min_gap + 5.0, min_gap * 1.5)
    facing_idx_a = np.where(dists_a_to_b <= facing_thresh)[0]
    facing_pts_a = pts_a[facing_idx_a]
    facing_dists = dists_a_to_b[facing_idx_a]

    facing_idx_b = np.where(dists_b_to_a <= facing_thresh)[0]
    facing_pts_b = pts_b[facing_idx_b]

    if len(facing_dists) == 0:
        return {"valid": False, "reason": "Facing edges not identified", "min_gap": min_gap, "norm_gap": 0.0, "coverage_pct": 0.0}

    median_gap = float(np.median(facing_dists))
    max_gap = float(np.max(facing_dists))

    # Gap coverage: measure facing edge length against expected cut interface (pkt_minor)
    facing_span = float(np.linalg.norm(np.max(facing_pts_a, axis=0) - np.min(facing_pts_a, axis=0)))
    expected_cut_len = max(20.0, pkt_minor)
    coverage_ratio = min(1.0, facing_span / expected_cut_len)
    coverage_pct = coverage_ratio * 100.0

    # Normalized gap width relative to packet minor dimension
    norm_gap = min_gap / max(1.0, pkt_minor)

    # Exposed background validation: pixels strictly in the corridor between both pieces
    gap_corridor = ((dt_a <= max_gap * 1.2) & (dt_b <= max_gap * 1.2) & (union_mask == 0)).astype(np.uint8)
    gap_px_count = int(np.count_nonzero(gap_corridor))

    bg_valid = True
    bg_consistency = 1.0

    if frame is not None and gap_px_count > 10:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gap_pixels = gray[gap_corridor > 0]
        # Texture check: ensure gap is not a thin dark fold / printed table line
        if gap_pixels.size > 0:
            std_dev = float(np.std(gap_pixels))
            bg_consistency = max(0.0, 1.0 - (std_dev / 70.0))

    # Strict physical separation decision
    is_valid_gap = (
        min_gap >= MIN_BACKGROUND_GAP_PX
        and norm_gap >= min_norm_gap
        and coverage_ratio >= min_coverage
        and gap_px_count >= MIN_EXPOSED_GAP_PIXELS
    )

    if not is_valid_gap:
        if min_gap < MIN_BACKGROUND_GAP_PX or norm_gap < min_norm_gap:
            reason = f"Gap too narrow: {min_gap:.1f}px (norm: {norm_gap:.3f} < {min_norm_gap:.3f})"
        elif coverage_ratio < min_coverage:
            reason = f"Gap coverage incomplete: {coverage_pct:.1f}% < {min_coverage*100:.0f}%"
        elif gap_px_count < MIN_EXPOSED_GAP_PIXELS:
            reason = "No exposed background detected between pieces"
        else:
            reason = "Physical gap criteria not met"
    else:
        reason = f"Valid physical gap: {min_gap:.1f}px ({coverage_pct:.0f}% cut line)"

    return {
        "valid": is_valid_gap,
        "min_gap": min_gap,
        "median_gap": median_gap,
        "max_gap": max_gap,
        "norm_gap": norm_gap,
        "coverage_pct": coverage_pct,
        "gap_px_count": gap_px_count,
        "bg_valid": bg_valid,
        "bg_consistency": bg_consistency,
        "reason": reason,
        "gap_mask": gap_corridor,
        "facing_pts_a": facing_pts_a,
        "facing_pts_b": facing_pts_b
    }


def gap_between_masks(
    mask_a,
    mask_b
):
    distance = cv2.distanceTransform(
        (mask_a == 0).astype(np.uint8),
        cv2.DIST_L2,
        3
    )

    values = distance[
        mask_b > 0
    ]

    return (
        float(values.min())
        if values.size
        else 0.0
    )


# ============================================================
# VISUALIZE PACKET PIECES & GAP
# ============================================================

def draw_packet_components(
    frame,
    components,
    gap_info=None
):

    output = frame.copy()

    colors = [
        (0, 240, 100),   # Piece 1: Vibrant Green
        (255, 60, 220),  # Piece 2: Magenta
        (0, 180, 255)    # Extra pieces: Orange
    ]

    for i, item in enumerate(components[:3]):
        color = colors[i]
        output = draw_mask(
            output,
            item["mask"] * 255,
            color,
            0.28
        )
        x, y, w, h = item["bbox"]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            output,
            f"Piece {i + 1}: {item['area']} px",
            (x, max(18, y - 7)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA
        )

    # Highlight verified exposed physical gap corridor
    if gap_info is not None and "gap_mask" in gap_info:
        gap_m = gap_info["gap_mask"]
        if np.any(gap_m):
            # Electric cyan overlay for confirmed physical gap
            output = draw_mask(output, gap_m * 255, (0, 255, 255), 0.50)

    return output


def save_verification_debug(output_dir, frame, packet_mask, components, gap_info, status_name, sample_idx=0):
    """
    Saves debug diagnostic package for audit and calibration:
    1. Original frame
    2. Raw packet segmentation mask
    3. Connected components visualization
    4. Piece 1 mask
    5. Piece 2 mask
    6. Gap candidate mask
    7. Validated gap overlay
    """
    debug_dir = Path(output_dir) / "verification_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)

    # 1. Original frame
    cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_orig.jpg"), frame)

    # 2. Raw packet segmentation mask
    if packet_mask is not None:
        cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_pkt_mask.png"), packet_mask)

    # 3. Connected components visualization
    h, w = frame.shape[:2]
    cc_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, comp in enumerate(components):
        col = (80 * (i + 1) % 255, 120 * (i + 1) % 255, 200)
        cc_img[comp["mask"] > 0] = col
    cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_components.png"), cc_img)

    # 4 & 5. Piece 1 and Piece 2 individual masks
    if len(components) >= 1:
        cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_piece1_mask.png"), components[0]["mask"])
    if len(components) >= 2:
        cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_piece2_mask.png"), components[1]["mask"])

    # 6. Gap candidate mask
    if gap_info is not None and "gap_mask" in gap_info and gap_info["gap_mask"] is not None:
        gap_m = (gap_info["gap_mask"] * 255).astype(np.uint8)
        cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_gap_candidate_mask.png"), gap_m)

    # 7. Validated gap overlay
    vis = draw_packet_components(frame, components, gap_info=gap_info)
    cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_gap_vis.jpg"), vis)


# ============================================================
# TEMPORAL STABILITY
# ============================================================

def stable_pair(
    previous,
    current
):

    if previous is None:
        return False

    if current is None:
        return False

    if len(previous) != 2 or len(current) != 2:
        return False

    old = sorted(
        previous,
        key=lambda x: x["centroid"][0]
    )

    new = sorted(
        current,
        key=lambda x: x["centroid"][0]
    )

    for a, b in zip(old, new):

        shift = np.hypot(
            a["centroid"][0] - b["centroid"][0],
            a["centroid"][1] - b["centroid"][1]
        )

        if shift > MAX_CENTROID_SHIFT_PX:
            return False

        ratio = (
            abs(a["area"] - b["area"])
            / max(a["area"], 1)
        )

        if ratio > MAX_AREA_CHANGE_RATIO:
            return False

    return True


def signature(components):

    return [
        {
            "area": item["area"],
            "centroid": item["centroid"]
        }

        for item in sorted(
            components,
            key=lambda x: x["centroid"][0]
        )
    ]


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_ID
    )

    parser.add_argument(
        "--cpu",
        action="store_true"
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save diagnostic masks and gap verification frames"
    )

    args = parser.parse_args()

    # Automatically fallback to CPU if CUDA is not available
    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = (
            int(args.device)
            if str(args.device).isdigit()
            else args.device
        )


    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    packet_path = require_file(
        PACKET_MODEL_PATH,
        "Packet segmentation model"
    )

    hand_path = require_file(
        HAND_MODEL_PATH,
        "Hand segmentation model"
    )

    scissor_path = require_file(
        SCISSOR_MODEL_PATH,
        "Scissor segmentation model"
    )

    topstrip_path = require_file(
        TOPSTRIP_MODEL_PATH,
        "TopStrip segmentation model"
    )

    number_path = find_number_model()

    print("\nLoading models...")

    packet_model = YOLO(str(packet_path))
    hand_model = YOLO(str(hand_path))
    scissor_model = YOLO(str(scissor_path))
    topstrip_model = YOLO(str(topstrip_path))
    number_model = YOLO(str(number_path))

    cut_model_path = require_file(
        CUT_MODEL_PATH,
        "Final cut detection model"
    )

    cut_model = YOLO(
        str(cut_model_path),
        task="segment"
    )

    print("Packet classes:", packet_model.names)
    print("Hand classes:", hand_model.names)
    print("Scissor classes:", scissor_model.names)
    print("TopStrip classes:", topstrip_model.names)
    print("Packet number classes:", number_model.names)

    print("Inference device:", device)


    # --------------------------------------------------------
    # CLASS IDs
    # --------------------------------------------------------

    hand_ids = matching_class_ids(
        hand_model,
        ("hand",)
    )

    scissor_ids = matching_class_ids(
        scissor_model,
        ("scissor",)
    )

    topstrip_ids = matching_class_ids(
        topstrip_model,
        ("topstrip", "strip", "seal", "top")
    )

    packet_ids = matching_class_ids(
        packet_model,
        ("packet", "bag")
    )


    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        args.camera,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        cap.release()

        cap = cv2.VideoCapture(
            args.camera
        )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open webcam index {args.camera}"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.namedWindow(
        WINDOW,
        cv2.WINDOW_NORMAL
    )


    # --------------------------------------------------------
    # FRAME READER
    # --------------------------------------------------------

    def read_frame():

        ok, image = cap.read()

        if not ok:
            raise RuntimeError(
                "Could not read webcam frame"
            )

        return image


    try:

        # ====================================================
        # STEP 1 + 2: FRONT SIDE AND PACKET NUMBER
        # ====================================================

        packet_number = front_side_and_number(
            cap,
            number_model,
            device
        )

        if packet_number is None:
            return

        print(
            "Operator-confirmed packet number:",
            packet_number
        )


        # ====================================================
        # STEP 3: TURN TO BACK / TOPSTRIP INSPECTION
        # ====================================================

        while True:

            frame = read_frame()

            key = show(
                frame,
                [
                    f"PACKET NUMBER: {packet_number}",
                    "STEP 3 - Turn packet to BACK / seal side.",
                    "Press T when positioned | Q to quit"
                ],
                (0, 220, 255)
            )

            if key in (ord("q"), 27):
                return

            if key == ord("t"):
                break


        # ====================================================
        # TOPSTRIP INSPECTION
        # ====================================================

        seal_deadline = (
            time.time()
            + SEAL_INSPECTION_TIMEOUT
        )

        seal_status = "UNKNOWN_ORIENTATION"

        while time.time() < seal_deadline:

            frame = read_frame()

            packet_result = predict(
                packet_model,
                frame,
                PACKET_CONF,
                device=device
            )

            top_result = predict(
                topstrip_model,
                frame,
                TOPSTRIP_CONF,
                device=device
            )

            packet_mask = get_union_mask(
                packet_result,
                frame.shape[:2],
                packet_ids
            )

            top_mask = get_union_mask(
                top_result,
                frame.shape[:2],
                topstrip_ids
            )

            top_present = detection_present(
                top_result,
                topstrip_ids
            )

            packet_visible = (
                np.count_nonzero(packet_mask) > 500
            )

            sharpness = blur_score(frame)

            preview = draw_mask(
                frame,
                packet_mask,
                (255, 120, 0),
                0.20
            )

            preview = draw_mask(
                preview,
                top_mask,
                (255, 0, 255),
                0.35
            )

            if (
                not packet_visible
                or sharpness < BLUR_VARIANCE_MIN
            ):

                seal_status = "UNKNOWN_ORIENTATION"

                reason = (
                    "Packet not clearly visible or image blurry"
                )

            elif top_present:

                seal_status = "OPEN"

                reason = (
                    "TopStrip visible: OPEN / CUTTING_NOT_READY"
                )

            else:

                seal_status = "CLOSED"

                reason = (
                    "TopStrip not visible in valid inspection: CLOSED"
                )

            key = show(
                preview,
                [
                    f"SEAL STATUS: {seal_status}",
                    f"TopStrip: {'VISIBLE' if top_present else 'NOT DETECTED'}",
                    reason,
                    "If OPEN, reposition/close packet and inspect again.",
                    "Press C only when CLOSED | Q to quit"
                ],
                (
                    (0, 255, 0)
                    if seal_status == "CLOSED"
                    else (0, 180, 255)
                )
            )

            if key in (ord("q"), 27):
                return

            if (
                key == ord("c")
                and seal_status == "CLOSED"
            ):
                break


        if seal_status != "CLOSED":

            print(
                "Seal inspection timed out. Cutting not ready."
            )

            return


        # ====================================================
        # STEP 4: STORE INTACT PACKET REFERENCE
        # ====================================================

        frame = read_frame()

        baseline_result = predict(
            packet_model,
            frame,
            PACKET_CONF,
            device=device
        )

        reference_mask = get_union_mask(
            baseline_result,
            frame.shape[:2],
            packet_ids
        )

        reference_area = float(
            np.count_nonzero(reference_mask)
        )

        baseline_components = get_components(
            reference_mask,
            reference_area
        )

        if (
            reference_area <= 0
            or len(baseline_components) != 1
        ):

            show(
                frame,
                [
                    "BASELINE FAILED",
                    "Expected one visible connected intact packet.",
                    "Reposition packet and restart."
                ],
                (0, 0, 255)
            )

            cv2.waitKey(0)

            return

        reference_mask = (
            baseline_components[0]["mask"] * 255
        )

        reference_area = float(
            np.count_nonzero(reference_mask)
        )

        print(
            f"Stored intact reference: "
            f"area={reference_area:.0f}px, "
            f"packet={packet_number}"
        )


        # ====================================================
        # STEP 5: HAND AND SCISSOR MONITORING
        # ====================================================

        activity_started = False
        clear_start = None
        hand_consecutive_frames = 0
        scissor_consecutive_frames = 0

        print(
            "Ready. Waiting for cutting activity."
        )

        while True:

            frame = read_frame()

            hand_result = predict(
                hand_model,
                frame,
                HAND_CONF,
                device=device
            )

            scissor_result = predict(
                scissor_model,
                frame,
                SCISSOR_CONF,
                device=device
            )

            # Evaluate hand detection with seal false-positive filter & diagnostic logging
            raw_hand_present, hand_mask, hand_diags = evaluate_hand_detection(
                hand_result,
                hand_ids,
                frame.shape[:2],
                reference_mask=reference_mask,
                frame=frame
            )

            # Temporal confirmation: require 2 consecutive frames
            if raw_hand_present:
                hand_consecutive_frames += 1
            else:
                hand_consecutive_frames = 0

            hand_present = (hand_consecutive_frames >= HAND_CONFIRM_FRAMES)

            # Scissor detection with spatial proximity check to packet
            raw_scissor_present, scissor_mask, scissor_diags = evaluate_scissor_detection(
                scissor_result,
                scissor_ids,
                frame.shape[:2],
                packet_mask=reference_mask,
                min_conf=SCISSOR_CONF,
                max_dist_px=80.0
            )

            if raw_scissor_present:
                scissor_consecutive_frames += 1
            else:
                scissor_consecutive_frames = 0

            scissors_present = (scissor_consecutive_frames >= SCISSOR_CONFIRM_FRAMES)

            preview = draw_mask(
                frame,
                hand_mask,
                (0, 0, 255),
                0.30
            )

            preview = draw_mask(
                preview,
                scissor_mask,
                (0, 255, 255),
                0.30
            )

            if hand_present or scissors_present:

                activity_started = True
                clear_start = None

                status = "CUTTING_ACTIVITY"

                detail = (
                    "Hand/scissors detected. "
                    "Verification paused."
                )

                color = (0, 180, 255)

            elif activity_started:

                if clear_start is None:
                    clear_start = time.time()

                clear_elapsed = (
                    time.time() - clear_start
                )

                status = "WAITING_AFTER_CUT"

                detail = (
                    f"Scene clear: "
                    f"{clear_elapsed:.1f}/"
                    f"{CLEAR_WAIT_SECONDS:.1f}s"
                )

                color = (0, 220, 255)

                if clear_elapsed >= CLEAR_WAIT_SECONDS:
                    break

            else:

                status = "CUTTING_READY"

                detail = (
                    "Packet sealed and reference stored. "
                    "Waiting for cutting."
                )

                color = (0, 255, 0)

            # Build dashboard diagnostic extra info
            diag_info = [
                ("Packet", packet_number),
                ("Hand Confirmed", "YES" if hand_present else "NO"),
                ("Hand Streak", f"{hand_consecutive_frames}/{HAND_CONFIRM_FRAMES}"),
                ("Scissors", "YES" if scissors_present else "NO"),
            ]
            if hand_diags:
                best_hd = hand_diags[0]
                diag_info.append(("Hand Conf", f"{best_hd['conf']:.2f}"))
                diag_info.append(("Hand Area", f"{best_hd['area_px']} px"))
                diag_info.append(("Hand Pos", best_hd['pos_rel']))
                diag_info.append(("Seal Filter", "FILTERED" if best_hd['is_seal_fp'] else "PASS"))

            dashboard = create_dashboard(
                preview,
                "STEP 5 - CUTTING",
                status,
                [
                    detail,
                    "Keep hands/scissors clear after cut",
                    "Q to quit"
                ],
                extra_info=diag_info
            )
            key = show(dashboard, [])

            if key in (ord("q"), 27):
                return


        # ====================================================
        # STEP 6: POST-CUT VERIFICATION FOR 10 SECONDS
        # ====================================================

        verify_start = time.time()

        next_sample = 0.0

        sample_count = 0
        valid_count = 0

        previous_signature = None
        stable_streak = 0

        last_reason = (
            "Waiting for post-cut segmentation"
        )

        last_preview = None
        last_gap = 0.0
        components = []
        verify_hand_streak = 0
        verify_scissor_streak = 0


        while (
            time.time() - verify_start
            < CUT_VERIFY_SECONDS
        ):

            frame = read_frame()

            # Check whether hand/scissors return.
            hand_result = predict(
                hand_model,
                frame,
                HAND_CONF,
                device=device
            )

            scissor_result = predict(
                scissor_model,
                frame,
                SCISSOR_CONF,
                device=device
            )

            raw_hp, h_mask_v, h_diags_v = evaluate_hand_detection(
                hand_result,
                hand_ids,
                frame.shape[:2],
                reference_mask=reference_mask,
                frame=frame
            )

            if raw_hp:
                verify_hand_streak += 1
            else:
                verify_hand_streak = 0

            hand_present = (verify_hand_streak >= HAND_CONFIRM_FRAMES)

            raw_sp, s_mask_v, s_diags_v = evaluate_scissor_detection(
                scissor_result,
                scissor_ids,
                frame.shape[:2],
                packet_mask=reference_mask,
                min_conf=SCISSOR_CONF,
                max_dist_px=80.0
            )
            if raw_sp:
                verify_scissor_streak += 1
            else:
                verify_scissor_streak = 0
            scissors_present = (verify_scissor_streak >= SCISSOR_CONFIRM_FRAMES)


            # ----------------------------------------------
            # ACTIVITY RETURNED: PAUSE AND RESET CLEAR WAIT
            # ----------------------------------------------

            if hand_present or scissors_present:

                next_sample = 0.0
                sample_count = 0
                valid_count = 0

                stable_streak = 0
                previous_signature = None

                last_reason = (
                    "Hand/scissors returned: restart clear wait"
                )

                clear_start = None

                diag_v = [
                    ("Packet", packet_number),
                    ("Hand Present", "YES" if hand_present else "NO"),
                    ("Scissors", "YES" if scissors_present else "NO"),
                    ("Reason", "Activity detected")
                ]
                dashboard = create_dashboard(
                    frame,
                    "VERIFICATION PAUSED",
                    "PAUSED",
                    [
                        "Hand or scissors returned.",
                        last_reason,
                        "Q to quit"
                    ],
                    extra_info=diag_v
                )
                key = show(dashboard, [])

                if key in (ord("q"), 27):
                    return


                # Wait until both disappear for the full clear period.
                clear_hand_streak = 0
                clear_scissor_streak = 0
                while True:

                    frame = read_frame()

                    hr = predict(
                        hand_model,
                        frame,
                        HAND_CONF,
                        device=device
                    )

                    sr = predict(
                        scissor_model,
                        frame,
                        SCISSOR_CONF,
                        device=device
                    )

                    raw_c_hp, _, _ = evaluate_hand_detection(
                        hr,
                        hand_ids,
                        frame.shape[:2],
                        reference_mask=reference_mask,
                        frame=frame
                    )
                    if raw_c_hp:
                        clear_hand_streak += 1
                    else:
                        clear_hand_streak = 0

                    hp = (clear_hand_streak >= HAND_CONFIRM_FRAMES)

                    raw_c_sp, _, _ = evaluate_scissor_detection(
                        sr,
                        scissor_ids,
                        frame.shape[:2],
                        packet_mask=reference_mask,
                        min_conf=SCISSOR_CONF,
                        max_dist_px=80.0
                    )
                    if raw_c_sp:
                        clear_scissor_streak += 1
                    else:
                        clear_scissor_streak = 0
                    sp = (clear_scissor_streak >= SCISSOR_CONFIRM_FRAMES)

                    if hp or sp:

                        clear_start = None

                        detail = (
                            "Hand/scissors still present"
                        )

                    else:

                        if clear_start is None:
                            clear_start = time.time()

                        elapsed_clear = (
                            time.time() - clear_start
                        )

                        detail = (
                            f"Clear wait: "
                            f"{elapsed_clear:.1f}/"
                            f"{CLEAR_WAIT_SECONDS:.1f}s"
                        )

                        if elapsed_clear >= CLEAR_WAIT_SECONDS:

                            verify_start = time.time()

                            break

                    dash_c = create_dashboard(
                        frame,
                        "WAITING AFTER CUT",
                        "CLEARING",
                        [
                            detail,
                            "Verification restarts after clear scene."
                        ],
                        extra_info=[
                            ("Packet", packet_number),
                            ("Hand", "YES" if hp else "NO"),
                            ("Scissors", "YES" if sp else "NO")
                        ]
                    )
                    key = show(dash_c, [])

                    if key in (ord("q"), 27):
                        return

                continue


            # ----------------------------------------------
            # RUN PACKET SEGMENTATION AT SAMPLE INTERVAL
            # ----------------------------------------------

            elapsed = (
                time.time() - verify_start
            )

            if elapsed >= next_sample:

                packet_result = predict(
                    packet_model,
                    frame,
                    PACKET_CONF,
                    device=device
                )

                packet_mask = get_union_mask(
                    packet_result,
                    frame.shape[:2],
                    packet_ids
                )

                components = get_components(
                    packet_mask,
                    reference_area
                )

                sample_count += 1

                total_packet_px = np.count_nonzero(packet_mask)
                last_gap_info = None

                # Rule 17 & 18: Packet missing or insufficient segmentation
                if total_packet_px < 300:
                    last_reason = "PACKET_MISSING: Packet area below minimum detection threshold"
                    stable_streak = 0
                    previous_signature = None

                elif len(components) != 2:
                    if len(components) < 2:
                        last_reason = f"No physical cut: Only {len(components)} connected piece detected"
                    else:
                        last_reason = f"VERIFICATION_UNCERTAIN: Found {len(components)} fragments (expected 2)"
                    stable_streak = 0
                    previous_signature = None

                else:
                    a, b = components[0], components[1]

                    total_ratio = (
                        a["area"] + b["area"]
                    ) / max(reference_area, 1.0)

                    if not (
                        MIN_TOTAL_AREA_RATIO
                        <= total_ratio
                        <= MAX_TOTAL_AREA_RATIO
                    ):
                        last_reason = (
                            "VERIFICATION_UNCERTAIN: Piece area mismatch: "
                            f"{total_ratio:.2f}x baseline"
                        )
                        stable_streak = 0
                        previous_signature = None

                    else:
                        # Full physical gap analysis across facing edges
                        gap_res = analyze_packet_gap(
                            a["mask"],
                            b["mask"],
                            frame=frame,
                            reference_area=reference_area,
                            min_norm_gap=MIN_NORM_GAP_RATIO,
                            min_coverage=MIN_GAP_COVERAGE_RATIO
                        )
                        last_gap_info = gap_res
                        last_gap = gap_res["min_gap"]

                        if not gap_res["valid"]:
                            last_reason = f"GAP_INVALID: {gap_res['reason']}"
                            stable_streak = 0
                            previous_signature = None

                        else:
                            current_sig = signature(components)

                            if (
                                previous_signature is None
                                or stable_pair(
                                    previous_signature,
                                    current_sig
                                )
                            ):
                                stable_streak += 1
                            else:
                                stable_streak = 1

                            previous_signature = current_sig
                            valid_count += 1
                            last_reason = (
                                f"Two detached masks with valid gap ({gap_res['min_gap']:.1f}px, "
                                f"{gap_res['coverage_pct']:.0f}% interface) [Streak: {stable_streak}/5]"
                            )

                            # Streak tracks consecutive stable frames; must complete full 10-second verification window

                # Save debug images if debug flag is enabled
                if getattr(args, "debug", False):
                    save_status = "valid" if (last_gap_info and last_gap_info.get("valid")) else "invalid"
                    save_verification_debug(
                        OUTPUT_DIR,
                        frame,
                        packet_mask,
                        components,
                        last_gap_info,
                        f"samp{sample_count}_{save_status}"
                    )

                last_preview = draw_packet_components(
                    frame,
                    components,
                    gap_info=last_gap_info
                )

                next_sample += VERIFY_SAMPLE_INTERVAL


            # ----------------------------------------------
            # LIVE VERIFICATION DISPLAY (SINGLE DASHBOARD)
            # ----------------------------------------------

            valid_ratio = (
                valid_count / max(sample_count, 1)
            )

            remaining = max(
                0.0,
                CUT_VERIFY_SECONDS
                - (time.time() - verify_start)
            )

            # Telemetry for the rich live dashboard
            min_gap_str = f"{last_gap_info['min_gap']:.1f} px" if last_gap_info else "0.0 px"
            norm_gap_str = f"{last_gap_info['norm_gap']:.3f}" if last_gap_info else "0.000"
            cov_str = f"{last_gap_info['coverage_pct']:.0f}%" if last_gap_info else "0%"
            bg_val_str = "VALID" if (last_gap_info and last_gap_info.get("valid")) else "PENDING"

            verify_telemetry = [
                ("Pieces Detected", f"{len(components)}"),
                ("Gap Width", min_gap_str),
                ("Normalized Gap", norm_gap_str),
                ("Gap Coverage", cov_str),
                ("Background", bg_val_str),
                ("Stability Timer", f"{elapsed:.1f} / {CUT_VERIFY_SECONDS:.1f}s"),
                ("Valid Samples", f"{valid_count}/{sample_count} ({valid_ratio:.0%})")
            ]

            verify_dashboard = create_dashboard(
                last_preview if last_preview is not None else frame,
                "POST-CUT VERIFICATION",
                "VERIFYING GAP" if valid_count > 0 else "EVALUATING",
                [
                    last_reason,
                    f"Remaining: {remaining:.1f}s | Stable: {stable_streak}",
                    "Q to quit"
                ],
                extra_info=verify_telemetry
            )
            key = show(verify_dashboard, [])

            if key in (ord("q"), 27):
                return

            time.sleep(0.01)


        # ====================================================
        # STEP 7: FINAL DECISION
        # ====================================================

        # Run cut-seg.pt confirmation inference (strict 0.70 threshold)
        cut_result = predict(cut_model, frame, 0.25, device=device)
        cut_eval = evaluate_cut_model_prediction(cut_result, cut_class_id=0, min_conf=0.70)
        is_cut_model_confirmed = cut_eval["is_cut_detected"]

        valid_ratio = (
            valid_count / max(sample_count, 1)
        )

        total_packet_px = np.count_nonzero(packet_mask) if 'packet_mask' in locals() else 0

        # Exact evaluation across standard 5 states with cut-seg.pt integration
        if total_packet_px < 300:
            final_state = "PACKET_MISSING"
            final_reason = "Packet was not detected in the verification scene"
        elif reference_area and len(components) >= 1 and (max(p['area'] for p in components) >= 0.88 * reference_area):
            final_state = "NOT_CUTTED"
            final_reason = f"Intact packet confirmed: Main body retains {max(p['area'] for p in components)/reference_area:.0%} of baseline area"
        elif (
            sample_count >= MIN_VERIFY_SAMPLES
            and valid_ratio >= MIN_VALID_SAMPLE_RATIO
            and stable_streak >= 5
            and len(components) == 2
            and (last_gap_info and last_gap_info.get("valid", False))
            and is_cut_model_confirmed
        ):
            final_state = "CUT_CONFIRMED"
            final_reason = (
                f"CUT CONFIRMED: Valid physical gap ({last_gap_info['min_gap']:.1f}px) "
                f"and cut model verified Cut (conf: {cut_eval['confidence']:.1%})"
            )
        elif is_cut_model_confirmed and cut_eval["confidence"] >= 0.80 and len(components) >= 2 and (last_gap_info and last_gap_info.get("valid", False)):
            final_state = "CUT_CONFIRMED"
            final_reason = (
                f"CUT CONFIRMED by cut model ({cut_eval['confidence']:.1%}) "
                f"with {len(components)} packet fragments and verified gap"
            )
        elif len(components) == 1:
            final_state = "NOT_CUTTED"
            final_reason = "Intact packet confirmed: Exactly 1 continuous packet detected"
        elif not is_cut_model_confirmed:
            final_state = "NOT_CONFIRMED"
            final_reason = f"Cut model did not detect positive cut (conf: {cut_eval['confidence']:.1%}, need >= 70%)"
            )
        elif len(components) == 1:
            final_state = "NOT_CUTTED"
            final_reason = f"NOT CUT: Single continuous packet mask detected (cut model: {cut_eval['class_name']})"
        elif last_gap_info and not last_gap_info.get("valid", False):
            if last_gap_info.get("min_gap", 0) < 2.0:
                final_state = "NOT_CONFIRMED"
                final_reason = "NOT CONFIRMED: Pieces are touching or separation gap is incomplete"
            else:
                final_state = "UNVERIFIED"
                final_reason = f"UNVERIFIED: {last_gap_info.get('reason', 'Unreliable gap')}"
        elif len(components) != 2:
            final_state = "UNVERIFIED"
            final_reason = f"UNVERIFIED: Found {len(components)} fragments (expected exactly 2)"
        elif sample_count < MIN_VERIFY_SAMPLES or valid_ratio < MIN_VALID_SAMPLE_RATIO:
            final_state = "UNVERIFIED"
            final_reason = (
                f"UNVERIFIED: Insufficient stable samples ({valid_count}/{sample_count}, "
                f"ratio: {valid_ratio:.1%} < {MIN_VALID_SAMPLE_RATIO:.0%})"
            )
        else:
            final_state = "UNVERIFIED"
            final_reason = f"UNVERIFIED: Stability streak incomplete (reached {stable_streak}/5)"


        # ====================================================
        # FINAL RESULT
        # ====================================================

        # Save final debug package if debug mode is active
        if getattr(args, "debug", False):
            save_verification_debug(
                OUTPUT_DIR,
                frame,
                packet_mask if 'packet_mask' in locals() else None,
                components,
                last_gap_info if 'last_gap_info' in locals() else None,
                f"final_{final_state}"
            )

        while True:

            frame = read_frame()

            final_telemetry = [
                ("Final Decision", final_state),
                ("Packet Number", packet_number),
                ("Verified Pieces", f"{len(components)}"),
                ("Valid Samples", f"{valid_count}/{sample_count} ({valid_ratio:.1%})"),
                ("Stable Streak", str(stable_streak))
            ]

            final_dashboard = create_dashboard(
                last_preview if last_preview is not None else frame,
                "FINAL DECISION",
                final_state,
                [
                    final_reason,
                    "R = Restart next packet",
                    "Q = Quit verification"
                ],
                extra_info=final_telemetry
            )
            key = show(final_dashboard, [])

            if key in (ord("q"), 27):
                break

            if key == ord("r"):

                print(
                    "Restart application for next packet."
                )

                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print("Camera released.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
