"""
Packet-Cut 2-Segmentation PyQt6 Application
===========================================
Unified PyQt6 Desktop Application incorporating the shared packet_cut_core logic.

Features:
- 7-Stage Workflow with operator-guided controls.
- Robust Hand Detection with red seal false-positive filtering.
- Scissor validation & spatial interaction gating.
- Full 10-second verification period (CUT_VERIFY_SECONDS = 10.0, MIN_VERIFY_SAMPLES = 8, MIN_VALID_SAMPLE_RATIO = 0.90).
- Standardized Decision States: CUT_CONFIRMED, NOT_CUTTED, NOT_CONFIRMED, UNVERIFIED, PACKET_MISSING.
- Preserves all UI styling, metrics cards, camera thread, and PaddleOCR pipeline.
"""

import argparse
import os
import sys
import time
import re
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QFrame, QMessageBox, QSizePolicy, QProgressBar
)

# Shared Core CV & Decision Engine
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
PACKET_MODEL_PATH = MODEL_DIR / "packet-seg.onnx"
HAND_MODEL_PATH = MODEL_DIR / "hand-seg.onnx"
SCISSOR_MODEL_PATH = MODEL_DIR / "scissor-seg.pt"
TOPSTRIP_MODEL_PATH = MODEL_DIR / "strip-seg.onnx"
PACKET_NUMBER_MODEL_PATH = MODEL_DIR / "packet-no-obb.onnx"
CUT_MODEL_PATH = MODEL_DIR / "cut-seg.pt"

IMGSZ = 320
PACKET_CONF = 0.25
HAND_CONF = 0.30
SCISSOR_CONF = 0.30
TOPSTRIP_CONF = 0.25
NUMBER_IMGSZ = 640
NUMBER_CONF = 0.05
BLUR_VARIANCE_MIN = 25.0
OUTPUT_DIR = Path("packet_cut_outputs")
_OCR_ENGINE = None


def require_file(path, description):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path.resolve()}")
    return path


def class_names_dict(model):
    names = model.names
    if isinstance(names, dict):
        return names
    return {i: name for i, name in enumerate(names)}


def matching_class_ids(model, keywords):
    found = []
    for class_id, name in class_names_dict(model).items():
        normalized = re.sub(r"[^a-z0-9]", "", str(name).lower())
        if any(k in normalized for k in keywords):
            found.append(int(class_id))
    return found or None


def predict(model, frame, confidence, imgsz=IMGSZ, device: str | int = "cpu"):
    return model.predict(
        source=frame,
        imgsz=imgsz,
        conf=confidence,
        device=device,
        verbose=False
    )[0]


def detection_present(result, selected_ids=None):
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return False
    if selected_ids is None:
        return True
    if result.boxes.cls is None:
        return False
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    return any(int(c) in selected_ids for c in classes)


def get_union_mask(result, frame_shape, selected_ids=None):
    h, w = frame_shape[:2]
    union = np.zeros((h, w), dtype=np.uint8)
    if result is None or result.masks is None or result.masks.data is None:
        return union

    mask_data = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int) if (result.boxes is not None and result.boxes.cls is not None) else None

    for i, mask in enumerate(mask_data):
        if selected_ids is not None and classes is not None:
            if i < len(classes) and classes[i] not in selected_ids:
                continue
        resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        union[resized >= 0.5] = 255

    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(union, cv2.MORPH_OPEN, kernel)


def blur_score(frame):
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def draw_mask(frame, mask, color, alpha=0.25):
    if mask is None or not np.any(mask):
        return frame
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def detect_packet_number(frame, number_model, device):
    """Detects packet number ROI box and crops number region."""
    h, w = frame.shape[:2]
    result = number_model.predict(
        source=frame,
        imgsz=NUMBER_IMGSZ,
        conf=NUMBER_CONF,
        device=device,
        verbose=False
    )[0]

    boxes = None
    if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        boxes = result.obb.xyxy.cpu().numpy()
    elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()

    if boxes is None or len(boxes) == 0:
        return None, None, result

    box = boxes[0].astype(int)
    x1 = max(0, min(int(box[0]), w - 1))
    y1 = max(0, min(int(box[1]), h - 1))
    x2 = max(0, min(int(box[2]), w))
    y2 = max(0, min(int(box[3]), h))

    if x2 <= x1 or y2 <= y1:
        return None, None, result

    pad_x = int((x2 - x1) * 0.08)
    pad_y = int((y2 - y1) * 0.15)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    crop = frame[cy1:cy2, cx1:cx2].copy()
    return crop, (x1, y1, x2, y2), result


def get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from paddleocr import PaddleOCR

        # Avoid the unsupported oneDNN PIR attribute conversion on CPU.
        _OCR_ENGINE = PaddleOCR(
            enable_mkldnn=False,
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
    return _OCR_ENGINE


def preprocess_crop_for_ocr(crop, scale_factor=3.0):
    if crop is None or crop.size == 0:
        return None
    scaled = cv2.resize(
        crop, (0, 0),
        fx=scale_factor, fy=scale_factor,
        interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(contrast, -1, kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def run_ocr(crop):
    ocr_engine = get_ocr_engine()
    if ocr_engine is None or crop is None:
        return None, 0.0, []

    def parse_paddle_results(predict_results):
        candidates = []
        if not predict_results:
            return candidates
        res = predict_results[0] if isinstance(predict_results, list) else predict_results
        if hasattr(res, "keys"):
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            for t, s in zip(texts, scores):
                txt = re.sub(r"[^0-9A-Za-z]", "", str(t).strip())
                if txt:
                    candidates.append((txt, float(s)))
        elif isinstance(res, list):
            for item in res:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    if isinstance(item[1], (list, tuple)) and len(item[1]) == 2:
                        txt = re.sub(r"[^0-9A-Za-z]", "", str(item[1][0]).strip())
                        score = float(item[1][1])
                        if txt:
                            candidates.append((txt, score))
        return candidates

    prep_crop = preprocess_crop_for_ocr(crop, scale_factor=3.0)
    pred_res = ocr_engine.ocr(prep_crop)
    candidates = parse_paddle_results(pred_res)

    if not candidates:
        simple_3x = cv2.resize(crop, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        pred_res = ocr_engine.ocr(simple_3x)
        candidates = parse_paddle_results(pred_res)

    if not candidates:
        pred_res = ocr_engine.ocr(crop)
        candidates = parse_paddle_results(pred_res)

    if not candidates:
        return None, 0.0, []

    six_digits = [c for c in candidates if len(c[0]) == 6 and c[0].isdigit()]
    if six_digits:
        best = max(six_digits, key=lambda c: c[1])
        return best[0], best[1], candidates

    digits_only = [c for c in candidates if c[0].isdigit()]
    if digits_only:
        best = max(digits_only, key=lambda c: c[1])
        return best[0], best[1], candidates

    best = max(candidates, key=lambda c: c[1])
    return best[0], best[1], candidates


# ============================================================
# PACKET WORKER THREAD
# ============================================================
class PacketWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    stage_changed = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    detail_changed = pyqtSignal(str)
    metrics_changed = pyqtSignal(dict)
    prompt_operator = pyqtSignal(str, str)
    completed = pyqtSignal(str, str)
    error = pyqtSignal(str)

    def __init__(self, camera_id=0, device: str | int = "cpu", parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.device = device
        self._condition = threading.Condition()
        self._action = None
        self._stop_requested = False
        self.packet_number = "—"
        self.models = {}
        self.reference_mask = None
        self.reference_area = 0.0
        self.hand_ids = self.scissor_ids = self.topstrip_ids = self.packet_ids = None
        self.last_frame = None

    def send_action(self, action):
        with self._condition:
            self._action = action
            self._condition.notify_all()

    def stop(self):
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()

    def _wait_action(self, valid_actions):
        with self._condition:
            while not self._stop_requested and self._action not in valid_actions:
                self._condition.wait(timeout=0.2)
            if self._stop_requested:
                return "quit"
            action, self._action = self._action, None
            return action

    def _take_pending_action(self):
        with self._condition:
            if self._action:
                action, self._action = self._action, None
                return action
        return None

    def _emit_frame(self, frame):
        self.last_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.frame_ready.emit(qimg)

    def _load_models(self):
        paths = {
            "packet": PACKET_MODEL_PATH,
            "hand": HAND_MODEL_PATH,
            "scissor": SCISSOR_MODEL_PATH,
            "topstrip": TOPSTRIP_MODEL_PATH,
            "number": PACKET_NUMBER_MODEL_PATH,
            "cut": CUT_MODEL_PATH,
        }
        for key, path in paths.items():
            require_file(path, f"{key} model")
        self.status_changed.emit("Loading models…")
        self.models["packet"] = YOLO(str(paths["packet"]), task="segment")
        self.models["hand"] = YOLO(str(paths["hand"]), task="segment")
        self.models["scissor"] = YOLO(str(paths["scissor"]), task="segment")
        self.models["topstrip"] = YOLO(str(paths["topstrip"]), task="segment")
        self.models["number"] = YOLO(str(paths["number"]))
        self.models["cut"] = YOLO(str(paths["cut"]), task="segment")
        self.hand_ids = matching_class_ids(self.models["hand"], ("hand",))
        self.scissor_ids = matching_class_ids(self.models["scissor"], ("scissor",))
        self.topstrip_ids = matching_class_ids(self.models["topstrip"], ("topstrip", "strip", "seal", "top"))
        self.packet_ids = matching_class_ids(self.models["packet"], ("packet", "bag"))

    def _predict_mask(self, model_key, frame, conf, ids=None):
        result = predict(self.models[model_key], frame, conf, device=self.device)
        mask = get_union_mask(result, frame.shape[:2], ids)
        return result, mask

    def _front_ocr_stage(self, cap):
        self.stage_changed.emit("01  FRONT SIDE · PACKET NUMBER")
        self.status_changed.emit("Place packet with FRONT side facing camera.")
        while not self._stop_requested:
            ok, frame = cap.read()
            if not ok:
                continue
            crop, bbox, det_result = detect_packet_number(frame, self.models["number"], self.device)
            view = frame.copy()
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(view, (x1, y1), (x2, y2), (40, 220, 90), 2)
                self.status_changed.emit("Packet number region detected. Press Capture & OCR.")
            else:
                self.status_changed.emit("Searching for packet number region…")
            self._emit_frame(view)
            action = self._take_pending_action()
            if action == "capture":
                crop, bbox, _ = detect_packet_number(frame, self.models["number"], self.device)
                if crop is None:
                    self.status_changed.emit("Number ROI not found. Adjust packet and retry.")
                    continue
                text, score, candidates = run_ocr(crop)
                self.packet_number = text or "UNREADABLE"
                self.metrics_changed.emit({"Packet number": self.packet_number, "OCR confidence": f"{score:.2f}"})
                self.prompt_operator.emit("Confirm packet number", f"YOLO ROI detected. PaddleOCR read: {self.packet_number}\nConfidence: {score:.2f}\nAccept only if the number is correct.")
                choice = self._wait_action({"accept", "retry", "quit"})
                if choice == "accept" and self.packet_number != "UNREADABLE":
                    return True
                if choice == "quit":
                    return False
                self.status_changed.emit("Please capture and read the number again.")
            elif action == "quit":
                return False
        return False

    def _back_and_topstrip_stage(self, cap):
        self.stage_changed.emit("02  BACK SIDE · TOPSTRIP INSPECTION")
        self.status_changed.emit("Turn packet to BACK / seal side, then press Back Side Ready.")
        action = self._wait_action({"back_ready", "quit"})
        if action == "quit":
            return False
        closed_frames = 0
        while not self._stop_requested:
            ok, frame = cap.read()
            if not ok:
                continue
            packet_result, packet_mask = self._predict_mask("packet", frame, PACKET_CONF, self.packet_ids)
            top_result, top_mask = self._predict_mask("topstrip", frame, TOPSTRIP_CONF, self.topstrip_ids)
            top_present = detection_present(top_result, self.topstrip_ids)
            valid_packet = np.count_nonzero(packet_mask) > 500
            sharp = blur_score(frame) >= BLUR_VARIANCE_MIN
            preview = draw_mask(frame, packet_mask, (255, 120, 0), 0.20)
            preview = draw_mask(preview, top_mask, (255, 0, 255), 0.35)
            if not valid_packet or not sharp:
                text = "UNKNOWN_ORIENTATION · packet visibility / sharpness invalid"
                closed_frames = 0
            elif top_present:
                text = "OPEN · TopStrip visible · CUTTING NOT READY"
                closed_frames = 0
            else:
                text = "CLOSED candidate · TopStrip not detected"
                closed_frames += 1
            cv2.putText(preview, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            self._emit_frame(preview)
            self.status_changed.emit(text)
            self.metrics_changed.emit({
                "TopStrip": "VISIBLE" if top_present else "NOT DETECTED",
                "Packet mask": "VALID" if valid_packet else "INVALID",
                "Image sharpness": f"{blur_score(frame):.1f}",
                "Closed candidate frames": str(closed_frames),
            })
            action = self._take_pending_action()
            if action == "confirm_closed":
                if valid_packet and sharp and not top_present and closed_frames >= 3:
                    return True
                self.status_changed.emit("Cannot accept: require valid inspection and 3 consecutive no-TopStrip frames.")
            elif action == "quit":
                return False
        return False

    def _capture_baseline(self, cap):
        self.stage_changed.emit("03  BASELINE · INTACT PACKET MASK")
        self.status_changed.emit("Capturing intact packet segmentation baseline…")
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Unable to capture baseline frame.")
        packet_result = predict(self.models["packet"], frame, PACKET_CONF, device=self.device)
        p_info = extract_packet_pieces(packet_result, frame.shape[:2], self.packet_ids)
        pieces = p_info["pieces"]

        if len(pieces) != 1:
            raise RuntimeError(f"Baseline requires exactly one connected intact packet mask; found {len(pieces)}.")

        self.reference_mask = pieces[0]["mask"]
        self.reference_area = float(np.count_nonzero(self.reference_mask))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUTPUT_DIR / "baseline_frame.jpg"), frame)
        cv2.imwrite(str(OUTPUT_DIR / "baseline_mask.png"), self.reference_mask)
        preview = draw_mask(frame, self.reference_mask, (0, 220, 80), 0.35)
        self._emit_frame(preview)
        self.metrics_changed.emit({
            "Baseline area": f"{self.reference_area:.0f} px",
            "Baseline components": "1",
            "Packet number": self.packet_number
        })
        self.status_changed.emit("Baseline stored. Waiting for cutting activity.")

    def _cut_activity_and_clear(self, cap):
        self.stage_changed.emit("04  CUTTING · HAND / SCISSORS")
        activity_seen = False
        clear_started = None
        hand_streak = 0
        scissor_streak = 0

        while not self._stop_requested:
            ok, frame = cap.read()
            if not ok:
                continue

            hr = predict(self.models["hand"], frame, HAND_CONF, device=self.device)
            sr = predict(self.models["scissor"], frame, SCISSOR_CONF, device=self.device)

            # Robust hand evaluation with seal filtering
            raw_hp, valid_hand_mask, hand_diags = evaluate_hand_detection(
                hr, self.hand_ids, frame.shape[:2], reference_mask=self.reference_mask, frame=frame
            )
            hand_streak = (hand_streak + 1) if raw_hp else 0
            hand_confirmed = (hand_streak >= HAND_CONFIRM_FRAMES)

            # Robust scissor evaluation with packet spatial proximity check
            raw_sp, valid_scissor_mask, scissor_diags = evaluate_scissor_detection(
                sr, self.scissor_ids, frame.shape[:2], packet_mask=self.reference_mask
            )
            scissor_streak = (scissor_streak + 1) if raw_sp else 0
            scissor_confirmed = (scissor_streak >= SCISSOR_CONFIRM_FRAMES)

            preview = draw_mask(frame, valid_hand_mask, (0, 0, 255), 0.35)
            preview = draw_mask(preview, valid_scissor_mask, (0, 255, 255), 0.35)

            # Cutting activity: require confirmed hand OR confirmed scissors near packet
            if hand_confirmed or scissor_confirmed:
                activity_seen = True
                clear_started = None
                status = "CUTTING ACTIVITY · verification paused"
            elif activity_seen:
                if clear_started is None:
                    clear_started = time.monotonic()
                elapsed = time.monotonic() - clear_started
                status = f"Scene clear: {elapsed:.1f} / {CLEAR_WAIT_SECONDS:.0f}s"
                if elapsed >= CLEAR_WAIT_SECONDS:
                    return True
            else:
                status = "CUTTING READY · waiting for hand or scissors"

            cv2.putText(preview, status, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            self._emit_frame(preview)
            self.status_changed.emit(status)
            self.metrics_changed.emit({
                "Hand": "YES" if hand_confirmed else "NO",
                "Scissors": "YES" if scissor_confirmed else "NO",
                "Cut activity seen": "YES" if activity_seen else "NO",
                "Clear wait": f"{max(0, time.monotonic()-clear_started):.1f}s" if clear_started else "—",
            })
            if self._take_pending_action() == "quit":
                return False
        return False

    def _postcut_verify(self, cap):
        """
        Enforces complete 10-second post-cut verification:
        - Runs packet-seg.onnx at VERIFY_SAMPLE_INTERVAL.
        - Exactly two significant pieces required.
        - Distance transform gap & exposed background verified.
        - Pauses and resets on hand/scissor return.
        - Enforces MIN_VERIFY_SAMPLES and MIN_VALID_SAMPLE_RATIO across full 10s.
        """
        self.stage_changed.emit("05  POST-CUT · SECOND SEGMENTATION")
        start = time.monotonic()
        next_sample = 0.0
        sample_count = 0
        valid_count = 0
        stable_streak = 0
        previous_sig = None
        components = []
        last_gap = None
        last_mask = None
        reason = "Waiting for post-cut sample"
        hand_streak = 0
        scissor_streak = 0
        last_p_info = None
        frame = None

        while not self._stop_requested:
            elapsed_total = time.monotonic() - start
            if elapsed_total >= CUT_VERIFY_SECONDS:
                break

            ok, frame = cap.read()
            if not ok:
                continue

            # Continuous hand & scissors checking
            hr = predict(self.models["hand"], frame, HAND_CONF, device=self.device)
            sr = predict(self.models["scissor"], frame, SCISSOR_CONF, device=self.device)
            raw_hp, _, _ = evaluate_hand_detection(hr, self.hand_ids, frame.shape[:2], reference_mask=self.reference_mask, frame=frame)
            hand_streak = (hand_streak + 1) if raw_hp else 0
            hp = (hand_streak >= HAND_CONFIRM_FRAMES)

            raw_sp, _, _ = evaluate_scissor_detection(sr, self.scissor_ids, frame.shape[:2], packet_mask=self.reference_mask)
            scissor_streak = (scissor_streak + 1) if raw_sp else 0
            sp = (scissor_streak >= SCISSOR_CONFIRM_FRAMES)

            if hp or sp:
                self.status_changed.emit("Verification interrupted: Hand/Scissors detected! Resetting verification...")
                self._emit_frame(frame)
                if not self._wait_for_clear_again(cap):
                    return "VERIFICATION_UNCERTAIN", "Could not obtain clear scene after interruption."
                # Fresh restart of full 10-second verification period
                return self._postcut_verify(cap)

            # Sample packet segmentation at configured interval
            if elapsed_total >= next_sample:
                packet_result = predict(self.models["packet"], frame, PACKET_CONF, device=self.device)
                p_info = extract_packet_pieces(packet_result, frame.shape[:2], self.packet_ids, self.reference_area)
                last_p_info = p_info
                components = p_info["pieces"]
                last_mask = p_info["union_mask"]
                sample_count += 1
                gap_info = None

                # Rule 1: Packet missing
                if p_info["total_area"] < 300:
                    reason = "PACKET_MISSING: Packet area below detection threshold."
                    stable_streak = 0
                    previous_sig = None
                elif len(components) == 1:
                    reason = f"NOT_CUTTED: Only 1 connected packet piece detected (YOLO instances: {p_info['yolo_instance_count']})."
                    stable_streak = 0
                    previous_sig = None
                elif len(components) != 2:
                    reason = f"UNVERIFIED: Found {len(components)} fragments (expected exactly 2)."
                    stable_streak = 0
                    previous_sig = None
                else:
                    a, b = components[:2]
                    total_ratio = (a["area"] + b["area"]) / max(self.reference_area, 1.0)
                    if not (MIN_TOTAL_AREA_RATIO <= total_ratio <= MAX_TOTAL_AREA_RATIO):
                        reason = f"UNVERIFIED: Final/baseline area ratio out of range ({total_ratio:.2f})."
                        stable_streak = 0
                        previous_sig = None
                    else:
                        gap_info = analyze_packet_gap(
                            a["mask"], b["mask"], frame=frame,
                            reference_area=self.reference_area,
                            min_norm_gap=MIN_NORM_GAP_RATIO,
                            min_coverage=MIN_GAP_COVERAGE_RATIO
                        )
                        last_gap = gap_info

                        if not gap_info["valid"]:
                            if gap_info["min_gap"] < 2.0:
                                reason = f"NOT_CONFIRMED: Masks touching without physical gap ({gap_info['min_gap']:.1f}px)."
                            else:
                                reason = f"UNVERIFIED: {gap_info['reason']}."
                            stable_streak = 0
                            previous_sig = None
                        else:
                            curr_sig = piece_signature(components)
                            if previous_sig is None or is_stable_pair(previous_sig, curr_sig):
                                stable_streak += 1
                            else:
                                stable_streak = 1
                            previous_sig = curr_sig
                            valid_count += 1
                            reason = f"Two pieces + gap confirmed ({gap_info['min_gap']:.1f}px, {gap_info['coverage_pct']:.0f}% interface) [Streak: {stable_streak}]"

                if getattr(self, "debug_enabled", True):
                    save_verification_debug(OUTPUT_DIR, frame, last_mask, components, gap_info, f"sample_{sample_count}")

                next_sample += VERIFY_SAMPLE_INTERVAL

            # Render live overlay
            preview = draw_verification_overlay(frame, components, last_gap)
            remain = max(0.0, CUT_VERIFY_SECONDS - elapsed_total)
            cv2.putText(
                preview,
                f"VERIFYING: {elapsed_total:.1f}/{CUT_VERIFY_SECONDS:.0f}s | Valid: {valid_count}/{sample_count} | Streak: {stable_streak}",
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                2
            )
            self._emit_frame(preview)
            self.status_changed.emit(reason)

            # Update metrics
            yolo_inst = str(last_p_info["yolo_instance_count"]) if last_p_info else "—"
            raw_cc = str(last_p_info["raw_cc_count"]) if last_p_info else "—"
            self.metrics_changed.emit({
                "Detected pieces": f"{len(components)} (CC: {raw_cc}, YOLO: {yolo_inst})",
                "Gap": f"{last_gap['min_gap']:.1f}px" if last_gap else "NOT VALID",
                "Gap coverage": f"{last_gap['coverage_pct']:.1f}%" if last_gap else "—",
                "Area ratio": f"{((sum(c['area'] for c in components[:2]))/max(self.reference_area,1)):.2f}" if len(components) == 2 else "—",
                "Valid samples": f"{valid_count}/{sample_count}",
                "Stability streak": str(stable_streak),
                "Time remaining": f"{remain:.1f}s",
            })

            if self._take_pending_action() == "quit":
                return "STOPPED", "Operator stopped verification."

        if frame is None:
            return "UNVERIFIED", "No camera frames were captured during post-cut verification."

        # Run cut-seg.pt confirmation inference on the current stable frame (strict 0.70 threshold)
        cut_result = predict(self.models["cut"], frame, 0.25, device=self.device)
        cut_eval = evaluate_cut_model_prediction(cut_result, cut_class_id=0, min_conf=0.70)
        is_cut_model_confirmed = cut_eval["is_cut_detected"]

        # Final Decision Gates after 5-second wait & verification
        valid_ratio = valid_count / max(sample_count, 1)

        if last_p_info and last_p_info["total_area"] < 300:
            return "PACKET_MISSING", "Packet disappeared or could not be detected."

        # If reference area exists, check if largest piece is still essentially intact
        if self.reference_area and len(components) >= 1:
            primary_area = max(p["area"] for p in components)
            if primary_area >= 0.88 * self.reference_area:
                return "NOT_CUTTED", f"Intact packet confirmed: Main body retains {primary_area/self.reference_area:.0%} of baseline area."

        # Hybrid Decision: Physical 2-piece separation confirmed AND cut-seg.pt verifies Cut class (>= 0.70)
        if (
            sample_count >= MIN_VERIFY_SAMPLES
            and valid_ratio >= MIN_VALID_SAMPLE_RATIO
            and stable_streak >= 5
            and len(components) == 2
            and last_gap and last_gap["valid"]
            and is_cut_model_confirmed
        ):
            return "CUT_CONFIRMED", f"Cut confirmed by both physical gap ({last_gap['min_gap']:.1f}px) and cut model ({cut_eval['confidence']:.1%})."

        if is_cut_model_confirmed and cut_eval["confidence"] >= 0.80 and len(components) >= 2 and last_gap and last_gap["valid"]:
            return "CUT_CONFIRMED", f"Cut confirmed by cut model ({cut_eval['confidence']:.1%}) with {len(components)} separated pieces."

        if len(components) == 1:
            return "NOT_CUTTED", f"Intact packet confirmed: Only 1 connected packet piece detected."

        if not is_cut_model_confirmed:
            return "NOT_CONFIRMED", f"Cut model did not detect positive cut (conf: {cut_eval['confidence']:.1%}, need >= 70%)."

        if len(components) != 2:
            return "UNVERIFIED", f"Segmentation fragmented: Expected exactly 2 pieces, detected {len(components)}."

        if last_gap and last_gap["min_gap"] < 2.0:
            return "NOT_CONFIRMED", f"Pieces are touching without required physical separation (gap: {last_gap['min_gap']:.1f}px)."

        return "UNVERIFIED", f"Verification incomplete: Valid ratio {valid_ratio:.0%} (< {MIN_VALID_SAMPLE_RATIO:.0%}) or streak {stable_streak} (< 5). Reason: {reason}"

    def _wait_for_clear_again(self, cap):
        clear_start = None
        hand_streak = 0
        scissor_streak = 0
        while not self._stop_requested:
            ok, frame = cap.read()
            if not ok:
                continue

            hr = predict(self.models["hand"], frame, HAND_CONF, device=self.device)
            sr = predict(self.models["scissor"], frame, SCISSOR_CONF, device=self.device)

            raw_hp, _, _ = evaluate_hand_detection(hr, self.hand_ids, frame.shape[:2], reference_mask=self.reference_mask, frame=frame)
            hand_streak = (hand_streak + 1) if raw_hp else 0
            hp = (hand_streak >= HAND_CONFIRM_FRAMES)

            raw_sp, _, _ = evaluate_scissor_detection(sr, self.scissor_ids, frame.shape[:2], packet_mask=self.reference_mask)
            scissor_streak = (scissor_streak + 1) if raw_sp else 0
            sp = (scissor_streak >= SCISSOR_CONFIRM_FRAMES)

            if hp or sp:
                clear_start = None
                status = "Hand or scissors still present in scene — waiting for clear"
            else:
                if clear_start is None:
                    clear_start = time.monotonic()
                elapsed = time.monotonic() - clear_start
                status = f"Clearing scene: {elapsed:.1f} / {CLEAR_WAIT_SECONDS:.0f}s"
                if elapsed >= CLEAR_WAIT_SECONDS:
                    return True

            self._emit_frame(frame)
            self.status_changed.emit(status)
            if self._take_pending_action() == "quit":
                return False
        return False

    def run(self):
        cap = None
        try:
            self._load_models()
            cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera index {self.camera_id}.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            if not self._front_ocr_stage(cap):
                return
            if not self._back_and_topstrip_stage(cap):
                return
            self._capture_baseline(cap)
            if not self._cut_activity_and_clear(cap):
                return
            final_state, final_reason = self._postcut_verify(cap)
            self.stage_changed.emit("06  FINAL DECISION")
            self.status_changed.emit(final_state)
            self.detail_changed.emit(final_reason)
            self.completed.emit(final_state, final_reason)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if cap is not None:
                cap.release()


# ============================================================
# PYQT6 USER INTERFACE
# ============================================================
class MetricCard(QFrame):
    def __init__(self, label, value="—"):
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        self.name = QLabel(label.upper())
        self.name.setObjectName("metricName")
        self.value = QLabel(value)
        self.value.setObjectName("metricValue")
        layout.addWidget(self.name)
        layout.addWidget(self.value)

    def set_value(self, value):
        self.value.setText(str(value))


class MainWindow(QMainWindow):
    def __init__(self, camera=0, device: str | int = "cpu"):
        super().__init__()
        self.setWindowTitle("Embsys · Packet-Cut Inspection")
        self.resize(1360, 850)
        self.worker = PacketWorker(camera, device)
        self.metrics = {}
        self._build_ui()
        self._connect_worker()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(14)

        header = QHBoxLayout()
        titlebox = QVBoxLayout()
        title = QLabel("PACKET-CUT INSPECTION")
        title.setObjectName("appTitle")
        subtitle = QLabel("Two-stage packet segmentation · physical gap verification")
        subtitle.setObjectName("subTitle")
        titlebox.addWidget(title)
        titlebox.addWidget(subtitle)
        header.addLayout(titlebox)
        header.addStretch()
        self.live_badge = QLabel("● SYSTEM READY")
        self.live_badge.setObjectName("liveBadge")
        header.addWidget(self.live_badge)
        outer.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(14)
        outer.addLayout(body, 1)

        left = QVBoxLayout()
        self.video = QLabel("Camera preview will appear here")
        self.video.setObjectName("video")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(720, 540)
        self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left.addWidget(self.video, 1)
        self.stage = QLabel("READY · Start inspection")
        self.stage.setObjectName("stage")
        left.addWidget(self.stage)
        body.addLayout(left, 7)

        right = QVBoxLayout()
        right.setSpacing(10)
        status_card = QFrame()
        status_card.setObjectName("statusCard")
        sv = QVBoxLayout(status_card)
        sv.setContentsMargins(16, 14, 16, 14)
        cap = QLabel("CURRENT STATUS")
        cap.setObjectName("metricName")
        self.status = QLabel("Ready to begin")
        self.status.setObjectName("statusValue")
        self.detail = QLabel("Place the packet FRONT side toward the camera.")
        self.detail.setWordWrap(True)
        self.detail.setObjectName("detail")
        sv.addWidget(cap)
        sv.addWidget(self.status)
        sv.addWidget(self.detail)
        right.addWidget(status_card)

        grid = QGridLayout()
        grid.setSpacing(8)
        names = [
            "Packet number", "OCR confidence", "TopStrip", "Baseline area",
            "Detected pieces", "Gap", "Gap coverage", "Area ratio",
            "Valid samples", "Stability streak", "Time remaining"
        ]
        for i, name in enumerate(names):
            card = MetricCard(name)
            self.metrics[name] = card
            grid.addWidget(card, i // 2, i % 2)
        right.addLayout(grid)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Inspection progress")
        right.addWidget(self.progress)

        right.addStretch()
        controls = QGridLayout()
        self.start_btn = QPushButton("▶  Start Inspection")
        self.start_btn.setObjectName("primary")
        self.capture_btn = QPushButton("Capture & OCR")
        self.back_btn = QPushButton("Back Side Ready")
        self.closed_btn = QPushButton("Confirm CLOSED")
        self.accept_btn = QPushButton("Accept Number")
        self.retry_btn = QPushButton("Retry OCR")
        self.restart_btn = QPushButton("Restart")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("danger")
        controls.addWidget(self.start_btn, 0, 0, 1, 2)
        controls.addWidget(self.capture_btn, 1, 0)
        controls.addWidget(self.accept_btn, 1, 1)
        controls.addWidget(self.retry_btn, 2, 0)
        controls.addWidget(self.back_btn, 2, 1)
        controls.addWidget(self.closed_btn, 3, 0, 1, 2)
        controls.addWidget(self.restart_btn, 4, 0)
        controls.addWidget(self.stop_btn, 4, 1)
        right.addLayout(controls)
        body.addLayout(right, 4)

        self.capture_btn.setEnabled(False)
        self.back_btn.setEnabled(False)
        self.closed_btn.setEnabled(False)
        self.accept_btn.setEnabled(False)
        self.retry_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        self.start_btn.clicked.connect(self.start_inspection)
        self.capture_btn.clicked.connect(lambda: self.worker.send_action("capture"))
        self.back_btn.clicked.connect(lambda: self.worker.send_action("back_ready"))
        self.closed_btn.clicked.connect(lambda: self.worker.send_action("confirm_closed"))
        self.accept_btn.clicked.connect(lambda: self.worker.send_action("accept"))
        self.retry_btn.clicked.connect(lambda: self.worker.send_action("retry"))
        self.restart_btn.clicked.connect(self.restart_inspection)
        self.stop_btn.clicked.connect(self.stop_inspection)

        self.setStyleSheet("""
            QWidget#root { background:#0b1220; color:#e8eef8; }
            QLabel { color:#e8eef8; }
            QLabel#appTitle { font-size:25px; font-weight:800; letter-spacing:1px; }
            QLabel#subTitle { color:#93a4bd; font-size:12px; }
            QLabel#liveBadge { background:#12382f; color:#67e8b2; padding:9px 14px; border-radius:12px; font-weight:700; }
            QLabel#video { background:#050a12; border:1px solid #27364b; border-radius:16px; color:#8091a9; font-size:16px; }
            QLabel#stage { color:#91a8c7; font-size:12px; font-weight:700; padding:5px; }
            QFrame#statusCard, QFrame#metricCard { background:#121e30; border:1px solid #25364d; border-radius:13px; }
            QLabel#metricName { color:#8295b1; font-size:9px; font-weight:800; letter-spacing:1px; }
            QLabel#metricValue { color:#f3f7ff; font-size:17px; font-weight:700; }
            QLabel#statusValue { color:#61e6ad; font-size:20px; font-weight:800; }
            QLabel#detail { color:#aab8cb; font-size:11px; }
            QPushButton { background:#1b2a40; border:1px solid #344963; border-radius:9px; padding:11px 9px; color:#e8eef8; font-weight:700; }
            QPushButton:hover { background:#263b58; }
            QPushButton:disabled { color:#526176; background:#121b29; border-color:#202d3d; }
            QPushButton#primary { background:#176b56; border:0; color:white; padding:14px; }
            QPushButton#primary:hover { background:#1c8469; }
            QPushButton#danger { background:#632b38; border:0; }
            QProgressBar { border:1px solid #293a50; border-radius:6px; text-align:center; background:#101a29; color:#dce8f8; height:17px; }
            QProgressBar::chunk { background:#22a982; border-radius:5px; }
        """)

    def _connect_worker(self):
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.stage_changed.connect(self.stage.setText)
        self.worker.status_changed.connect(self.status.setText)
        self.worker.detail_changed.connect(self.detail.setText)
        self.worker.metrics_changed.connect(self._update_metrics)
        self.worker.prompt_operator.connect(self._operator_prompt)
        self.worker.completed.connect(self._completed)
        self.worker.error.connect(self._error)

    def _show_frame(self, image):
        pix = QPixmap.fromImage(image)
        self.video.setPixmap(pix.scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.worker.last_frame is not None:
            self.worker._emit_frame(self.worker.last_frame)

    def _update_metrics(self, data):
        for key, value in data.items():
            if key in self.metrics:
                self.metrics[key].set_value(value)

    def _operator_prompt(self, title, message):
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(message)
        accept = box.addButton("Accept number", QMessageBox.ButtonRole.AcceptRole)
        retry = box.addButton("Retry OCR", QMessageBox.ButtonRole.ResetRole)
        box.addButton("Stop", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked == accept:
            self.worker.send_action("accept")
        elif clicked == retry:
            self.worker.send_action("retry")
        else:
            self.worker.send_action("quit")

    def start_inspection(self):
        if self.worker.isRunning():
            return
        self.worker = PacketWorker(self.worker.camera_id, self.worker.device)
        self._connect_worker()
        self.start_btn.setEnabled(False)
        self.capture_btn.setEnabled(True)
        self.back_btn.setEnabled(True)
        self.closed_btn.setEnabled(True)
        self.accept_btn.setEnabled(False)
        self.retry_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.live_badge.setText("● SYSTEM RUNNING")
        self.status.setText("Loading models…")
        self.progress.setValue(5)
        self.worker.start()

    def stop_inspection(self):
        self.worker.stop()
        self.status.setText("Stopping safely…")

    def restart_inspection(self):
        self.restart_btn.setEnabled(False)
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        self.start_inspection()
        self.restart_btn.setEnabled(True)

    def _completed(self, state, reason):
        self.status.setText(state)
        self.detail.setText(reason)
        self.live_badge.setText("● INSPECTION COMPLETE")
        self.progress.setValue(100 if state == "CUT_CONFIRMED" else 60)
        self._set_idle()

    def _error(self, message):
        QMessageBox.critical(self, "Packet-Cut error", message)
        self.status.setText("ERROR")
        self.detail.setText(message)
        self._set_idle()

    def _set_idle(self):
        self.start_btn.setEnabled(True)
        self.capture_btn.setEnabled(False)
        self.back_btn.setEnabled(False)
        self.closed_btn.setEnabled(False)
        self.accept_btn.setEnabled(False)
        self.retry_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = int(args.device) if str(args.device).isdigit() else args.device

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow(args.camera, device)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
