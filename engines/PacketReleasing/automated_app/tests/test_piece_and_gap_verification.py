"""
Unit Test Suite for Packet-Cut Verification Pipeline
===================================================
Tests 7 synthetic edge cases and 1 real-world inference test:
1. Two clearly separated packet pieces (Gap > 4px) -> Valid separation.
2. Two touching packet pieces (Gap < 2px) -> Rejected: Touching.
3. Two overlapping masks -> Rejected: Overlapping.
4. One continuous packet -> Single piece: Correctly handled.
5. One very small false-positive component (< 150px) -> Filtered with documented reason.
6. Two valid pieces with narrow background gap (5px) -> Valid separation.
7. Letterbox coordinate unpadding & ROI transformation -> Accurate 1:1 mapping.
8. Real model inference on sample_1794575407282_samp9_invalid_orig.jpg -> Confirmed.
"""

import sys
import unittest
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))
if str(_APP_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT.parent))

import cv2
import numpy as np

import packet_cut_core
from packet_cut_core import (
    unpad_letterbox_mask,
    extract_packet_pieces,
    analyze_packet_gap,
    evaluate_cut_model_prediction,
    evaluate_final_cut_decision,
    piece_signature,
    is_stable_pair,
    draw_verification_overlay,
    MIN_BACKGROUND_GAP_PX,
    MIN_GAP_COVERAGE_RATIO,
    MIN_PIECE_AREA_PX,
    MIN_PIECE_AREA_RATIO
)


class MockYoloResult:
    """Helper mock object mimicking Ultralytics Results."""
    def __init__(self, boxes_xyxy, confidences, classes, polygons=None, mask_tensors=None):
        import torch

        class Boxes:
            def __init__(self, b, c, cl):
                self.xyxy = torch.tensor(b, dtype=torch.float32)
                self.conf = torch.tensor(c, dtype=torch.float32)
                self.cls = torch.tensor(cl, dtype=torch.float32)
            def __len__(self):
                return len(self.xyxy)

        class Masks:
            def __init__(self, polys, data_tensors):
                self.xy = [np.array(p, dtype=np.float32) for p in polys] if polys else None
                self.data = torch.tensor(np.array(data_tensors), dtype=torch.float32) if data_tensors else None

        self.boxes = Boxes(boxes_xyxy, confidences, classes) if boxes_xyxy is not None else None
        self.masks = Masks(polygons, mask_tensors) if (polygons is not None or mask_tensors is not None) else None


class TestPacketCutPipeline(unittest.TestCase):
    def setUp(self):
        self.frame_shape = (480, 640, 3)
        self.h, self.w = self.frame_shape[:2]

    def test_1_two_clearly_separated_pieces(self):
        """Case 1: Two clearly separated pieces (40px gap). Should pass physical verification."""
        mask_a = np.zeros((self.h, self.w), dtype=np.uint8)
        mask_b = np.zeros((self.h, self.w), dtype=np.uint8)

        # Piece A: Top strip y: 60..140, x: 150..350
        cv2.rectangle(mask_a, (150, 60), (350, 140), 255, -1)
        # Piece B: Main body y: 180..420, x: 150..350 (40px gap)
        cv2.rectangle(mask_b, (150, 180), (350, 420), 255, -1)

        gap_info = analyze_packet_gap(mask_a, mask_b, reference_area=60000)

        self.assertTrue(gap_info["valid"], f"Expected valid gap, got: {gap_info['reason']}")
        self.assertAlmostEqual(gap_info["min_gap"], 40.0, delta=1.5)
        self.assertGreaterEqual(gap_info["coverage_pct"], 40.0)
        self.assertEqual(gap_info["overlap_px"], 0)
        self.assertIsNotNone(gap_info["gap_line"])

        # Final decision test: Cut prediction + 2 valid pieces + gap = CUT_PHYSICALLY_CONFIRMED
        components = [
            {"mask": mask_b, "area": int(np.count_nonzero(mask_b)), "centroid": (250.0, 300.0), "bbox": (150, 180, 200, 240)},
            {"mask": mask_a, "area": int(np.count_nonzero(mask_a)), "centroid": (250.0, 100.0), "bbox": (150, 60, 200, 80)},
        ]
        cut_eval = {"is_cut_detected": True, "confidence": 0.89, "class_name": "Cut"}
        state, reason = evaluate_final_cut_decision(
            components, cut_eval, last_gap_info=gap_info,
            sample_count=10, valid_count=9, stable_streak=6,
            min_verify_samples=8, min_valid_sample_ratio=0.70, min_streak=5
        )
        self.assertEqual(state, "CUT_PHYSICALLY_CONFIRMED")
        print(f"[PASS] Test 1 (Two clearly separated pieces): min_gap={gap_info['min_gap']:.1f}px -> {state}")

    def test_2_two_touching_pieces(self):
        """Case 2: Two touching pieces (gap 0-1px). Must be rejected with reason indicating touching."""
        mask_a = np.zeros((self.h, self.w), dtype=np.uint8)
        mask_b = np.zeros((self.h, self.w), dtype=np.uint8)

        # Piece A y: 60..150, Piece B y: 151..400 (touching / 1px gap)
        cv2.rectangle(mask_a, (150, 60), (350, 150), 255, -1)
        cv2.rectangle(mask_b, (150, 151), (350, 400), 255, -1)

        gap_info = analyze_packet_gap(mask_a, mask_b, reference_area=60000)

        self.assertFalse(gap_info["valid"], "Touching pieces must not pass physical verification")
        self.assertLess(gap_info["min_gap"], 2.0)
        self.assertIn("touching", gap_info["reason"].lower())

        components = [
            {"mask": mask_b, "area": int(np.count_nonzero(mask_b)), "centroid": (250.0, 275.0), "bbox": (150, 151, 200, 249)},
            {"mask": mask_a, "area": int(np.count_nonzero(mask_a)), "centroid": (250.0, 105.0), "bbox": (150, 60, 200, 90)},
        ]
        cut_eval = {"is_cut_detected": True, "confidence": 0.85, "class_name": "Cut"}
        state, reason = evaluate_final_cut_decision(
            components, cut_eval, last_gap_info=gap_info,
            sample_count=10, valid_count=0, stable_streak=0
        )
        self.assertIn(state, ("CUT_VERIFICATION_PENDING", "MODEL_PREDICTED_CUT"))
        self.assertNotEqual(state, "CUT_PHYSICALLY_CONFIRMED")
        print(f"[PASS] Test 2 (Two touching pieces): min_gap={gap_info['min_gap']:.1f}px -> {gap_info['reason']}")

    def test_3_two_overlapping_masks(self):
        """Case 3: Two overlapping masks. Must be rejected as physically separated."""
        mask_a = np.zeros((self.h, self.w), dtype=np.uint8)
        mask_b = np.zeros((self.h, self.w), dtype=np.uint8)

        # 30px vertical overlap between y: 140 and 170
        cv2.rectangle(mask_a, (150, 60), (350, 170), 255, -1)
        cv2.rectangle(mask_b, (150, 140), (350, 400), 255, -1)

        gap_info = analyze_packet_gap(mask_a, mask_b, reference_area=60000)

        self.assertFalse(gap_info["valid"])
        self.assertGreater(gap_info["overlap_px"], 0)
        self.assertIn("overlapping", gap_info["reason"].lower())
        print(f"[PASS] Test 3 (Two overlapping masks): overlap={gap_info['overlap_px']}px -> {gap_info['reason']}")

    def test_4_one_continuous_packet(self):
        """Case 4: One continuous packet. Must yield 1 piece and NOT_CUTTED if intact predicted."""
        # Mock 1 YOLO detection of an intact packet
        poly = np.array([[150, 60], [350, 60], [350, 400], [150, 400]], dtype=np.float32)
        mock_res = MockYoloResult(
            boxes_xyxy=[[150, 60, 350, 400]],
            confidences=[0.95],
            classes=[0],
            polygons=[poly]
        )

        p_info = extract_packet_pieces(mock_res, self.frame_shape, reference_area=68000, verbose=False)
        self.assertEqual(p_info["accepted_piece_count"], 1)
        self.assertEqual(len(p_info["pieces"]), 1)

        # Intact model prediction
        cut_eval_intact = {"is_cut_detected": False, "confidence": 0.92, "class_name": "Packet"}
        state, reason = evaluate_final_cut_decision(p_info["pieces"], cut_eval_intact, last_gap_info=None)
        self.assertEqual(state, "NOT_CUTTED")

        # Cut model prediction on single piece -> must NOT confirm cut and must NOT say NOT_CUTTED!
        cut_eval_cut = {"is_cut_detected": True, "confidence": 0.89, "class_name": "Cut"}
        state2, reason2 = evaluate_final_cut_decision(p_info["pieces"], cut_eval_cut, last_gap_info=None)
        self.assertEqual(state2, "CUT_VERIFICATION_PENDING")
        print(f"[PASS] Test 4 (One continuous packet): intact->{state}, cut_pred->{state2}")

    def test_5_small_false_positive_component(self):
        """Case 5: One main piece + one tiny false-positive blob (< 150px). Blob must be filtered with documented reason."""
        poly_main = np.array([[150, 60], [350, 60], [350, 400], [150, 400]], dtype=np.float32)
        # Tiny 8x8 blob (area 64px)
        poly_noise = np.array([[10, 10], [18, 10], [18, 18], [10, 18]], dtype=np.float32)

        mock_res = MockYoloResult(
            boxes_xyxy=[[150, 60, 350, 400], [10, 10, 18, 18]],
            confidences=[0.95, 0.40],
            classes=[0, 0],
            polygons=[poly_main, poly_noise]
        )

        p_info = extract_packet_pieces(mock_res, self.frame_shape, reference_area=68000, verbose=False)

        self.assertEqual(p_info["yolo_instance_count"], 2)
        self.assertEqual(p_info["accepted_piece_count"], 1)
        self.assertEqual(p_info["rejected_piece_count"], 1)
        self.assertIn("minimum threshold", p_info["rejection_summary"][0].lower())
        print(f"[PASS] Test 5 (Small false-positive component): rejected reason='{p_info['rejection_summary'][0]}'")

    def test_6_valid_pieces_with_narrow_gap(self):
        """Case 6: Two valid pieces with narrow gap (5px gap >= 3px threshold). Should be preserved and verified."""
        mask_a = np.zeros((self.h, self.w), dtype=np.uint8)
        mask_b = np.zeros((self.h, self.w), dtype=np.uint8)

        # Top strip y: 80..145, Main piece y: 150..420 (5.0px gap)
        cv2.rectangle(mask_a, (150, 80), (350, 145), 255, -1)
        cv2.rectangle(mask_b, (150, 150), (350, 420), 255, -1)

        gap_info = analyze_packet_gap(mask_a, mask_b, reference_area=65000)

        self.assertTrue(gap_info["valid"], f"Expected narrow gap to be valid, reason: {gap_info['reason']}")
        self.assertAlmostEqual(gap_info["min_gap"], 5.0, delta=1.5)
        self.assertGreaterEqual(gap_info["coverage_pct"], 40.0)
        print(f"[PASS] Test 6 (Narrow 5px gap): min_gap={gap_info['min_gap']:.1f}px, coverage={gap_info['coverage_pct']:.1f}% -> VALID")

    def test_7_letterbox_coordinate_unpadding_and_roi(self):
        """Case 7: Letterbox coordinate unpadding and ROI transformation accuracy."""
        # Create a raw 320x320 tensor with letterbox padding (40px top/bottom padding for 640x480 frame)
        # In 640x480, scale=0.5 -> new_w=320, new_h=240, pad_y=40
        # Suppose target object in 640x480 is at y: 68..116, x: 160..360
        # In 320x320 letterbox: y is 40 + [34..58] = 74..98, x is [80..180]
        raw_mask = np.zeros((320, 320), dtype=np.float32)
        raw_mask[74:98, 80:180] = 1.0

        # Direct resize without unpadding (THE BUG)
        direct_resized = cv2.resize(raw_mask, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        y_direct_min, y_direct_max = np.where(direct_resized >= 0.5)[0].min(), np.where(direct_resized >= 0.5)[0].max()

        # Unpadded resize (OUR FIX)
        unpadded = unpad_letterbox_mask(raw_mask, (self.h, self.w), (320, 320))
        y_unpad_min, y_unpad_max = np.where(unpadded >= 0.5)[0].min(), np.where(unpadded >= 0.5)[0].max()

        # Direct resize shifts y down by ~40-60px
        self.assertNotEqual(y_direct_min, 68)
        # Unpadded matches original coordinates [68..116] within 1px
        self.assertAlmostEqual(y_unpad_min, 68, delta=1)
        self.assertAlmostEqual(y_unpad_max, 115, delta=1)

        # Test ROI offset transformation
        poly = np.array([[100, 50], [200, 50], [200, 150], [100, 150]], dtype=np.float32)
        mock_res = MockYoloResult([[100, 50, 200, 150]], [0.90], [0], [poly])
        p_info_roi = extract_packet_pieces(mock_res, (self.h, self.w), roi_offset=(30, 40), verbose=False)
        piece_bbox = p_info_roi["pieces"][0]["bbox"]
        # Expected bbox shifted by (30, 40): x: 130, y: 90
        self.assertAlmostEqual(piece_bbox[0], 130, delta=1)
        self.assertAlmostEqual(piece_bbox[1], 90, delta=1)
        print(f"[PASS] Test 7 (Coordinate unpadding & ROI): direct_y=[{y_direct_min}..{y_direct_max}], unpadded_y=[{y_unpad_min}..{y_unpad_max}], shifted_bbox={piece_bbox}")

    def test_8_real_model_inference_on_sample_frame(self):
        """Case 8: Actual model inference on historical invalid sample 9 from user's bug report."""
        sample_path = Path(r"D:\Project\Embsys\Packet-cut-final\packet_cut_outputs\verification_debug\sample_1794575407282_samp9_invalid_orig.jpg")
        model_dir = Path(r"D:\Project\Embsys\Packet-cut-new\models")
        model_path = model_dir / "packet_seg_yolov8s_320.onnx"
        cut_model_path = model_dir / "final(cut_detection).pt"

        if not sample_path.exists() or not model_path.exists() or not cut_model_path.exists():
            self.skipTest("Sample image or model weights not present on disk")

        from ultralytics import YOLO
        frame = cv2.imread(str(sample_path))
        self.assertIsNotNone(frame)

        pkt_model = YOLO(str(model_path), task="segment")
        cut_model = YOLO(str(cut_model_path))

        pkt_res = pkt_model.predict(frame, imgsz=320, conf=0.12, verbose=False)[0]
        cut_res = cut_model.predict(frame, conf=0.25, verbose=False)[0]

        p_info = extract_packet_pieces(pkt_res, frame.shape[:2], reference_area=55000, min_conf=0.12, verbose=False)
        cut_eval = evaluate_cut_model_prediction(cut_res, model=cut_model, cut_class_id=0, min_conf=0.25)

        self.assertEqual(p_info["accepted_piece_count"], 2, f"Expected 2 accepted pieces, got: {p_info['accepted_piece_count']}")
        self.assertTrue(cut_eval["is_cut_detected"], f"Cut model should detect CUT: {cut_eval}")

        a, b = p_info["pieces"][:2]
        gap_info = analyze_packet_gap(a["mask"], b["mask"], frame=frame, reference_area=55000)

        self.assertTrue(gap_info["valid"], f"Gap verification failed: {gap_info['reason']}")
        self.assertGreaterEqual(gap_info["min_gap"], 30.0)
        self.assertGreaterEqual(gap_info["coverage_pct"], 40.0)
        self.assertEqual(gap_info["overlap_px"], 0)

        state, reason = evaluate_final_cut_decision(
            p_info["pieces"], cut_eval, last_gap_info=gap_info,
            sample_count=10, valid_count=9, stable_streak=6,
            min_verify_samples=8, min_valid_sample_ratio=0.70, min_streak=5
        )
        self.assertEqual(state, "CUT_PHYSICALLY_CONFIRMED")
        print(f"[PASS] Test 8 (Real model inference on sample 9): pieces={len(p_info['pieces'])}, min_gap={gap_info['min_gap']:.1f}px, cov={gap_info['coverage_pct']:.1f}% -> {state}")


if __name__ == "__main__":
    unittest.main()

