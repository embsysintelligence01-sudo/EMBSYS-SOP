"""Unit Tests for Centralized Model Manager and Stage-Wise On-Demand Inference.

Validates the 12 core requirements:
1. All 8 model components initialize only once.
2. Stage 1 executes only packet segmentation.
3. Stage 2 executes packet segmentation, OBB and OCR.
4. Stage 3 executes packet segmentation and strip segmentation.
5. Stage 4 executes packet, hand and scissor models.
6. Stage 5 executes ONLY packet segmentation and cut model.
7. Stage 5 does not call hand or scissor inference.
8. Stage 5 checks two pieces and a stable visible gap.
9. Stage 6 executes packet segmentation and jewellery tracking.
10. Stage 7 executes packet, jewellery tracking and hand segmentation.
11. Inactive models receive zero inference calls.
12. Model instances are reused across stage transitions.
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))
if str(_APP_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT.parent))

import numpy as np
from workflow_monitor.config import WorkflowMonitorConfig
from workflow_monitor.models import ModelManager, STAGE_MODELS, UnifiedModelManager
from packet_cut.desktop.state_machine import AutoStage


class MockVisionModel:
    """Mock vision model tracking predict invocations."""
    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count = 0

    def predict(self, *args, **kwargs) -> Any:
        self.call_count += 1
        mock_res = MagicMock()
        mock_res.boxes = MagicMock()
        mock_res.boxes.__len__.return_value = 0
        mock_res.masks = None
        return [mock_res]


class TestCentralizedModelManager(unittest.TestCase):
    """Test suite verifying ModelManager architecture and stage execution rules."""

    def setUp(self) -> None:
        self.cfg = WorkflowMonitorConfig.load()
        self.mock_models = {
            "packet": MockVisionModel("packet-seg.onnx"),
            "number": MockVisionModel("packet-no-obb.onnx"),
            "topstrip": MockVisionModel("strip-seg.onnx"),
            "hand": MockVisionModel("hand-seg.onnx"),
            "scissor": MockVisionModel("scissor-seg.pt"),
            "cut": MockVisionModel("cut-seg.pt"),
            "jewel": MockVisionModel("jeweltrack.onnx"),
            "ocr": MagicMock(name="PaddleOCR"),
        }
        self.mgr = ModelManager(self.cfg, preloaded_models=self.mock_models)
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_01_all_eight_components_initialized_once(self) -> None:
        """1. All eight model components initialize only once."""
        self.assertEqual(len(self.mgr.models), 8)
        for key in ("packet", "number", "topstrip", "hand", "scissor", "cut", "jewel", "ocr"):
            self.assertIn(key, self.mgr.models)
            self.assertIsNotNone(self.mgr.get_model(key))
            self.assertTrue(self.mgr.has(key))

        # Re-calling load_all_models must not re-instantiate existing models
        pkt_before = id(self.mgr.get_model("packet_seg"))
        self.mgr.load_all_models()
        pkt_after = id(self.mgr.get_model("packet_seg"))
        self.assertEqual(pkt_before, pkt_after)

    def test_02_stage1_executes_only_packet_segmentation(self) -> None:
        """2. Stage 1 executes only packet segmentation. Inactive models = 0 calls."""
        res = self.mgr.run_stage("EMPTY_TESTBED", self.dummy_frame)
        self.assertIn("packet_seg", res)
        self.assertEqual(self.mock_models["packet"].call_count, 1)
        self.assertEqual(self.mgr.inference_counts["packet_seg"], 1)

        # Inactive models must receive zero calls
        self.assertEqual(self.mock_models["number"].call_count, 0)
        self.assertEqual(self.mock_models["topstrip"].call_count, 0)
        self.assertEqual(self.mock_models["hand"].call_count, 0)
        self.assertEqual(self.mock_models["scissor"].call_count, 0)
        self.assertEqual(self.mock_models["cut"].call_count, 0)
        self.assertEqual(self.mock_models["jewel"].call_count, 0)

    def test_03_stage2_executes_packet_obb_and_ocr(self) -> None:
        """3. Stage 2 executes packet segmentation, OBB and OCR."""
        crop = np.zeros((60, 120, 3), dtype=np.uint8)
        res = self.mgr.run_stage(AutoStage.WAIT_FRONTSIDE_AND_OCR, self.dummy_frame, crop=crop, overrides={"ocr_text": ("324894", 0.99)})
        self.assertIn("packet_seg", res)
        self.assertIn("packet_no_obb", res)
        self.assertIn("paddle_ocr", res)

        self.assertEqual(self.mock_models["packet"].call_count, 1)
        self.assertEqual(self.mock_models["number"].call_count, 1)
        self.assertEqual(self.mgr.inference_counts["paddle_ocr"], 1)

        # Unrelated models skipped
        self.assertEqual(self.mock_models["hand"].call_count, 0)
        self.assertEqual(self.mock_models["scissor"].call_count, 0)
        self.assertEqual(self.mock_models["topstrip"].call_count, 0)
        self.assertEqual(self.mock_models["cut"].call_count, 0)

    def test_04_stage3_executes_packet_and_strip_segmentation(self) -> None:
        """4. Stage 3 executes packet segmentation and strip segmentation."""
        res = self.mgr.run_stage("BACK_STRIP", self.dummy_frame)
        self.assertIn("packet_seg", res)
        self.assertIn("strip_seg", res)

        self.assertEqual(self.mock_models["packet"].call_count, 1)
        self.assertEqual(self.mock_models["topstrip"].call_count, 1)

        # Inactive models receive zero calls
        self.assertEqual(self.mock_models["scissor"].call_count, 0)
        self.assertEqual(self.mock_models["cut"].call_count, 0)
        self.assertEqual(self.mock_models["hand"].call_count, 0)

    def test_05_stage4_executes_packet_hand_and_scissor(self) -> None:
        """5. Stage 4 executes packet, hand and scissor models."""
        res = self.mgr.run_stage(AutoStage.WAIT_CUTTING_ACTIVITY, self.dummy_frame)
        self.assertIn("packet_seg", res)
        self.assertIn("hand_seg", res)
        self.assertIn("scissor_seg", res)

        self.assertEqual(self.mock_models["packet"].call_count, 1)
        self.assertEqual(self.mock_models["hand"].call_count, 1)
        self.assertEqual(self.mock_models["scissor"].call_count, 1)

        # Strip, cut, OCR inactive
        self.assertEqual(self.mock_models["topstrip"].call_count, 0)
        self.assertEqual(self.mock_models["cut"].call_count, 0)

    def test_06_stage5_executes_only_packet_and_cut(self) -> None:
        """6. Stage 5 executes ONLY packet segmentation and cut model."""
        res = self.mgr.run_stage(AutoStage.VERIFY_PHYSICAL_CUT, self.dummy_frame)
        self.assertIn("packet_seg", res)
        self.assertIn("cut_seg", res)

        self.assertEqual(self.mock_models["packet"].call_count, 1)
        self.assertEqual(self.mock_models["cut"].call_count, 1)

    def test_07_stage5_does_not_call_hand_or_scissor(self) -> None:
        """7. Stage 5 does not call hand or scissor inference."""
        self.mgr.run_stage("CUT_VERIFICATION", self.dummy_frame)
        self.assertEqual(self.mock_models["hand"].call_count, 0)
        self.assertEqual(self.mock_models["scissor"].call_count, 0)
        self.assertEqual(self.mgr.inference_counts["hand_seg"], 0)
        self.assertEqual(self.mgr.inference_counts["scissor_seg"], 0)

    def test_08_stage5_cut_verification_two_pieces_and_gap(self) -> None:
        """8. Stage 5 checks two pieces and a stable visible gap."""
        import packet_cut_core

        # Synthesize two separated pieces with a clear 40px gap
        mask_a = np.zeros((480, 640), dtype=np.uint8)
        mask_b = np.zeros((480, 640), dtype=np.uint8)
        mask_a[100:300, 100:280] = 255
        mask_b[100:300, 320:500] = 255

        gap_info = packet_cut_core.analyze_packet_gap(mask_a, mask_b, reference_area=60000.0)
        self.assertTrue(gap_info["valid"])
        self.assertGreater(gap_info["min_gap"], 4.0)
        self.assertEqual(gap_info["overlap_px"], 0)

        # Touching pieces must fail gap check
        mask_touch = np.zeros((480, 640), dtype=np.uint8)
        mask_touch[100:300, 279:450] = 255
        gap_touch = packet_cut_core.analyze_packet_gap(mask_a, mask_touch, reference_area=60000.0)
        self.assertFalse(gap_touch["valid"])

    def test_09_stage6_executes_packet_and_jeweltrack(self) -> None:
        """9. Stage 6 executes packet segmentation and jewellery tracking."""
        res = self.mgr.run_stage(AutoStage.WAIT_UPRIGHT_MOUTH, self.dummy_frame)
        self.assertIn("packet_seg", res)
        self.assertIn("jeweltrack", res)

        self.assertEqual(self.mock_models["packet"].call_count, 1)
        self.assertEqual(self.mock_models["jewel"].call_count, 1)

        # Inactive models
        self.assertEqual(self.mock_models["scissor"].call_count, 0)
        self.assertEqual(self.mock_models["topstrip"].call_count, 0)

    def test_10_stage7_executes_packet_jeweltrack_and_hand(self) -> None:
        """10. Stage 7 executes packet, jewellery tracking and hand segmentation."""
        res = self.mgr.run_stage(AutoStage.WAIT_JEWELLERY_AND_COUNT, self.dummy_frame)
        self.assertIn("packet_seg", res)
        self.assertIn("jeweltrack", res)
        self.assertIn("hand_seg", res)

        self.assertEqual(self.mock_models["packet"].call_count, 1)
        self.assertEqual(self.mock_models["jewel"].call_count, 1)
        self.assertEqual(self.mock_models["hand"].call_count, 1)

        # Scissors, strip, cut inactive
        self.assertEqual(self.mock_models["scissor"].call_count, 0)
        self.assertEqual(self.mock_models["topstrip"].call_count, 0)
        self.assertEqual(self.mock_models["cut"].call_count, 0)

    def test_11_inactive_models_receive_zero_inference_calls(self) -> None:
        """11. Inactive models receive zero inference calls."""
        # Run Stage 1 only
        self.mgr.run_stage("EMPTY_TESTBED", self.dummy_frame)
        inactive = ["packet_no_obb", "strip_seg", "hand_seg", "scissor_seg", "cut_seg", "jeweltrack", "paddle_ocr"]
        for inact in inactive:
            self.assertEqual(self.mgr.inference_counts[inact], 0, f"Expected 0 calls for {inact}")

    def test_12_model_instances_reused_across_transitions(self) -> None:
        """12. Model instances are reused across stage transitions."""
        pkt_before = id(self.mgr.get_model("packet_seg"))
        jwl_before = id(self.mgr.get_model("jeweltrack"))

        # Transition through all stages
        for stage in STAGE_MODELS.keys():
            self.mgr.run_stage(stage, self.dummy_frame, overrides={"ocr_text": ("123456", 1.0)})

        pkt_after = id(self.mgr.get_model("packet_seg"))
        jwl_after = id(self.mgr.get_model("jeweltrack"))

        self.assertEqual(pkt_before, pkt_after)
        self.assertEqual(jwl_before, jwl_after)

    def test_13_scissor_roi_and_bottom_rail_rejection(self) -> None:
        """13. Scissor detection rejects bottom rail artifacts and respects ROI."""
        import cv2
        import torch
        import packet_cut_core

        # Packet at [110..340, 150..390]
        pkt_mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(pkt_mask, (110, 150), (340, 390), 255, -1)

        # 1. Spurious bottom rail detection (wide horizontal bar at y=410..470)
        rail_box = torch.tensor([[20.0, 410.0, 500.0, 470.0]])
        rail_conf = torch.tensor([0.85])
        rail_cls = torch.tensor([0])
        mock_rail = MagicMock()
        mock_boxes_rail = MagicMock()
        mock_boxes_rail.__len__.return_value = 1
        mock_boxes_rail.xyxy = rail_box
        mock_boxes_rail.conf = rail_conf
        mock_boxes_rail.cls = rail_cls
        mock_rail.boxes = mock_boxes_rail
        mock_rail.masks = None

        roi = (40, 40, 600, 400)
        valid, mask, diags = packet_cut_core.evaluate_scissor_detection(
            mock_rail, None, (480, 640), packet_mask=pkt_mask, roi=roi
        )
        self.assertFalse(valid, "Bottom rail artifact must be rejected!")
        self.assertTrue(
            any("BOTTOM_RAIL" in d.get("reason", "") or "OUTSIDE_ROI" in d.get("reason", "") or "BELOW_PACKET" in d.get("reason", "") for d in diags)
        )

        # 2. Valid scissor interacting at cutting region inside ROI
        valid_box = torch.tensor([[180.0, 160.0, 260.0, 230.0]])
        valid_conf = torch.tensor([0.88])
        valid_cls = torch.tensor([0])
        mock_valid = MagicMock()
        mock_boxes_valid = MagicMock()
        mock_boxes_valid.__len__.return_value = 1
        mock_boxes_valid.xyxy = valid_box
        mock_boxes_valid.conf = valid_conf
        mock_boxes_valid.cls = valid_cls
        mock_valid.boxes = mock_boxes_valid
        mock_valid.masks = None

        valid, mask, diags = packet_cut_core.evaluate_scissor_detection(
            mock_valid, None, (480, 640), packet_mask=pkt_mask, roi=roi
        )
        self.assertTrue(valid, "Legitimate scissor inside ROI must be accepted!")


if __name__ == "__main__":
    unittest.main(verbosity=2)

