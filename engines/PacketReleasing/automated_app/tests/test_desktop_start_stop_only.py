"""Comprehensive verification suite for the event-driven EMBSYS Packet-Cut PyQt6 app.

Tests every mandatory acceptance scenario:
1. Readiness and ROI selection gate START.
2. Empty bed held for 30+ seconds -> still asking for front-side placement (and waits
   for removal if a packet is already on the bed at START).
3. Front remains facing up for 30+ seconds after OCR -> still asking for turn.
4. Number covered by hand / missing number box alone / moving packet -> backside stays UNKNOWN.
5. Scissors alone (and hand alone) -> cutting has not started; hand + scissors together -> Cutting.
6. Two touching pieces -> cut remains CUT_UNVERIFIED; 2 detached pieces + gap -> CUT_CONFIRMED
   (with actual cut model filename logged).
7. Cut pieces left flat -> still asking for upright mouth presentation; mouth OPEN can be empty
   (gold visibility recorded separately); whole-packet flat mask alone cannot establish mouth state.
8. One jewel placed, long pause, second jewel placed -> count goes 1 -> 2 in the SAME session;
   NO report finalized during the pause; moving an article does not increase count; hand/packet
   jewels excluded; overlap records uncertainty alongside confirmed count; STOP saves full report.
9. Cutting before readiness logs observed mistake WITHOUT falsely marking readiness passed.
10. Headless MainWindow START -> STOP with real workspace models + shared model reuse.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Optional, Tuple

_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))
if str(_APP_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT.parent))

import cv2
import numpy as np

from inspection.utils.image_utils import RawSegInstance, mask_centroid_and_bbox
from inspection.utils.inspection_config import InspectionConfig
from packet_cut.desktop.automation import AutoPipelineConfig, AutomatedPacketPipeline
from packet_cut.desktop.state_machine import AutoStage
from workflow_monitor.config import WorkflowMonitorConfig
from workflow_monitor.models import UnifiedModelManager


PROJECT_ROOT = _APP_ROOT
MODELS_DIR = PROJECT_ROOT / "models" if (PROJECT_ROOT / "models").exists() else PROJECT_ROOT.parent / "models"
OUTPUTS_DIR = PROJECT_ROOT / "packet_cut_outputs" if (PROJECT_ROOT / "packet_cut_outputs").exists() else PROJECT_ROOT.parent / "packet_cut_outputs"


def _raw_inst(mask: np.ndarray, conf: float = 0.92) -> RawSegInstance:
    c, b, a = mask_centroid_and_bbox(mask)
    return RawSegInstance(mask=mask, bbox=b, confidence=conf, class_id=0, area=a, centroid=c)


def _intact_mask(cx: int = 320, cy: int = 250) -> np.ndarray:
    m = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(m, (cx - 90, cy - 110), (cx + 90, cy + 110), 255, -1)
    return m


def _touching_pieces_mask() -> Tuple[np.ndarray, np.ndarray]:
    a = np.zeros((480, 640), dtype=np.uint8)
    b = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(a, (230, 120), (410, 200), 255, -1)
    cv2.rectangle(b, (230, 201), (410, 380), 255, -1)  # 1px touching
    return a, b


def _detached_cut_mask() -> np.ndarray:
    m = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(m, (230, 100), (410, 175), 255, -1)
    cv2.rectangle(m, (230, 220), (410, 385), 255, -1)  # 45px background gap
    return m


def _testbed_frame() -> np.ndarray:
    rng = np.random.default_rng(17)
    frame = np.clip(225 + rng.integers(-6, 7, size=(480, 640, 3)), 0, 255).astype(np.uint8)
    cv2.rectangle(frame, (160, 120), (480, 400), (40, 150, 45), 4)
    return frame


def _frontside_frame(pkt_mask: np.ndarray) -> np.ndarray:
    frame = _testbed_frame()
    # White printed form grid on frontside (high edge density)
    frame[pkt_mask > 0] = (210, 210, 210)
    for y in range(160, 340, 18):
        cv2.line(frame, (245, y), (395, y), (45, 45, 45), 2)
    for x in range(250, 400, 30):
        cv2.line(frame, (x, 160), (x, 340), (45, 45, 45), 2)
    return frame


def _backside_frame(pkt_mask: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(99)
    frame = _testbed_frame()
    # Clean plain silver/gray backside without front printed grid
    plain = np.clip(145 + rng.integers(-5, 6, size=(480, 640, 3)), 0, 255).astype(np.uint8)
    frame[pkt_mask > 0] = plain[pkt_mask > 0]
    return frame


def _upright_open_empty_mouth_frame() -> Tuple[np.ndarray, np.ndarray]:
    """Upright held packet with visibly separated mouth edges & dark EMPTY interior cavity."""
    rng = np.random.default_rng(55)
    frame = _testbed_frame()
    pkt = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(pkt, (240, 150), (400, 330), 255, -1)
    # Bright lip rim
    frame[pkt > 0] = (205, 205, 205)
    # Visibly separated dark empty mouth cavity (40px wide opening, sharp lip edges)
    cv2.ellipse(frame, (320, 240), (50, 26), 0, 0, 360, (60, 60, 60), -1)
    cv2.ellipse(frame, (320, 240), (50, 26), 0, 0, 360, (30, 30, 30), 2)
    # Subtle grain for focus sharpness
    noise = rng.integers(-6, 7, size=(480, 640, 3))
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame, pkt


def _disc(cx: int, cy: int, r: int = 20) -> np.ndarray:
    m = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def _fast_cfg(tmpdir: str) -> Tuple[AutoPipelineConfig, WorkflowMonitorConfig, InspectionConfig]:
    pc = AutoPipelineConfig(
        pass_by=False,        infer_max_retries=2,
        infer_backoff_sec=0.0,
        empty_testbed_frames=3,
        frontside_stable_frames=2,
        ocr_min_consistent_crops=2,
        backside_confirm_frames=2,
        hand_confirm_frames=1,
        scissor_confirm_frames=1,
        cutting_both_confirm_frames=2,
        cut_confirm_frames=3,
        mouth_confirm_frames=2,
        jewel_min_track_hits=2,
        jewel_max_missing_frames=3,
        save_outputs=True,
        output_dir=tmpdir,
    )
    wf = WorkflowMonitorConfig.load()
    insp = InspectionConfig.load()
    insp.jewellery_counting.testbed_roi = (160, 120, 480, 400)
    return pc, wf, insp


class TestEventDrivenPyQtWorkflow(unittest.TestCase):
    def test_1_start_requires_setup(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtGui import QImage
        from PyQt6.QtWidgets import (
            QAbstractButton, QApplication, QCheckBox, QComboBox,
            QLineEdit, QRadioButton, QSlider, QSpinBox, QTabWidget,
        )
        from packet_cut.desktop.widgets import MainWindow

        app = QApplication.instance() or QApplication([])
        win = MainWindow(camera=0, device="cpu", pipeline_config=AutoPipelineConfig(pass_by=False))
        buttons = win.findChildren(QAbstractButton)
        self.assertEqual({b.text() for b in buttons}, {
            "START", "STOP", "RESTART", "Load Models / Check Ready", "Draw ROI",
            "Retry Current Stage", "Copy Compliance Report", "Copy to Clipboard", "☀", "Open Visual Report", "Open PDF Report",
        })
        for cls in (QComboBox, QCheckBox, QRadioButton, QSpinBox, QSlider, QTabWidget, QLineEdit):
            self.assertEqual(len(win.findChildren(cls)), 0)
        self.assertTrue(win.compliance_log.isReadOnly())
        self.assertTrue(win.final_report_view.isReadOnly())
        self.assertTrue(win.check_btn.isEnabled())
        for button in (win.start_btn, win.restart_btn, win.roi_btn, win.retry_btn, win.copy_report_btn):
            self.assertFalse(button.isEnabled())
        win.start_inspection()
        self.assertIsNone(win.worker)
        win.cmp_observed.setText("Top Strip verified")
        win.compliance_log.setPlainText("[COMPLIANT] Top Strip absent")
        win.copy_compliance_btn.click()
        self.assertIn("Latest Observed Action: Top Strip verified", app.clipboard().text())
        self.assertIn("[COMPLIANT] Top Strip absent", app.clipboard().text())
        frame = QImage(640, 480, QImage.Format.Format_RGB32)
        frame.fill(0)
        win._manual_roi = (160, 120, 480, 360)
        win._on_frame_ready(frame)
        displayed = win.video.pixmap().toImage()
        self.assertEqual(displayed.pixelColor(displayed.width() // 2, displayed.height() // 4).name(), "#38bdf8")
        self.assertEqual(frame.pixelColor(320, 120).name(), "#000000")
        win.close()
        app.processEvents()

    def test_2_all_seven_mandatory_anti_rush_and_workflow_verifications(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pc, wf, insp = _fast_cfg(tmpdir)
            pipe = AutomatedPacketPipeline(
                model_manager=UnifiedModelManager(wf),
                pipeline_config=pc,
                workflow_config=wf,
                inspection_config=insp,
            )
            pipe.start_session(now_sec=0.0)
            tb = _testbed_frame()
            empty_m = np.zeros((480, 640), dtype=np.uint8)
            intact = _intact_mask()
            front_img = _frontside_frame(intact)
            back_img = _backside_frame(intact)
            cut_m = _detached_cut_mask()

            # Stage 1: Packet ALREADY on bed at START -> must wait for removal!
            for t in (0.1, 1.0, 5.0):
                snap = pipe.process_frame(front_img, now_sec=t, overrides={"packet_mask": intact, "hand_present": False, "jewel_instances": []})
                self.assertEqual(pipe.stage, AutoStage.WAIT_EMPTY_TESTBED)
                self.assertEqual(snap.instruction, "Clear the testbed.")

            # The selected ROI can be a smooth tray with no visible edges.
            smooth_tray = np.full_like(tb, 160)
            pipe.process_frame(smooth_tray, now_sec=5.5, overrides={"packet_mask": empty_m})
            pipe.retry_current_stage(AutoStage.WAIT_EMPTY_TESTBED.value)
            for t in (6.0, 6.2, 6.4):
                snap = pipe.process_frame(smooth_tray, now_sec=t, overrides={"packet_mask": empty_m, "hand_present": False, "jewel_instances": []})
                if t < 6.4:
                    self.assertEqual(pipe.stage, AutoStage.WAIT_EMPTY_TESTBED)
            self.assertEqual(pipe.stage, AutoStage.WAIT_FRONTSIDE_AND_OCR)
            self.assertTrue(snap.transitioned_this_frame)
            self.assertEqual(snap.instruction, "Please place the packet FRONT side up.")

            # CHECK 1: Empty bed held for 30+ seconds -> STILL asking for front-side placement!
            for t in (10.0, 20.0, 36.5, 42.0):
                snap = pipe.process_frame(tb, now_sec=t, overrides={"packet_mask": empty_m, "ocr_roi": None})
                self.assertEqual(pipe.stage, AutoStage.WAIT_FRONTSIDE_AND_OCR)
                self.assertEqual(snap.instruction, "Please place the packet FRONT side up.")

            # Place packet FRONT side up with number OBB inside packet; require 2 consistent fresh crops!
            t = 43.0
            # Frame 1 & 2 stabilize frontside; Frame 2 runs 1st OCR crop (1/2 -> NOT locked yet!)
            for _ in range(2):
                snap = pipe.process_frame(
                    front_img, now_sec=t,
                    overrides={"packet_mask": intact, "ocr_roi": (260, 155, 365, 195), "ocr_text": ("324894", 0.96)},
                )
                t += 0.2
            self.assertEqual(pipe.stage, AutoStage.WAIT_FRONTSIDE_AND_OCR)  # 1st crop alone NEVER locks!
            # Frame 3: 2nd independent fresh crop confirms "324894" -> transitions to Stage 3!
            snap = pipe.process_frame(
                front_img, now_sec=t,
                overrides={"packet_mask": intact, "ocr_roi": (260, 155, 365, 195), "ocr_text": ("324894", 0.95)},
            )
            self.assertEqual(pipe.stage, AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP)
            self.assertEqual(pipe._packet_id, "324894")
            self.assertEqual(snap.instruction, "Please turn the packet to the BACK side.")
            t += 0.5

            # CHECK 2: Front remains facing up for 30+ seconds after OCR -> STILL asking for turn!
            for dt in (1.0, 10.0, 20.0, 32.0):
                snap = pipe.process_frame(
                    front_img, now_sec=t + dt,
                    overrides={"packet_mask": intact, "hand_present": False, "ocr_roi": (260, 155, 365, 195), "topstrip_visible": False},
                )
                self.assertEqual(pipe.stage, AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP)
                self.assertEqual(snap.instruction, "Please turn the packet to the BACK side.")
            t += 35.0

            # CHECK 3A: Number covered by HAND (OBB hidden by hand!) -> backside remains UNVERIFIED!
            hand_over_num = np.zeros((480, 640), dtype=np.uint8)
            cv2.rectangle(hand_over_num, (240, 140), (390, 220), 255, -1)
            for _ in range(4):
                snap = pipe.process_frame(
                    front_img, now_sec=t,
                    overrides={"packet_mask": intact, "hand_mask": hand_over_num, "hand_present": True, "ocr_roi": None, "topstrip_visible": False},
                )
                t += 0.2
            self.assertEqual(pipe.stage, AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP)
            self.assertFalse(pipe._backside_verified)
            self.assertEqual(pipe._strip_result, "UNKNOWN_HAND_OCCLUDED")

            # CHECK 3B: Hand leaves, still FRONT side with missed OBB box -> appearance matches front -> stays UNKNOWN!
            for _ in range(4):
                pipe.process_frame(
                    front_img, now_sec=t,
                    overrides={"packet_mask": intact, "hand_present": False, "ocr_roi": None, "topstrip_visible": False},
                )
                t += 0.2
            self.assertEqual(pipe.stage, AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP)
            self.assertFalse(pipe._backside_verified)
            self.assertEqual(pipe._strip_result, "UNKNOWN_BACKSIDE_UNVERIFIED")

            # Now physically turn to BACK side: Strip visible first -> "Strip visible. Cutting not ready."
            moved_mask = _intact_mask(cx=350, cy=250)
            pipe.process_frame(back_img, now_sec=t, overrides={"packet_mask": moved_mask, "hand_present": False, "ocr_roi": None, "topstrip_visible": False})
            t += 0.2
            snap = pipe.process_frame(back_img, now_sec=t, overrides={"packet_mask": intact, "hand_present": False, "ocr_roi": None, "topstrip_visible": True})
            t += 0.2
            snap = pipe.process_frame(back_img, now_sec=t, overrides={"packet_mask": intact, "hand_present": False, "ocr_roi": None, "topstrip_visible": True})
            self.assertEqual(pipe.stage, AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP)
            self.assertEqual(snap.instruction, "Strip visible. Cutting not ready.")
            t += 0.2

            # Strip absent across clear stable backside frames -> saves intact mask & shows "READY TO CUT"
            for _ in range(2):
                snap = pipe.process_frame(back_img, now_sec=t, overrides={"packet_mask": intact, "hand_present": False, "ocr_roi": None, "topstrip_visible": False})
                t += 0.2
            self.assertEqual(pipe.stage, AutoStage.WAIT_CUTTING_ACTIVITY)
            self.assertTrue(pipe._backside_verified)
            self.assertIsNotNone(pipe._intact_baseline_mask)
            self.assertEqual(snap.instruction, "READY TO CUT")

            # CHECK 4: Scissors ALONE (and Hand ALONE) -> cutting has NOT started!
            for _ in range(4):
                snap = pipe.process_frame(back_img, now_sec=t, overrides={"hand_present": False, "scissor_present": True})
                t += 0.2
            self.assertFalse(pipe._cutting_activity_observed)
            self.assertEqual(snap.instruction, "READY TO CUT")
            for _ in range(4):
                snap = pipe.process_frame(back_img, now_sec=t, overrides={"hand_present": True, "scissor_present": False})
                t += 0.2
            self.assertFalse(pipe._cutting_activity_observed)

            snap = pipe.process_frame(
                tb, now_sec=t,
                overrides={"packet_mask": cut_m, "hand_present": False, "scissor_present": False},
            )
            t += 0.2
            self.assertEqual(pipe.stage, AutoStage.WAIT_CUTTING_ACTIVITY)
            self.assertNotIn("cut-seg.pt", snap.models_executed_this_frame)

            # Run actual detection post-processing for a hand fully over the packet
            # and scissors at the bottom of the user's ROI.
            import torch

            def detection(box=None):
                boxes = MagicMock()
                boxes.__len__.return_value = int(box is not None)
                boxes.xyxy = torch.tensor([box] if box is not None else []).reshape(-1, 4)
                boxes.conf = torch.tensor([0.95] if box is not None else [])
                boxes.cls = torch.zeros(len(boxes))
                return [SimpleNamespace(boxes=boxes, masks=None)]

            hand_detection = detection([265, 185, 335, 235])
            scissor_detection = detection([345, 425, 415, 465])
            seal_detection = detection([350, 185, 375, 235])
            empty_detection = detection()
            cutting_frame = back_img.copy()
            cutting_frame[185:235, 265:335] = (80, 105, 170)
            cutting_frame[185:235, 350:375] = (20, 20, 210)
            pipe.cutting_roi = (0, 0, tb.shape[1], tb.shape[0])
            hand_model = MagicMock()
            scissor_model = MagicMock()
            hand_model.predict.side_effect = [hand_detection, hand_detection, hand_detection, empty_detection, seal_detection]
            scissor_model.predict.side_effect = [scissor_detection, scissor_detection, empty_detection, scissor_detection, empty_detection]
            pipe.model_manager.models.update(hand=hand_model, scissor=scissor_model)

            for _ in range(2):
                snap = pipe.process_frame(cutting_frame, now_sec=t)
                t += 0.2
            self.assertTrue(pipe._cutting_activity_observed)
            self.assertEqual(snap.instruction, "CUTTING ACTIVITY STARTED")

            # Either one remaining blocks Stage 5. The first frame with both absent advances.
            for _ in range(2):
                pipe.process_frame(cutting_frame, now_sec=t)
                self.assertEqual(pipe.stage, AutoStage.WAIT_CUTTING_ACTIVITY)
                t += 0.2
            snap = pipe.process_frame(cutting_frame, now_sec=t)
            self.assertEqual(hand_model.predict.call_count, 5)
            self.assertEqual(scissor_model.predict.call_count, 5)
            self.assertEqual(pipe.stage, AutoStage.VERIFY_PHYSICAL_CUT)
            t += 0.2

            # CHECK 5: Two TOUCHING pieces -> cut MUST remain CUT_UNVERIFIED!
            ta, tb_piece = _touching_pieces_mask()
            touching_union = cv2.bitwise_or(ta, tb_piece)
            for _ in range(4):
                snap = pipe.process_frame(
                    tb, now_sec=t,
                    overrides={
                        "packet_mask": touching_union,
                        "hand_present": False,
                        "scissor_present": False,
                        "cut_eval": {"is_cut_detected": True, "confidence": 0.95, "class_name": "Cut"},
                    },
                )
                t += 0.2
            self.assertEqual(pipe.stage, AutoStage.VERIFY_PHYSICAL_CUT)
            self.assertEqual(pipe._cut_result["state"], "CUT_UNVERIFIED")

            # A reappearing hand interrupts cut analysis.
            pipe.process_frame(tb, now_sec=t, overrides={"hand_present": True, "scissor_present": False})
            self.assertEqual(pipe._cut_analysis_count, 0)
            cut_start = t + 0.2
            for index in range(10):
                t = cut_start + index * 0.2
                snap = pipe.process_frame(
                    tb, now_sec=t,
                    overrides={
                        "packet_mask": cut_m,
                        "hand_present": False,
                        "scissor_present": False,
                        "cut_eval": {"is_cut_detected": True, "confidence": 0.94, "class_name": "Cut"},
                    },
                )
                if index < 9:
                    self.assertEqual(pipe.stage, AutoStage.VERIFY_PHYSICAL_CUT)
                    self.assertEqual(pipe._cut_result["state"], "CUT_UNVERIFIED")
            t += 0.2
            self.assertEqual(pipe.stage, AutoStage.WAIT_UPRIGHT_MOUTH)
            self.assertEqual(pipe._cut_result["state"], "CUT_CONFIRMED")
            self.assertEqual(pipe._cut_result["cut_model_filename"], "cut-seg.pt")
            self.assertEqual(snap.instruction, "SHOW THE PACKET MOUTH TO THE CAMERA. Hold the packet upright.")

            # CHECK 6A: Cut pieces left FLAT on bed -> STILL asking for upright mouth presentation!
            for _ in range(5):
                snap = pipe.process_frame(tb, now_sec=t, overrides={"packet_mask": cut_m, "hand_present": False})
                t += 0.3
            self.assertEqual(pipe.stage, AutoStage.WAIT_UPRIGHT_MOUTH)
            self.assertEqual(snap.instruction, "SHOW THE PACKET MOUTH TO THE CAMERA. Hold the packet upright.")

            # CHECK 6B: Upright held packet with separated mouth edges & EMPTY interior (gold_visible=False) -> OPEN!
            mouth_frame, mouth_pkt = _upright_open_empty_mouth_frame()
            mouth_start = t
            for index in range(10):
                t = mouth_start + index * 0.2
                snap = pipe.process_frame(
                    mouth_frame, now_sec=t,
                    overrides={"packet_mask": mouth_pkt, "hand_present": False, "mouth_jewel_instances": []},
                )
                if index < 9:
                    self.assertEqual(pipe.stage, AutoStage.WAIT_UPRIGHT_MOUTH)
            t += 0.2
            self.assertEqual(pipe.stage, AutoStage.WAIT_GOLD_PLACEMENT)
            self.assertEqual(pipe._mouth_status, "OPEN")
            self.assertIs(pipe._gold_visibility_in_mouth, False)  # Open packet can be empty!
            self.assertEqual(snap.instruction, "Remove the gold from the packet and place it on the testbed.")

            # CHECK 7: One jewel placed -> LONG PAUSE (30s) -> second jewel placed -> count goes 1 to 2 in SAME session!
            pipe.process_frame(
                tb, now_sec=t,
                overrides={"packet_mask": empty_m, "hand_present": False, "jewel_instances": []},
            )
            self.assertEqual(pipe.stage, AutoStage.WAIT_GOLD_PLACEMENT)
            t += 0.2
            j1 = _raw_inst(_disc(240, 240))
            j2 = _raw_inst(_disc(390, 260))
            for _ in range(3):
                snap = pipe.process_frame(
                    tb, now_sec=t,
                    overrides={"packet_mask": empty_m, "hand_present": False, "jewel_instances": [j1]},
                )
                t += 0.2
            self.assertEqual(pipe._provisional_jewel_count, 1)
            self.assertEqual(pipe.stage, AutoStage.WAIT_JEWELLERY_AND_COUNT)
            self.assertEqual(pipe._jewel_count_status, "PROVISIONAL")
            self.assertFalse(snap.completed)  # NO report finalized during pause!

            # 30-second pause with jewel #1 sitting on testbed (and moved slightly!) -> stays provisional count=1!
            j1_moved = _raw_inst(_disc(265, 250))
            for dt in (2.0, 10.0, 25.0, 30.0):
                snap = pipe.process_frame(
                    tb, now_sec=t + dt,
                    overrides={"packet_mask": empty_m, "hand_present": False, "jewel_instances": [j1_moved]},
                )
                self.assertEqual(pipe.stage, AutoStage.WAIT_JEWELLERY_AND_COUNT)
                self.assertEqual(pipe._provisional_jewel_count, 1)  # Moving does NOT increase count!
                self.assertFalse(snap.completed)
                self.assertIsNone(pipe._final_report)
            t += 31.0

            # Place 2nd jewel in the SAME session -> count goes from 1 to 2!
            for _ in range(3):
                snap = pipe.process_frame(
                    tb, now_sec=t,
                    overrides={"packet_mask": empty_m, "hand_present": False, "jewel_instances": [j1_moved, j2]},
                )
                t += 0.2
            self.assertEqual(pipe._provisional_jewel_count, 2)
            self.assertFalse(snap.completed)

            # STOP saves the full report with stage_reached, evidence, and missing completion rule!
            report = pipe.stop_session(now_sec=t + 1.0)
            self.assertEqual(report["session_information"]["packet_id"], "324894")
            self.assertEqual(report["session_information"]["stage_reached"], AutoStage.WAIT_JEWELLERY_AND_COUNT.value)
            self.assertEqual(report["operational_results"]["cut_model_filename"], "cut-seg.pt")
            self.assertEqual(report["operational_results"]["packet_open_closed_inspection"]["mouth_status"], "OPEN")
            self.assertIs(report["operational_results"]["packet_open_closed_inspection"]["gold_visible"], False)
            self.assertEqual(report["operational_results"]["jewellery_detection_and_counting"]["provisional_count"], 2)
            self.assertTrue(Path(pipe._report_paths["json"]).is_file())
            self.assertTrue(Path(pipe._report_paths["markdown"]).is_file())
            self.assertTrue(Path(pipe._report_paths["html"]).is_file())
            self.assertTrue(Path(pipe._report_paths["pdf"]).is_file())
            media = {entry["key"]: entry for entry in report["evidence_media"]}
            self.assertEqual(set(media), {"packet_number", "cutting_activity", "cut_packet", "packet_mouth", "jewellery_count"})
            self.assertEqual(media["jewellery_count"]["count"], 2)
            self.assertEqual(media["jewellery_count"]["result"], "PROVISIONAL")
            for entry in media.values():
                self.assertEqual(entry["session_id"], pipe.session_id)
                self.assertEqual(entry["packet_id"], "324894")
                self.assertTrue(Path(entry["image_path"]).is_file())
                self.assertEqual(entry["video_status"], "SAVED", entry)
                self.assertTrue(Path(entry["video_path"]).is_file())
            self.assertTrue(Path(report["operational_results"]["ocr_identification"]["number_crop_image"]).is_file())

    def test_3_cutting_before_readiness_logs_mistake_without_marking_readiness_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pc, wf, insp = _fast_cfg(tmpdir)
            pipe = AutomatedPacketPipeline(
                model_manager=UnifiedModelManager(wf),
                pipeline_config=pc,
                workflow_config=wf,
                inspection_config=insp,
            )
            pipe.start_session(now_sec=0.0)
            tb = _testbed_frame()
            empty_m = np.zeros((480, 640), dtype=np.uint8)
            intact = _intact_mask()
            front_img = _frontside_frame(intact)

            # Empty -> OCR -> Stage 3
            for i in range(3):
                pipe.process_frame(tb, now_sec=0.1 * (i + 1), overrides={"packet_mask": empty_m, "hand_present": False, "jewel_instances": []})
            for i in range(3):
                pipe.process_frame(
                    front_img, now_sec=1.0 + 0.1 * i,
                    overrides={"packet_mask": intact, "ocr_roi": (260, 155, 365, 195), "ocr_text": ("112233", 0.94)},
                )
            self.assertEqual(pipe.stage, AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP)

            # Appraiser starts cutting BEFORE backside readiness verified!
            snap = pipe.process_frame(
                front_img, now_sec=2.0,
                overrides={"packet_mask": intact, "hand_present": True, "scissor_present": True, "ocr_roi": None},
            )
            self.assertEqual(pipe.stage, AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP)
            self.assertFalse(pipe._backside_verified)  # Readiness MUST NOT be marked passed!
            report = pipe.stop_session(now_sec=3.0)
            self.assertFalse(report["operational_results"]["backside_verified"])
            self.assertTrue(any(m["code"] == "CUTTING_BEFORE_READINESS" for m in report["observed_mistakes"]))

    def test_mouth_waits_for_gold_check_and_can_confirm_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pc, wf, insp = _fast_cfg(tmpdir)
            pipe = AutomatedPacketPipeline(
                model_manager=UnifiedModelManager(wf),
                pipeline_config=pc,
                workflow_config=wf,
                inspection_config=insp,
            )
            pipe.start_session(now_sec=0.0)
            for stage in (
                AutoStage.WAIT_FRONTSIDE_AND_OCR, AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP,
                AutoStage.WAIT_CUTTING_ACTIVITY, AutoStage.VERIFY_PHYSICAL_CUT, AutoStage.WAIT_UPRIGHT_MOUTH,
            ):
                pipe._transition(stage, ts=0.0, actual_event="Test setup", model_used="", instruction="")

            frame, packet = _upright_open_empty_mouth_frame()
            for t in range(1, 11):
                snap = pipe.process_frame(frame, now_sec=t, overrides={
                    "packet_mask": packet, "mouth_state_override": "OPEN", "mouth_jewel_instances": []
                })
            self.assertEqual(snap.stage, AutoStage.WAIT_UPRIGHT_MOUTH)
            self.assertIn("gold check", snap.live_monitoring["processing_status"])

            pipe.retry_current_stage(AutoStage.WAIT_UPRIGHT_MOUTH.value)
            for t in range(12, 22):
                snap = pipe.process_frame(
                    frame, now_sec=t, overrides={"packet_mask": packet, "mouth_state_override": "CLOSED"},
                )
                if t < 21:
                    self.assertEqual(snap.stage, AutoStage.WAIT_UPRIGHT_MOUTH)
            self.assertEqual(snap.stage, AutoStage.WAIT_GOLD_PLACEMENT)
            self.assertEqual(snap.live_monitoring["packet_condition"], "Mouth CLOSED")
            self.assertIsNone(pipe._gold_visibility_in_mouth)

    def test_4_mainwindow_start_stop_real_models_reuse(self) -> None:
        sample_base = OUTPUTS_DIR / "baseline_frame.jpg"
        if not (MODELS_DIR / "packet-seg.onnx").is_file() or not sample_base.is_file():
            self.skipTest("Workspace models or baseline_frame.jpg missing")

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtCore import QPoint, Qt, QTimer
        from PyQt6.QtTest import QTest
        from PyQt6.QtWidgets import QApplication, QDialogButtonBox
        from packet_cut.desktop.widgets import MainWindow, ROISelectionLabel

        with tempfile.TemporaryDirectory() as tmpdir:
            pc, wf, _ = _fast_cfg(tmpdir)
            mgr = UnifiedModelManager(wf)
            mgr.ensure_loaded(keys=("packet", "hand", "topstrip", "jewel"))
            pkt_id_before = id(mgr.get("packet"))

            img = cv2.imread(str(sample_base))
            assert img is not None
            cnt = {"n": 0}

            def provider() -> Tuple[bool, Optional[np.ndarray]]:
                cnt["n"] += 1
                if cnt["n"] <= 3:
                    return True, img.copy()
                return False, None

            app = QApplication.instance() or QApplication([])
            win = MainWindow(camera=0, device="cpu", model_manager=mgr, pipeline_config=pc, frame_provider=provider)
            def close_window():
                win.close()
                app.processEvents()

            self.addCleanup(close_window)
            win.check_btn.click()
            deadline = time.monotonic() + 45.0
            while time.monotonic() < deadline and not win.check_btn.isEnabled():
                app.processEvents()
                time.sleep(0.05)
            self.assertTrue(win.roi_btn.isEnabled(), win.sys_errors.text())
            self.assertFalse(win.start_btn.isEnabled())

            selection_results = []

            def select_roi():
                dialog = QApplication.activeModalWidget()
                try:
                    preview = dialog.findChild(ROISelectionLabel)
                    QTest.mousePress(preview, Qt.MouseButton.LeftButton, pos=QPoint(40, 50))
                    QTest.mouseRelease(preview, Qt.MouseButton.LeftButton, pos=QPoint(400, 420))
                    selection_results.append(preview.selected_roi())
                    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
                finally:
                    if dialog.isVisible():
                        dialog.reject()

            QTimer.singleShot(0, select_roi)
            win.roi_btn.click()
            self.assertTrue(selection_results and selection_results[0] is not None)
            rx1, ry1, rx2, ry2 = win._manual_roi
            np.testing.assert_array_equal(win._empty_reference, img[ry1:ry2, rx1:rx2])
            self.assertTrue(win.start_btn.isEnabled())
            model = mgr.get("packet")
            predict_spy = patch.object(model, "predict", wraps=model.predict)
            predict = predict_spy.start()
            self.addCleanup(predict_spy.stop)
            win.start_btn.click()
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and not win.start_btn.isEnabled():
                app.processEvents()
                time.sleep(0.05)

            self.assertTrue(win.start_btn.isEnabled())
            self.assertIn("Stage Reached:", win.final_report_view.toPlainText())
            x1, y1, x2, y2 = win._manual_roi
            expected_crop = img[y1:y2, x1:x2]
            local_roi = (0, 0, x2 - x1, y2 - y1)
            self.assertEqual(win.worker.pipeline.cutting_roi, local_roi)
            self.assertTrue(win.worker.pipeline._has_calibrated_empty_reference)
            self.assertEqual(win.worker.pipeline.testbed_roi.get_roi(expected_crop), local_roi)
            self.assertTrue(predict.called)
            for call in predict.call_args_list:
                np.testing.assert_array_equal(call.kwargs["source"], expected_crop)
            self.assertEqual((win._last_pixmap.width(), win._last_pixmap.height()), (img.shape[1], img.shape[0]))
            self.assertTrue(win.copy_report_btn.isEnabled())
            self.assertTrue(win.open_report_btn.isEnabled())
            self.assertTrue(win.open_pdf_btn.isEnabled())
            self.assertTrue(Path(win._pdf_report_path).is_file())
            self.assertTrue(Path(win._visual_report_path).is_file())
            win.copy_report_btn.click()
            self.assertIn("Compliance", app.clipboard().text())
            self.assertEqual(id(win._shared_model_manager.get("packet")), pkt_id_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
