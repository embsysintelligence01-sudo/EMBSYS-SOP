"""Comprehensive unit & integration tests for the Automated Appraiser Workflow
Monitoring, Compliance & Reporting System (Sections 1–10).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))
if str(_APP_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT.parent))

import cv2
import numpy as np

from inspection.utils.image_utils import RawSegInstance, mask_centroid_and_bbox
from inspection.utils.inspection_config import InspectionConfig
from workflow_monitor.alerts import RealTimeAlertManager
from workflow_monitor.automated_engine import AutomatedStage, AutomatedWorkflowEngine
from workflow_monitor.config import WorkflowMonitorConfig
from workflow_monitor.models import UnifiedModelManager


PROJECT_ROOT = _APP_ROOT
MODELS_DIR = PROJECT_ROOT / "models" if (PROJECT_ROOT / "models").exists() else PROJECT_ROOT.parent / "models"
OUTPUTS_DIR = PROJECT_ROOT / "packet_cut_outputs" if (PROJECT_ROOT / "packet_cut_outputs").exists() else PROJECT_ROOT.parent / "packet_cut_outputs"


def _raw_inst(mask: np.ndarray, conf: float = 0.92) -> RawSegInstance:
    c, b, a = mask_centroid_and_bbox(mask)
    return RawSegInstance(mask=mask, bbox=b, confidence=conf, class_id=0, area=a, centroid=c)


def _intact_packet_mask(h: int = 480, w: int = 640) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(m, (180, 100), (380, 380), 255, -1)
    return m


def _cut_packet_mask(h: int = 480, w: int = 640) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(m, (180, 90), (380, 160), 255, -1)
    cv2.rectangle(m, (180, 200), (380, 390), 255, -1)  # 40px gap
    return m


def _make_stage1_open_frame() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = 480, 640
    rng = np.random.default_rng(123)
    frame = np.clip(205 + rng.integers(-6, 7, size=(h, w, 3)), 0, 255).astype(np.uint8)
    pkt = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(pkt, (220, 120), (420, 360), 255, -1)
    eroded = cv2.erode(pkt, np.ones((35, 35), np.uint8))
    frame[pkt > 0] = (190, 190, 190)
    frame[eroded > 0] = np.clip(85 + rng.integers(-12, 13, size=(h, w, 3)), 0, 255).astype(np.uint8)[eroded > 0]

    gold = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(gold, (320, 240), 30, 255, -1)
    yy, xx = np.ogrid[:h, :w]
    pat = ((np.sin(xx * 0.45) * np.cos(yy * 0.45)) * 38.0).astype(np.int16)
    gold_bgr = np.stack(
        [np.clip(28 + pat // 3, 5, 80), np.clip(165 + pat, 90, 225), np.clip(215 + pat, 140, 248)],
        axis=-1,
    ).astype(np.uint8)
    frame[gold > 0] = gold_bgr[gold > 0]
    return frame, pkt, gold


def _make_testbed_frame() -> np.ndarray:
    frame = np.full((480, 640, 3), 230, dtype=np.uint8)
    cv2.rectangle(frame, (160, 120), (480, 400), (40, 150, 45), 4)
    return frame


def _disc_mask(cx: int, cy: int, r: int = 20) -> np.ndarray:
    m = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def _fast_configs(tmpdir: str) -> Tuple[WorkflowMonitorConfig, InspectionConfig]:
    wf = WorkflowMonitorConfig.load()
    wf.ocr_protection.min_reliable_reads = 2
    wf.ocr_protection.ocr_attempt_interval_frames = 1
    wf.ocr_protection.max_unresolved_backside_frames_before_alert = 2
    wf.automated_processing.baseline_stable_frames = 2
    wf.automated_processing.orientation_closed_confirm_frames = 2
    wf.automated_processing.hand_confirm_frames = 1
    wf.automated_processing.scissor_confirm_frames = 1
    wf.automated_processing.clear_wait_seconds = 0.2
    wf.automated_processing.cut_verify_seconds = 0.4
    wf.automated_processing.verify_sample_interval = 0.1
    wf.automated_processing.min_verify_samples = 3
    wf.automated_processing.min_cut_streak = 2
    wf.automated_processing.post_cut_unknown_alert_frames = 2
    wf.automated_processing.auto_complete_stable_frames = 3
    wf.automated_processing.auto_complete_min_seconds = 0.0
    wf.automated_processing.outside_testbed_alert_frames = 2
    wf.automated_processing.uncertain_count_alert_frames = 2
    wf.alerts.suppression_cooldown_seconds = 5.0
    wf.output.output_dir = tmpdir
    wf.output.save_evidence_images = True
    wf.output.save_json_report = True
    wf.output.save_markdown_report = True

    insp = InspectionConfig.load()
    insp.packet_inspection.temporal_confirm_frames = 2
    insp.packet_inspection.temporal_window_size = 4
    insp.jewellery_counting.testbed_roi = (160, 120, 480, 400)
    insp.jewellery_counting.empty_testbed_confirm_frames = 2
    insp.jewellery_counting.tracking.min_track_confirm_hits = 2
    insp.jewellery_counting.tracking.max_missing_frames = 2
    insp.jewellery_counting.stability.required_stable_frames = 3
    insp.jewellery_counting.stability.min_stable_seconds = 0.0
    return wf, insp


class TestAutomatedWorkflowMonitoring(unittest.TestCase):
    def test_1_fully_compliant_single_start_automated_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wf_cfg, insp_cfg = _fast_configs(tmpdir)
            engine = AutomatedWorkflowEngine(wf_cfg, insp_cfg, auto_load_models=False)
            session_id = engine.start_session(now_sec=0.0)
            self.assertTrue(session_id.startswith("SES-"))

            base_frame = _make_testbed_frame()
            intact_mask = _intact_packet_mask()
            cut_mask = _cut_packet_mask()
            empty_m = np.zeros((480, 640), dtype=np.uint8)
            t = 0.1

            # 1. Present front side -> automatic OCR locks 324894 on high-confidence 6-digit read
            step = engine.process_frame(
                base_frame,
                now_sec=t,
                overrides={
                    "packet_mask": intact_mask,
                    "hand_present": False,
                    "scissor_present": False,
                    "topstrip_visible": False,
                    "ocr_roi": (220, 120, 320, 160),
                    "ocr_text": ("324894", 0.96),
                },
            )
            t += 0.1
            self.assertEqual(step.packet_id, "324894")
            self.assertEqual(step.identity_status, "RESOLVED")
            self.assertEqual(step.stage, AutomatedStage.ORIENTATION_AND_BASELINE)

            # 2. Closed back/seal orientation + automatic intact baseline capture
            for _ in range(3):
                step = engine.process_frame(
                    base_frame,
                    now_sec=t,
                    overrides={
                        "packet_mask": intact_mask,
                        "hand_present": False,
                        "scissor_present": False,
                        "topstrip_visible": False,
                    },
                )
                t += 0.1
            self.assertEqual(step.stage, AutomatedStage.CUTTING_MONITOR)

            # 3. Cutting activity (scissors + hand) then clear scene
            step = engine.process_frame(
                base_frame,
                now_sec=t,
                overrides={"packet_mask": intact_mask, "hand_present": True, "scissor_present": True, "topstrip_visible": False},
            )
            t += 0.1
            for _ in range(4):
                step = engine.process_frame(
                    base_frame,
                    now_sec=t,
                    overrides={"packet_mask": cut_mask, "hand_present": False, "scissor_present": False, "topstrip_visible": False},
                )
                t += 0.1
            self.assertEqual(step.stage, AutomatedStage.CUT_VERIFICATION)

            # 4. Physical cut separation verification -> auto CUT_CONFIRMED -> Stage 1
            for _ in range(6):
                step = engine.process_frame(
                    base_frame,
                    now_sec=t,
                    overrides={
                        "packet_mask": cut_mask,
                        "hand_present": False,
                        "scissor_present": False,
                        "topstrip_visible": False,
                        "cut_eval": {"is_cut_detected": True, "confidence": 0.94, "class_name": "Cut"},
                    },
                )
                t += 0.15
                if step.stage == AutomatedStage.POST_CUT_PACKET_INSPECTION:
                    break
            self.assertEqual(step.stage, AutomatedStage.POST_CUT_PACKET_INSPECTION)

            # 5. Stage 1: Post-cut packet inspection -> OPEN -> auto Stage 2 empty testbed
            s1_frame, s1_pkt, s1_gold = _make_stage1_open_frame()
            for _ in range(3):
                step = engine.process_frame(
                    s1_frame,
                    now_sec=t,
                    overrides={
                        "stage1": {
                            "packet_mask": s1_pkt,
                            "hand_mask": empty_m,
                            "jewel_instances": [_raw_inst(s1_gold, 0.93)],
                        }
                    },
                )
                t += 0.1
                if step.stage == AutomatedStage.TESTBED_EMPTY_CHECK:
                    break
            self.assertEqual(step.stage, AutomatedStage.TESTBED_EMPTY_CHECK)

            # 6. Stage 2: Empty green testbed -> 3 separate jewellery items -> auto complete + report!
            for _ in range(2):
                step = engine.process_frame(
                    base_frame,
                    now_sec=t,
                    overrides={"stage2": {"jewel_instances": [], "hand_mask": empty_m, "packet_mask": empty_m}},
                )
                t += 0.1
            self.assertEqual(step.stage, AutomatedStage.JEWELLERY_COUNTING)

            jewels = [_raw_inst(_disc_mask(220, 220)), _raw_inst(_disc_mask(320, 260)), _raw_inst(_disc_mask(420, 300))]
            for _ in range(8):
                step = engine.process_frame(
                    base_frame,
                    now_sec=t,
                    overrides={"stage2": {"jewel_instances": jewels, "hand_mask": empty_m, "packet_mask": empty_m}},
                )
                t += 0.15
                if step.completed:
                    break

            self.assertTrue(step.completed)
            self.assertEqual(step.stage, AutomatedStage.SESSION_COMPLETED)
            self.assertIsNotNone(step.final_report)
            report = step.final_report
            assert report is not None

            # Verify all 5 required report sections + saved JSON & Markdown files
            self.assertEqual(report["session_information"]["packet_id"], "324894")
            self.assertEqual(report["operational_results"]["cutting_verification"]["state"], "CUT_CONFIRMED")
            self.assertEqual(report["operational_results"]["packet_open_closed_inspection"]["status"], "OPEN")
            self.assertEqual(report["operational_results"]["jewellery_detection_and_counting"]["count"], 3)
            self.assertEqual(report["appraiser_compliance"]["confirmed_deviations"], 0)
            self.assertEqual(report["appraiser_compliance"]["correctly_followed_procedures"], 9)
            self.assertTrue(report["final_summary"]["packet_processing_completed"])
            self.assertTrue(report["final_summary"]["prescribed_procedure_followed"])
            self.assertTrue(Path(step.report_paths["json"]).is_file())
            self.assertTrue(Path(step.report_paths["markdown"]).is_file())

    def test_2_backside_cut_before_ocr_alerts_continues_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wf_cfg, insp_cfg = _fast_configs(tmpdir)
            engine = AutomatedWorkflowEngine(wf_cfg, insp_cfg, auto_load_models=False)
            engine.start_session(now_sec=0.0)

            frame = _make_testbed_frame()
            intact = _intact_packet_mask()
            cut_m = _cut_packet_mask()
            t = 0.1

            # Present packet WITHOUT ID side (backside first) -> triggers OCR_INCOMPLETE & OCR_BEFORE_CUTTING_REQUIRED
            for _ in range(3):
                step = engine.process_frame(
                    frame,
                    now_sec=t,
                    overrides={"packet_mask": intact, "hand_present": False, "scissor_present": False, "topstrip_visible": True},
                )
                t += 0.1
            msgs = [a["message"] for a in step.active_alerts]
            self.assertIn("Please present the packet identification side before cutting.", msgs)

            # Appraiser starts cutting BEFORE OCR -> deviation logged, workflow MUST CONTINUE!
            step = engine.process_frame(
                frame,
                now_sec=t,
                overrides={"packet_mask": intact, "hand_present": True, "scissor_present": True, "topstrip_visible": False},
            )
            t += 0.1
            self.assertEqual(step.stage, AutomatedStage.CUTTING_MONITOR)
            self.assertEqual(step.packet_id, "UNRESOLVED")

            # Appraiser responds to alert and shows ID side -> OCR recovers "908123" automatically!
            step = engine.process_frame(
                frame,
                now_sec=t,
                overrides={
                    "packet_mask": cut_m,
                    "hand_present": False,
                    "scissor_present": False,
                    "topstrip_visible": False,
                    "ocr_roi": (220, 120, 320, 160),
                    "ocr_text": ("908123", 0.95),
                },
            )
            self.assertEqual(step.packet_id, "908123")
            self.assertEqual(step.identity_status, "RECOVERED")

            report = engine.stop_and_generate_report(now_sec=t + 1.0)
            self.assertEqual(report["session_information"]["packet_identity_status"], "RECOVERED")
            dev_codes = {d["code"]: d for d in report["deviation_analysis"]}
            self.assertIn("OCR_SKIPPED_BEFORE_CUTTING", dev_codes)
            self.assertTrue(dev_codes["OCR_SKIPPED_BEFORE_CUTTING"]["corrected"])
            self.assertTrue(dev_codes["OCR_SKIPPED_BEFORE_CUTTING"]["processing_continued"])

    def test_3_unresolved_ocr_never_fabricated_and_processing_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wf_cfg, insp_cfg = _fast_configs(tmpdir)
            engine = AutomatedWorkflowEngine(wf_cfg, insp_cfg, auto_load_models=False)
            engine.start_session(now_sec=0.0)

            frame = _make_testbed_frame()
            intact = _intact_packet_mask()
            cut_m = _cut_packet_mask()
            empty_m = np.zeros((480, 640), dtype=np.uint8)
            t = 0.1

            # Packet appears -> immediately cut without OCR ever succeeding
            engine.process_frame(frame, now_sec=t, overrides={"packet_mask": intact, "hand_present": False, "scissor_present": False})
            t += 0.1
            engine.process_frame(frame, now_sec=t, overrides={"packet_mask": intact, "hand_present": True, "scissor_present": True})
            t += 0.1
            for _ in range(4):
                engine.process_frame(frame, now_sec=t, overrides={"packet_mask": cut_m, "hand_present": False, "scissor_present": False})
                t += 0.1
            for _ in range(6):
                step = engine.process_frame(
                    frame,
                    now_sec=t,
                    overrides={"packet_mask": cut_m, "hand_present": False, "scissor_present": False, "cut_eval": {"is_cut_detected": True, "confidence": 0.92, "class_name": "Cut"}},
                )
                t += 0.15
                if step.stage == AutomatedStage.POST_CUT_PACKET_INSPECTION:
                    break
            self.assertEqual(step.stage, AutomatedStage.POST_CUT_PACKET_INSPECTION)
            self.assertEqual(step.packet_id, "UNRESOLVED")

            # Complete Stage 1 + Stage 2 while OCR remains UNRESOLVED
            s1_frame, s1_pkt, s1_gold = _make_stage1_open_frame()
            for _ in range(3):
                step = engine.process_frame(
                    s1_frame, now_sec=t,
                    overrides={"stage1": {"packet_mask": s1_pkt, "hand_mask": empty_m, "jewel_instances": [_raw_inst(s1_gold)]}},
                )
                t += 0.1
            for _ in range(2):
                engine.process_frame(frame, now_sec=t, overrides={"stage2": {"jewel_instances": [], "hand_mask": empty_m, "packet_mask": empty_m}})
                t += 0.1
            jewels = [_raw_inst(_disc_mask(240, 240)), _raw_inst(_disc_mask(380, 260))]
            for _ in range(8):
                step = engine.process_frame(
                    frame, now_sec=t,
                    overrides={"stage2": {"jewel_instances": jewels, "hand_mask": empty_m, "packet_mask": empty_m}},
                )
                t += 0.15
                if step.completed:
                    break

            self.assertTrue(step.completed)
            report = step.final_report
            assert report is not None
            # Must retain UNRESOLVED (never invent a fake ID!)
            self.assertEqual(report["session_information"]["packet_id"], "UNRESOLVED")
            self.assertEqual(report["session_information"]["packet_identity_status"], "UNRESOLVED")
            self.assertEqual(report["operational_results"]["jewellery_detection_and_counting"]["count"], 2)
            self.assertTrue(any("UNRESOLVED" in r for r in report["final_summary"]["issues_requiring_review"]))

    def test_4_alert_suppression_prevents_spam(self) -> None:
        mgr = RealTimeAlertManager("SES-TEST", suppression_cooldown_seconds=10.0)
        a1 = mgr.trigger("JEWELLERY_OUTSIDE_TESTBED", "PKT-1", "JEWELLERY_COUNTING", now_sec=1.0)
        a2 = mgr.trigger("JEWELLERY_OUTSIDE_TESTBED", "PKT-1", "JEWELLERY_COUNTING", now_sec=2.0)
        a3 = mgr.trigger("JEWELLERY_OUTSIDE_TESTBED", "PKT-1", "JEWELLERY_COUNTING", now_sec=3.0)
        self.assertIsNotNone(a1)
        self.assertIsNone(a2)
        self.assertIsNone(a3)
        self.assertEqual(len(mgr.all_alerts), 1)
        self.assertEqual(mgr.all_alerts[0].suppressed_occurrences, 2)

        resolved = mgr.resolve("JEWELLERY_OUTSIDE_TESTBED", "Moved inside testbed")
        self.assertIsNotNone(resolved)
        self.assertTrue(mgr.all_alerts[0].corrected)
        # Still suppressed inside cooldown right after resolution
        self.assertIsNone(mgr.trigger("JEWELLERY_OUTSIDE_TESTBED", "PKT-1", "JEWELLERY_COUNTING", now_sec=4.0))
        # Allowed again after cooldown expires
        a4 = mgr.trigger("JEWELLERY_OUTSIDE_TESTBED", "PKT-1", "JEWELLERY_COUNTING", now_sec=15.0)
        self.assertIsNotNone(a4)

    def test_5_postcut_unclear_outside_testbed_and_overlap_deviations_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wf_cfg, insp_cfg = _fast_configs(tmpdir)
            engine = AutomatedWorkflowEngine(wf_cfg, insp_cfg, auto_load_models=False)
            engine.start_session(now_sec=0.0)
            # Fast-forward directly to Stage 1 for focused post-cut compliance testing
            engine.stage = AutomatedStage.POST_CUT_PACKET_INSPECTION
            t = 1.0

            s1_frame, s1_pkt, s1_gold = _make_stage1_open_frame()
            empty_m = np.zeros((480, 640), dtype=np.uint8)
            hand_occ = np.zeros((480, 640), dtype=np.uint8)
            cv2.rectangle(hand_occ, (200, 130), (380, 340), 255, -1)

            # 1) Occluded packet view -> triggers "Please hold the packet clearly under the inspection camera."
            for _ in range(3):
                step = engine.process_frame(
                    s1_frame, now_sec=t,
                    overrides={"stage1": {"packet_mask": s1_pkt, "hand_mask": hand_occ, "jewel_instances": []}},
                )
                t += 0.1
            self.assertIn(
                "Please hold the packet clearly under the inspection camera.",
                [a["message"] for a in step.active_alerts],
            )

            # Hold clearly -> recovers & advances to TESTBED_EMPTY_CHECK
            for _ in range(3):
                step = engine.process_frame(
                    s1_frame, now_sec=t,
                    overrides={"stage1": {"packet_mask": s1_pkt, "hand_mask": empty_m, "jewel_instances": [_raw_inst(s1_gold)]}},
                )
                t += 0.1
            self.assertEqual(step.stage, AutomatedStage.TESTBED_EMPTY_CHECK)

            # Empty testbed
            tb_frame = _make_testbed_frame()
            for _ in range(2):
                engine.process_frame(tb_frame, now_sec=t, overrides={"stage2": {"jewel_instances": [], "hand_mask": empty_m, "packet_mask": empty_m}})
                t += 0.1

            # 2) Jewellery placed OUTSIDE testbed ROI (at x=60, y=60)
            outside_jewel = [_raw_inst(_disc_mask(60, 60, 22))]
            for _ in range(3):
                step = engine.process_frame(
                    tb_frame, now_sec=t,
                    overrides={"stage2": {"jewel_instances": outside_jewel, "hand_mask": empty_m, "packet_mask": empty_m}},
                )
                t += 0.1
            self.assertIn(
                "Jewellery placement was detected outside the designated testbed.",
                [a["message"] for a in step.active_alerts],
            )

            # 3) Move inside testbed, but OVERLAPPING bangles -> triggers COUNT_UNCERTAIN alert
            overlap = np.zeros((480, 640), dtype=np.uint8)
            cv2.circle(overlap, (295, 260), 36, 255, 8)
            cv2.circle(overlap, (330, 260), 36, 255, 8)
            for _ in range(3):
                step = engine.process_frame(
                    tb_frame, now_sec=t,
                    overrides={"stage2": {"jewel_instances": [_raw_inst(overlap)], "hand_mask": empty_m, "packet_mask": empty_m}},
                )
                t += 0.1
            self.assertIn(
                "Please ensure all jewellery articles are visible for accurate counting.",
                [a["message"] for a in step.active_alerts],
            )

            # 4) Separate into 2 clear articles -> recovers and auto-completes!
            clear_pair = [_raw_inst(_disc_mask(240, 250)), _raw_inst(_disc_mask(390, 250))]
            for _ in range(8):
                step = engine.process_frame(
                    tb_frame, now_sec=t,
                    overrides={"stage2": {"jewel_instances": clear_pair, "hand_mask": empty_m, "packet_mask": empty_m}},
                )
                t += 0.15
                if step.completed:
                    break

            self.assertTrue(step.completed)
            report = step.final_report
            assert report is not None
            dev_map = {d["code"]: d for d in report["deviation_analysis"]}
            for code in ("POST_CUT_VIEW_UNCLEAR", "JEWELLERY_OUTSIDE_TESTBED", "JEWELLERY_SEPARATION_UNCERTAIN"):
                self.assertIn(code, dev_map)
                self.assertTrue(dev_map[code]["corrected"], f"Expected {code} to be marked corrected")

    def test_6_stage1_skipped_directly_to_jewellery_adapts_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wf_cfg, insp_cfg = _fast_configs(tmpdir)
            engine = AutomatedWorkflowEngine(wf_cfg, insp_cfg, auto_load_models=False)
            engine.start_session(now_sec=0.0)
            engine.stage = AutomatedStage.POST_CUT_PACKET_INSPECTION

            tb_frame = _make_testbed_frame()
            empty_m = np.zeros((480, 640), dtype=np.uint8)
            jewel = [_raw_inst(_disc_mask(320, 260))]

            # Appraiser skips holding cut packet under camera and puts jewellery right on testbed
            step = engine.process_frame(
                tb_frame,
                now_sec=1.0,
                overrides={
                    "stage1": {"packet_mask": empty_m, "hand_mask": empty_m, "jewel_instances": []},
                    "stage2": {"jewel_instances": jewel, "hand_mask": empty_m, "packet_mask": empty_m},
                },
            )
            self.assertEqual(step.stage, AutomatedStage.JEWELLERY_COUNTING)

            t = 1.2
            for _ in range(8):
                step = engine.process_frame(
                    tb_frame, now_sec=t,
                    overrides={"stage2": {"jewel_instances": jewel, "hand_mask": empty_m, "packet_mask": empty_m}},
                )
                t += 0.15
                if step.completed:
                    break

            self.assertTrue(step.completed)
            report = step.final_report
            assert report is not None
            self.assertEqual(report["operational_results"]["jewellery_detection_and_counting"]["count"], 1)
            self.assertEqual(report["operational_results"]["packet_open_closed_inspection"]["status"], "UNKNOWN")
            self.assertTrue(any(d["code"] == "POST_CUT_INSPECTION_SKIPPED" for d in report["deviation_analysis"]))

    def test_7_real_models_and_pyqt_single_start_window_headless(self) -> None:
        sample_base = OUTPUTS_DIR / "baseline_frame.jpg"
        if not (MODELS_DIR / "packet-seg.onnx").is_file() or not sample_base.is_file():
            self.skipTest("Workspace models or baseline_frame.jpg not available")

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from workflow_monitor.ui_window import AutomatedMonitoringWindow

        with tempfile.TemporaryDirectory() as tmpdir:
            wf_cfg, insp_cfg = _fast_configs(tmpdir)
            mgr = UnifiedModelManager(wf_cfg)
            mgr.ensure_loaded(keys=("packet", "hand", "topstrip", "jewel"))
            # Verify TopStrip is strictly excluded from post-cut shared models!
            shared = mgr.post_cut_shared_models()
            self.assertIn("packet", shared)
            self.assertIn("jewel", shared)
            self.assertNotIn("topstrip", shared)

            img = cv2.imread(str(sample_base))
            assert img is not None
            counter = {"n": 0}

            def provider() -> Tuple[bool, Optional[np.ndarray]]:
                counter["n"] += 1
                if counter["n"] <= 3:
                    return True, img.copy()
                return False, None

            app = QApplication.instance() or QApplication([])
            win = AutomatedMonitoringWindow(
                camera_source=0,
                workflow_config=wf_cfg,
                inspection_config=insp_cfg,
                preloaded_models=mgr.models,
            )
            reports_seen: List[Dict] = []
            win.controller.report_ready.connect(lambda r, p: reports_seen.append(r))
            # Single START action!
            win.start_monitoring(frame_provider=provider)

            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and not reports_seen:
                app.processEvents()
                time.sleep(0.05)

            win.stop_monitoring()
            app.processEvents()
            win.close()

            self.assertEqual(len(reports_seen), 1)
            rep = reports_seen[0]
            self.assertIn("session_information", rep)
            self.assertIn("appraiser_compliance", rep)
            self.assertIn("final_summary", rep)
            self.assertGreater(len(rep["activity_log"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
