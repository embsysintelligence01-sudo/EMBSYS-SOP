"""Comprehensive unit & integration test suite for the Post-Cutting Inspection Module.

Covers all Section 10 requirements:
- Stage 1 Packet Inspection:
  1. Gold visible inside packet -> OPEN
  2. No gold visible -> CLOSED
  3. Packet partially occluded (hand or contour distortion) -> UNKNOWN (never CLOSED)
  4. Packet rotated -> UNKNOWN
  5. Poor lighting (dark & overexposed) -> UNKNOWN
  6. False gold reflections (specular glare & flat yellow cast) -> rejected (not OPEN)
- Stage 2 Jewellery Counting:
  7. Empty testbed check -> transitions cleanly to WAIT_FOR_JEWELLERY
  8. One jewellery article -> count 1
  9. Multiple separate articles (3 separate chains / 2 separate bangles) -> 3 / 2
 10. Two touching articles (touching pendants & touching figure-8 bangles) -> separated to 2
 11. Overlapping articles (overlapping bangles / crossing chains / overlapping masks) -> COUNT_UNCERTAIN
 12. Jewellery placed sequentially -> unique IDs without duplicate counting
 13. Jewellery removed and placed again -> true active count (no cumulative double counting)
 14. Uncertain segmentation -> COUNT_UNCERTAIN & blocks INSPECTION_COMPLETED
 15. Jewellery inside packet excluded from Stage 2 counting
- State machine, result schema, real model inference on workspace camera images, and PyQt6 UI/QThread test.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import List, Tuple

_APP_ROOT = Path(__file__).resolve().parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))
if str(_APP_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT.parent))

import cv2
import numpy as np

from inspection.inspection_result import PostCutInspectionResult
from inspection.inspection_state_machine import InspectionState, InspectionStateMachine
from inspection.jewellery_counting import InstanceSeparator, JewelleryCountManager
from inspection.packet_inspection import PacketStatusEvaluator
from inspection.post_cut_inspection import PostCutInspectionEngine
from inspection.utils.image_utils import RawSegInstance, mask_centroid_and_bbox
from inspection.utils.inspection_config import InspectionConfig
from inspection.utils.model_loader import InspectionModelRegistry


PROJECT_ROOT = _APP_ROOT
MODELS_DIR = PROJECT_ROOT / "models" if (PROJECT_ROOT / "models").exists() else PROJECT_ROOT.parent / "models"
OUTPUTS_DIR = PROJECT_ROOT / "packet_cut_outputs" if (PROJECT_ROOT / "packet_cut_outputs").exists() else PROJECT_ROOT.parent / "packet_cut_outputs"


def _make_raw_instance(mask: np.ndarray, conf: float = 0.90, class_id: int = 0) -> RawSegInstance:
    centroid, bbox, area = mask_centroid_and_bbox(mask)
    return RawSegInstance(
        mask=mask,
        bbox=bbox,
        confidence=conf,
        class_id=class_id,
        area=area,
        centroid=centroid,
    )


def _make_packet_frame(
    h: int = 480,
    w: int = 640,
    brightness: int = 205,
    interior_val: int = 85,
    add_gold_texture: bool = False,
    add_specular_glare: bool = False,
    add_flat_yellow_reflection: bool = False,
    rotate_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Build a realistic controlled top-down overhead packet frame + packet mask."""
    rng = np.random.default_rng(42)
    frame = np.full((h, w, 3), brightness, dtype=np.uint8)
    # Subtle table grain so Laplacian focus score is realistic
    noise = rng.integers(-6, 7, size=(h, w, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    pkt_mask = np.zeros((h, w), dtype=np.uint8)
    rect_pts = np.array([[220, 120], [420, 120], [420, 360], [220, 360]], dtype=np.float32)

    if abs(rotate_deg) > 1e-3:
        center = (320.0, 240.0)
        rot_m = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
        ones = np.ones((4, 1), dtype=np.float32)
        rect_pts = (rot_m @ np.hstack([rect_pts, ones]).T).T

    poly = np.round(rect_pts).astype(np.int32)
    cv2.fillPoly(pkt_mask, [poly], 255)

    # Outer packet plastic rim (light gray) + recessed dark interior cavity
    frame[pkt_mask > 0] = (min(255, brightness - 15), min(255, brightness - 15), min(255, brightness - 15))
    eroded = cv2.erode(pkt_mask, np.ones((35, 35), np.uint8))
    interior_noise = rng.integers(-12, 13, size=(h, w, 3), dtype=np.int16)
    interior_canvas = np.clip(
        np.full((h, w, 3), interior_val, dtype=np.int16) + interior_noise, 0, 255
    ).astype(np.uint8)
    frame[eroded > 0] = interior_canvas[eroded > 0]
    cv2.polylines(frame, [poly], True, (40, 40, 40), 3)

    gold_mask: Optional[np.ndarray] = None
    if add_gold_texture:
        gold_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(gold_mask, (320, 240), 30, 255, -1)
        # Rich metallic gold BGR with highlights and shadow texture (engravings/links)
        yy, xx = np.ogrid[:h, :w]
        pattern = ((np.sin(xx * 0.45) * np.cos(yy * 0.45)) * 38.0).astype(np.int16)
        b_ch = np.clip(28 + pattern // 3, 5, 80)
        g_ch = np.clip(165 + pattern, 90, 225)
        r_ch = np.clip(215 + pattern, 140, 248)
        gold_bgr = np.stack([b_ch, g_ch, r_ch], axis=-1).astype(np.uint8)
        frame[gold_mask > 0] = gold_bgr[gold_mask > 0]

    if add_specular_glare:
        # Bright washed-out white specular overhead lamp glare inside opening
        cv2.ellipse(frame, (320, 240), (35, 18), 0, 0, 360, (252, 253, 254), -1)

    if add_flat_yellow_reflection:
        # Smooth, uniform warm yellow lighting cast (no metallic texture/edges)
        cv2.circle(frame, (320, 240), 28, (30, 185, 220), -1)

    return frame, pkt_mask, gold_mask


def _make_empty_testbed_frame(h: int = 480, w: int = 640) -> np.ndarray:
    frame = np.full((h, w, 3), 230, dtype=np.uint8)
    # Green rectangular testbed border matching ROI (160, 120, 480, 400)
    cv2.rectangle(frame, (160, 120), (480, 400), (40, 150, 45), 4)
    return frame


def _draw_solid_disc_mask(h: int, w: int, cx: int, cy: int, r: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def _draw_bangle_mask(h: int, w: int, cx: int, cy: int, outer_r: int = 34, thickness: int = 9) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (cx, cy), outer_r, 255, thickness)
    return m


class TestStage1PacketInspection(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = InspectionConfig.load()
        self.cfg.packet_inspection.temporal_confirm_frames = 3
        self.cfg.packet_inspection.temporal_window_size = 5
        self.evaluator = PacketStatusEvaluator(self.cfg, model_registry=None)

    def test_1_gold_visible_inside_packet_returns_open(self) -> None:
        frame, pkt_mask, gold_mask = _make_packet_frame(add_gold_texture=True)
        assert gold_mask is not None
        jewel_inst = [_make_raw_instance(gold_mask, conf=0.91)]
        empty_hand = np.zeros(frame.shape[:2], dtype=np.uint8)

        eval_res = None
        for _ in range(3):
            eval_res = self.evaluator.evaluate_frame(
                frame,
                packet_mask_override=pkt_mask,
                hand_mask_override=empty_hand,
                jewel_instances_override=jewel_inst,
            )

        assert eval_res is not None
        self.assertTrue(eval_res.is_confirmed)
        self.assertEqual(eval_res.confirmed_status, "OPEN")
        self.assertTrue(eval_res.outcome.gold_visible)
        self.assertTrue(eval_res.outcome.inspection_valid)

    def test_2_no_gold_visible_returns_closed(self) -> None:
        frame, pkt_mask, _ = _make_packet_frame(add_gold_texture=False)
        empty_hand = np.zeros(frame.shape[:2], dtype=np.uint8)

        eval_res = None
        for _ in range(3):
            eval_res = self.evaluator.evaluate_frame(
                frame,
                packet_mask_override=pkt_mask,
                hand_mask_override=empty_hand,
                jewel_instances_override=[],
            )

        assert eval_res is not None
        self.assertTrue(eval_res.is_confirmed)
        self.assertEqual(eval_res.confirmed_status, "CLOSED")
        self.assertFalse(eval_res.outcome.gold_visible)
        self.assertTrue(eval_res.outcome.inspection_valid)

    def test_3_packet_partially_occluded_returns_unknown_not_closed(self) -> None:
        frame, pkt_mask, _ = _make_packet_frame(add_gold_texture=False)
        # Hand covering ~55% of packet opening
        hand_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.rectangle(hand_mask, (200, 140), (360, 340), 255, -1)

        for _ in range(4):
            eval_res = self.evaluator.evaluate_frame(
                frame,
                packet_mask_override=pkt_mask,
                hand_mask_override=hand_mask,
                jewel_instances_override=[],
            )

        self.assertFalse(eval_res.is_confirmed)
        self.assertEqual(eval_res.raw_status, "UNKNOWN")
        self.assertEqual(eval_res.confirmed_status, "UNKNOWN")
        self.assertNotEqual(eval_res.confirmed_status, "CLOSED")
        self.assertIn("occluded", eval_res.outcome.reason.lower())

    def test_4_packet_rotated_returns_unknown(self) -> None:
        # Elongated packet rotated by 45 deg exceeds max_tilt_deviation_deg
        self.cfg.packet_inspection.max_tilt_deviation_deg = 25.0
        evaluator = PacketStatusEvaluator(self.cfg, model_registry=None)

        h, w = 480, 640
        frame = np.full((h, w, 3), 205, dtype=np.uint8)
        rng = np.random.default_rng(7)
        frame = np.clip(frame.astype(np.int16) + rng.integers(-8, 9, size=(h, w, 3)), 0, 255).astype(np.uint8)
        # Narrower elongated packet rotated 44 deg
        pts = np.array([[250, 80], [390, 80], [390, 400], [250, 400]], dtype=np.float32)
        rot_m = cv2.getRotationMatrix2D((320.0, 240.0), 44.0, 1.0)
        rot_pts = np.round((rot_m @ np.hstack([pts, np.ones((4, 1), dtype=np.float32)]).T).T).astype(np.int32)
        pkt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(pkt_mask, [rot_pts], 255)
        frame[pkt_mask > 0] = (90, 90, 90)
        empty_hand = np.zeros((h, w), dtype=np.uint8)

        eval_res = evaluator.evaluate_frame(
            frame,
            packet_mask_override=pkt_mask,
            hand_mask_override=empty_hand,
            jewel_instances_override=[],
        )
        self.assertEqual(eval_res.raw_status, "UNKNOWN")
        self.assertEqual(eval_res.confirmed_status, "UNKNOWN")
        self.assertIn("rotated", eval_res.outcome.reason.lower())

    def test_5_poor_lighting_returns_unknown(self) -> None:
        dark_frame, pkt_mask, _ = _make_packet_frame(brightness=15, interior_val=10)
        empty_hand = np.zeros(dark_frame.shape[:2], dtype=np.uint8)

        eval_dark = self.evaluator.evaluate_frame(
            dark_frame,
            packet_mask_override=pkt_mask,
            hand_mask_override=empty_hand,
            jewel_instances_override=[],
        )
        self.assertEqual(eval_dark.raw_status, "UNKNOWN")
        self.assertIn("lighting", eval_dark.outcome.reason.lower())

        # Overexposed frame
        bright_frame = np.full((480, 640, 3), 252, dtype=np.uint8)
        eval_bright = self.evaluator.evaluate_frame(
            bright_frame,
            packet_mask_override=pkt_mask,
            hand_mask_override=empty_hand,
            jewel_instances_override=[],
        )
        self.assertEqual(eval_bright.raw_status, "UNKNOWN")
        self.assertIn("lighting", eval_bright.outcome.reason.lower())

    def test_6_false_gold_reflections_rejected(self) -> None:
        # Case A: Specular white glare that a noisy model falsely tags as gold
        frame_glare, pkt_mask, _ = _make_packet_frame(add_specular_glare=True)
        glare_blob = np.zeros(frame_glare.shape[:2], dtype=np.uint8)
        cv2.ellipse(glare_blob, (320, 240), (35, 18), 0, 0, 360, 255, -1)
        false_det = [_make_raw_instance(glare_blob, conf=0.80)]
        empty_hand = np.zeros(frame_glare.shape[:2], dtype=np.uint8)

        eval_glare = self.evaluator.evaluate_frame(
            frame_glare,
            packet_mask_override=pkt_mask,
            hand_mask_override=empty_hand,
            jewel_instances_override=false_det,
        )
        self.assertNotEqual(eval_glare.raw_status, "OPEN")
        self.assertTrue(eval_glare.gold_result is not None and eval_glare.gold_result.glare_rejected)

        # Case B: Smooth flat yellow lighting reflection inside packet (no metallic texture, no model det)
        self.evaluator.reset()
        frame_yellow, pkt_mask2, _ = _make_packet_frame(add_flat_yellow_reflection=True)
        for _ in range(3):
            eval_yellow = self.evaluator.evaluate_frame(
                frame_yellow,
                packet_mask_override=pkt_mask2,
                hand_mask_override=empty_hand,
                jewel_instances_override=[],
            )
        self.assertEqual(eval_yellow.confirmed_status, "CLOSED")
        self.assertFalse(eval_yellow.outcome.gold_visible)
        self.assertTrue(eval_yellow.gold_result is not None and eval_yellow.gold_result.glare_rejected)


class TestStage2JewelleryCounting(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = InspectionConfig.load()
        self.cfg.jewellery_counting.testbed_roi = (160, 120, 480, 400)
        self.cfg.jewellery_counting.empty_testbed_confirm_frames = 3
        self.cfg.jewellery_counting.tracking.min_track_confirm_hits = 2
        self.cfg.jewellery_counting.tracking.max_missing_frames = 2
        self.cfg.jewellery_counting.stability.required_stable_frames = 4
        self.cfg.jewellery_counting.stability.min_stable_seconds = 0.0
        self.manager = JewelleryCountManager(self.cfg, model_registry=None)
        self.frame = _make_empty_testbed_frame()
        self.empty_mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)

    def _confirm_empty(self) -> None:
        for i in range(3):
            ev = self.manager.evaluate_frame(
                self.frame,
                now_sec=float(i) * 0.1,
                jewel_instances_override=[],
                hand_mask_override=self.empty_mask,
                packet_mask_override=self.empty_mask,
            )
        self.assertTrue(self.manager.is_empty_verified)
        self.assertEqual(ev.phase, "WAIT_FOR_JEWELLERY")

    def test_1_empty_testbed(self) -> None:
        self._confirm_empty()
        ev = self.manager.evaluate_frame(
            self.frame,
            now_sec=1.0,
            jewel_instances_override=[],
            hand_mask_override=self.empty_mask,
            packet_mask_override=self.empty_mask,
        )
        self.assertEqual(ev.phase, "WAIT_FOR_JEWELLERY")
        self.assertEqual(ev.outcome.live_count, 0)

    def test_2_one_jewellery_article(self) -> None:
        self._confirm_empty()
        bangle = _draw_bangle_mask(480, 640, 320, 260, outer_r=34, thickness=9)
        insts = [_make_raw_instance(bangle, conf=0.92)]

        ev = None
        for i in range(6):
            ev = self.manager.evaluate_frame(
                self.frame,
                now_sec=1.0 + i * 0.2,
                jewel_instances_override=insts,
                hand_mask_override=self.empty_mask,
                packet_mask_override=self.empty_mask,
            )

        assert ev is not None
        self.assertEqual(ev.phase, "COUNT_CONFIRMATION")
        self.assertEqual(ev.outcome.count_status, "PENDING_CONFIRMATION")
        self.assertEqual(ev.outcome.count, 1)
        self.assertEqual(len(ev.outcome.articles), 1)

    def test_3_multiple_separate_articles(self) -> None:
        self._confirm_empty()
        # Three separate gold articles (e.g., 3 chains/pendants) inside testbed ROI
        m1 = _draw_solid_disc_mask(480, 640, 220, 210, 20)
        m2 = _draw_solid_disc_mask(480, 640, 320, 260, 20)
        m3 = _draw_solid_disc_mask(480, 640, 420, 310, 20)
        insts = [_make_raw_instance(m1), _make_raw_instance(m2), _make_raw_instance(m3)]

        ev = None
        for i in range(6):
            ev = self.manager.evaluate_frame(
                self.frame,
                now_sec=1.0 + i * 0.2,
                jewel_instances_override=insts,
                hand_mask_override=self.empty_mask,
                packet_mask_override=self.empty_mask,
            )

        assert ev is not None
        self.assertEqual(ev.phase, "COUNT_CONFIRMATION")
        self.assertEqual(ev.outcome.count, 3)
        ids = [a.article_id for a in ev.outcome.articles]
        self.assertEqual(len(set(ids)), 3)

    def test_4_two_touching_articles_separated_into_two(self) -> None:
        # Case A: Two touching solid pendants merged into 1 mask with narrow tangent neck
        merged_discs = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(merged_discs, (280, 260), 30, 255, -1)
        cv2.circle(merged_discs, (336, 260), 30, 255, -1)  # touching with narrow pinch

        sep = InstanceSeparator(self.cfg)
        res_discs = sep.separate(self.frame, [_make_raw_instance(merged_discs, conf=0.88)])
        self.assertFalse(res_discs.is_uncertain, f"Unexpected uncertainty: {res_discs.uncertainty_reasons}")
        self.assertEqual(len(res_discs.articles), 2)
        self.assertTrue(all(a.separated_from_cluster for a in res_discs.articles))

        # Case B: Two touching hollow bangles (figure-8) merged into 1 mask
        merged_bangles = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(merged_bangles, (275, 260), 34, 255, 8)
        cv2.circle(merged_bangles, (341, 260), 34, 255, 8)  # tangent figure-8 touch

        res_bangles = sep.separate(self.frame, [_make_raw_instance(merged_bangles, conf=0.88)])
        self.assertFalse(res_bangles.is_uncertain, f"Unexpected uncertainty: {res_bangles.uncertainty_reasons}")
        self.assertEqual(len(res_bangles.articles), 2)

    def test_5_overlapping_articles_trigger_count_uncertain(self) -> None:
        self._confirm_empty()
        # Case A: Two heavily overlapping bangles (creates 3 enclosed regions: left, lens, right)
        overlapping_bangles = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(overlapping_bangles, (295, 260), 36, 255, 8)
        cv2.circle(overlapping_bangles, (330, 260), 36, 255, 8)

        ev = None
        for i in range(4):
            ev = self.manager.evaluate_frame(
                self.frame,
                now_sec=1.0 + i * 0.2,
                jewel_instances_override=[_make_raw_instance(overlapping_bangles, conf=0.86)],
                hand_mask_override=self.empty_mask,
                packet_mask_override=self.empty_mask,
            )

        assert ev is not None
        self.assertEqual(ev.phase, "COUNT_UNCERTAIN")
        self.assertEqual(ev.outcome.count_status, "COUNT_UNCERTAIN")
        self.assertIsNone(ev.outcome.count)

        # Case B: Two overlapping raw detector masks
        sep = InstanceSeparator(self.cfg)
        m1 = _draw_solid_disc_mask(480, 640, 300, 260, 28)
        m2 = _draw_solid_disc_mask(480, 640, 316, 260, 28)
        res_overlap = sep.separate(self.frame, [_make_raw_instance(m1), _make_raw_instance(m2)])
        self.assertTrue(res_overlap.is_uncertain)

        # Case C: X-crossing chains (4 deep concavities)
        x_chains = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(x_chains, (240, 190), (380, 330), 255, 14)
        cv2.line(x_chains, (380, 190), (240, 330), 255, 14)
        res_x = sep.separate(self.frame, [_make_raw_instance(x_chains)])
        self.assertTrue(res_x.is_uncertain)

    def test_6_jewellery_placed_sequentially(self) -> None:
        self._confirm_empty()
        m1 = _draw_solid_disc_mask(480, 640, 230, 220, 20)
        m2 = _draw_solid_disc_mask(480, 640, 340, 220, 20)
        m3 = _draw_solid_disc_mask(480, 640, 290, 320, 20)

        t = 1.0
        # Place item 1
        for _ in range(3):
            ev1 = self.manager.evaluate_frame(
                self.frame, now_sec=t, jewel_instances_override=[_make_raw_instance(m1)],
                hand_mask_override=self.empty_mask, packet_mask_override=self.empty_mask,
            )
            t += 0.2
        first_id = ev1.outcome.articles[0].article_id
        self.assertEqual(ev1.outcome.live_count, 1)

        # Add item 2 (item 1 must retain its exact ID!)
        for _ in range(3):
            ev2 = self.manager.evaluate_frame(
                self.frame, now_sec=t,
                jewel_instances_override=[_make_raw_instance(m1), _make_raw_instance(m2)],
                hand_mask_override=self.empty_mask, packet_mask_override=self.empty_mask,
            )
            t += 0.2
        ids2 = [a.article_id for a in ev2.outcome.articles]
        self.assertIn(first_id, ids2)
        self.assertEqual(ev2.outcome.live_count, 2)

        # Add item 3 and let placement stabilize
        for _ in range(6):
            ev3 = self.manager.evaluate_frame(
                self.frame, now_sec=t,
                jewel_instances_override=[_make_raw_instance(m1), _make_raw_instance(m2), _make_raw_instance(m3)],
                hand_mask_override=self.empty_mask, packet_mask_override=self.empty_mask,
            )
            t += 0.2
        self.assertEqual(ev3.phase, "COUNT_CONFIRMATION")
        self.assertEqual(ev3.outcome.count, 3)
        self.assertIn(first_id, [a.article_id for a in ev3.outcome.articles])

    def test_7_jewellery_removed_and_placed_again_no_double_count(self) -> None:
        self._confirm_empty()
        m1 = _draw_solid_disc_mask(480, 640, 310, 260, 22)
        t = 1.0

        # Place 1 item
        for _ in range(3):
            ev = self.manager.evaluate_frame(
                self.frame, now_sec=t, jewel_instances_override=[_make_raw_instance(m1)],
                hand_mask_override=self.empty_mask, packet_mask_override=self.empty_mask,
            )
            t += 0.2
        self.assertEqual(ev.outcome.live_count, 1)

        # Remove item completely (beyond max_missing_frames)
        for _ in range(5):
            ev = self.manager.evaluate_frame(
                self.frame, now_sec=t, jewel_instances_override=[],
                hand_mask_override=self.empty_mask, packet_mask_override=self.empty_mask,
            )
            t += 0.2
        self.assertEqual(ev.outcome.live_count, 0)
        self.assertEqual(ev.phase, "WAIT_FOR_JEWELLERY")

        # Place the 1 item back on the testbed -> final count MUST be 1, NOT 2!
        for _ in range(6):
            ev = self.manager.evaluate_frame(
                self.frame, now_sec=t, jewel_instances_override=[_make_raw_instance(m1)],
                hand_mask_override=self.empty_mask, packet_mask_override=self.empty_mask,
            )
            t += 0.2
        self.assertEqual(ev.phase, "COUNT_CONFIRMATION")
        self.assertEqual(ev.outcome.count, 1)

    def test_8_jewellery_inside_packet_is_never_counted_in_stage2(self) -> None:
        self._confirm_empty()
        pkt_mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(pkt_mask, (200, 150), (400, 360), 255, -1)
        gold_inside_pkt = _draw_solid_disc_mask(480, 640, 300, 250, 24)

        ev = self.manager.evaluate_frame(
            self.frame,
            now_sec=1.0,
            jewel_instances_override=[_make_raw_instance(gold_inside_pkt)],
            hand_mask_override=self.empty_mask,
            packet_mask_override=pkt_mask,
        )
        self.assertEqual(ev.detection_result.excluded_inside_packet, 1)
        self.assertEqual(len(ev.detection_result.detections), 0)
        self.assertEqual(ev.outcome.live_count, 0)


class TestEndToEndEngineAndStateMachine(unittest.TestCase):
    def test_end_to_end_flow_and_uncertainty_completion_guard(self) -> None:
        cfg = InspectionConfig.load()
        cfg.packet_inspection.temporal_confirm_frames = 2
        cfg.jewellery_counting.empty_testbed_confirm_frames = 2
        cfg.jewellery_counting.tracking.min_track_confirm_hits = 2
        cfg.jewellery_counting.stability.required_stable_frames = 3
        cfg.jewellery_counting.stability.min_stable_seconds = 0.0
        cfg.output.save_evidence_images = False
        cfg.output.save_json_report = False

        engine = PostCutInspectionEngine(config=cfg, auto_load_models=False)
        engine.on_cut_confirmed("PKT-1025", cut_status="CUTTED")
        self.assertEqual(engine.state_machine.state, InspectionState.PACKET_VIEW_CHECK)

        # Stage 1: Confirm PACKET_OPEN
        frame_open, pkt_mask, gold_mask = _make_packet_frame(add_gold_texture=True)
        assert gold_mask is not None
        empty_hand = np.zeros(frame_open.shape[:2], dtype=np.uint8)
        for _ in range(2):
            step = engine.process_frame(
                frame_open,
                stage1_overrides={
                    "packet_mask": pkt_mask,
                    "hand_mask": empty_hand,
                    "jewel_instances": [_make_raw_instance(gold_mask, conf=0.93)],
                },
            )
        self.assertEqual(step.state, InspectionState.PACKET_OPEN)
        self.assertEqual(step.result.packet_inspection.status, "OPEN")
        self.assertTrue(step.result.packet_inspection.gold_visible)

        # Transition to Stage 2
        engine.proceed_to_jewellery_counting(require_empty_check=True)
        self.assertEqual(engine.state_machine.state, InspectionState.TESTBED_EMPTY_CHECK)

        # Empty testbed verification
        tb_frame = _make_empty_testbed_frame()
        for i in range(2):
            step = engine.process_frame(
                tb_frame,
                now_sec=float(i) * 0.2,
                stage2_overrides={
                    "jewel_instances": [],
                    "hand_mask": empty_hand,
                    "packet_mask": empty_hand,
                },
            )
        self.assertEqual(step.state, InspectionState.WAIT_FOR_JEWELLERY)

        # Test uncertainty guard: overlapping bangles -> COUNT_UNCERTAIN -> cannot mark COMPLETED!
        overlap = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(overlap, (295, 260), 36, 255, 8)
        cv2.circle(overlap, (330, 260), 36, 255, 8)
        step = engine.process_frame(
            tb_frame,
            now_sec=1.0,
            stage2_overrides={
                "jewel_instances": [_make_raw_instance(overlap, conf=0.87)],
                "hand_mask": empty_hand,
                "packet_mask": empty_hand,
            },
        )
        self.assertEqual(step.state, InspectionState.COUNT_UNCERTAIN)
        res_blocked = engine.confirm_final_count()
        self.assertNotEqual(res_blocked.inspection_status, "COMPLETED")
        self.assertEqual(res_blocked.jewellery_counting.count_status, "COUNT_UNCERTAIN")

        # Operator separates into 3 clear articles -> recount -> confirm -> COMPLETED!
        engine.retry_jewellery_counting(require_empty_check=False)
        m1 = _draw_solid_disc_mask(480, 640, 220, 220, 20)
        m2 = _draw_solid_disc_mask(480, 640, 320, 260, 20)
        m3 = _draw_solid_disc_mask(480, 640, 420, 300, 20)
        for i in range(5):
            step = engine.process_frame(
                tb_frame,
                now_sec=2.0 + i * 0.2,
                stage2_overrides={
                    "jewel_instances": [_make_raw_instance(m1), _make_raw_instance(m2), _make_raw_instance(m3)],
                    "hand_mask": empty_hand,
                    "packet_mask": empty_hand,
                },
            )
        self.assertEqual(step.state, InspectionState.COUNT_CONFIRMATION)
        final_res = engine.confirm_final_count()
        self.assertEqual(engine.state_machine.state, InspectionState.INSPECTION_COMPLETED)
        self.assertEqual(final_res.inspection_status, "COMPLETED")
        payload = final_res.to_dict()
        self.assertEqual(payload["packet_id"], "PKT-1025")
        self.assertEqual(payload["cut_status"], "CUTTED")
        self.assertEqual(payload["packet_inspection"]["status"], "OPEN")
        self.assertEqual(payload["jewellery_counting"]["count"], 3)
        self.assertEqual(payload["jewellery_counting"]["count_status"], "CONFIRMED")


class TestRealWorkspaceModelsAndImages(unittest.TestCase):
    """Runs real ONNX models on real camera frames captured in this workspace."""

    def test_real_models_loaded_without_topstrip_and_run_on_workspace_frames(self) -> None:
        pkt_model = MODELS_DIR / "packet-seg.onnx"
        jwl_model = MODELS_DIR / "jeweltrack.onnx"
        hnd_model = MODELS_DIR / "hand-seg.onnx"
        sample_gold = OUTPUTS_DIR / "verification_debug" / "sample_1794906516686_sample_1_orig.jpg"
        sample_baseline = OUTPUTS_DIR / "baseline_frame.jpg"

        if not (pkt_model.is_file() and jwl_model.is_file() and sample_gold.is_file() and sample_baseline.is_file()):
            self.skipTest("Workspace models or sample frames not available")

        cfg = InspectionConfig.load()
        registry = InspectionModelRegistry(cfg)
        report = registry.ensure_loaded()

        # Verify TopStrip is NEVER loaded or present in registry
        self.assertNotIn("topstrip", registry.models)
        self.assertNotIn("strip", registry.models)
        self.assertTrue(registry.has("packet"))
        self.assertTrue(registry.has("jewel"))
        self.assertTrue(registry.has("hand"))
        self.assertIn("sam", report.missing)  # Honestly reports optional SAM missing

        # 1. Test real jeweltrack.onnx on real gold cluster image (top center [272, 0, 360, 100])
        img_gold = cv2.imread(str(sample_gold))
        self.assertIsNotNone(img_gold)
        cfg.jewellery_counting.testbed_roi = (200, 0, 420, 140)
        cfg.jewellery_counting.empty_testbed_confirm_frames = 1
        mgr = JewelleryCountManager(cfg, model_registry=registry)
        mgr.skip_empty_check()
        ev = mgr.evaluate_frame(img_gold, now_sec=1.0)
        self.assertGreaterEqual(len(ev.detection_result.detections), 1)
        self.assertGreater(ev.detection_result.detections[0].confidence, 0.50)

        # 2. Test real packet-seg.onnx + jeweltrack.onnx on baseline_frame.jpg (intact flat packet, no gold)
        img_base = cv2.imread(str(sample_baseline))
        self.assertIsNotNone(img_base)
        s1 = PacketStatusEvaluator(cfg, model_registry=registry)
        s1_eval = s1.evaluate_frame(img_base)
        self.assertTrue(s1_eval.roi_result.detected)
        self.assertGreater(s1_eval.roi_result.confidence, 0.70)
        # No gold should be falsely detected on baseline_frame.jpg!
        self.assertNotEqual(s1_eval.raw_status, "OPEN")


class TestPyQtPanelHeadless(unittest.TestCase):
    """Verify PostCutInspectionPanel instantiates cleanly and responds to signals without blocking."""

    def test_panel_instantiation_and_signal_updates(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from inspection.ui_panel import PostCutInspectionPanel

        app = QApplication.instance() or QApplication([])
        panel = PostCutInspectionPanel()
        panel._on_packet_status_updated({
            "status": "OPEN",
            "gold_visible": True,
            "confidence": 0.91,
            "reason": "Gold jewellery detected inside packet interior.",
        })
        self.assertEqual(panel.pkt_status_val.text(), "OPEN")
        self.assertEqual(panel.gold_vis_val.text(), "YES")

        panel._on_jewellery_count_updated({
            "count_status": "COUNT_UNCERTAIN",
            "live_count": 0,
            "count": None,
            "stable_frames": 2,
            "required_stable_frames": 8,
            "reason": "Overlapping bangles detected",
            "articles": [],
        })
        self.assertFalse(panel.confirm_count_btn.isEnabled())
        self.assertEqual(panel.live_count_val.text(), "?")
        self.assertEqual(panel.final_count_val.text(), "COUNT_UNCERTAIN")

        panel._on_jewellery_count_updated({
            "count_status": "PENDING_CONFIRMATION",
            "live_count": 3,
            "count": 3,
            "stable_frames": 8,
            "required_stable_frames": 8,
            "reason": "Stable count 3",
            "articles": [
                {"article_id": "JWL-001", "confidence": 0.92, "area_px": 1400, "separated_from_cluster": False, "uncertain": False},
                {"article_id": "JWL-002", "confidence": 0.89, "area_px": 1320, "separated_from_cluster": True, "uncertain": False},
                {"article_id": "JWL-003", "confidence": 0.90, "area_px": 1500, "separated_from_cluster": False, "uncertain": False},
            ],
        })
        self.assertTrue(panel.confirm_count_btn.isEnabled())
        self.assertEqual(panel.live_count_val.text(), "3")
        self.assertEqual(panel.articles_list.count(), 3)
        panel.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main(verbosity=2)

