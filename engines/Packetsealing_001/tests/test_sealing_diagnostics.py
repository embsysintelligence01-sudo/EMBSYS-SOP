import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from standalone.flow import PackingFlow
from standalone.models import PacketModels, Detection
from standalone.app import PacketWorker
from standalone.seal_context import SealContext


class SealingDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.flow = PackingFlow([1])
        self.step(10., 'present')

    def step(self, t, strip='absent', **changes):
        args = dict(packet_ok=True, packet_context_ok=True, strip_view_ok=True, hands_clear=True, strip=strip)
        args.update(changes)
        self.flow.update(t, {}, **args)

    def test_short_uncertainty_pauses_without_counting_time_or_hits(self):
        self.step(10.2)
        self.step(10.5)
        self.step(10.7, 'uncertain')
        self.assertEqual(2, self.flow.strip_absent_hits)
        self.assertTrue(self.flow.sealing_diagnostic['uncertainty_paused'])
        self.step(10.9)
        self.assertAlmostEqual(.3, self.flow.strip_clear_seconds)
        self.step(11.2)
        self.step(11.5)
        self.assertFalse(self.flow.sealed)  # Five observations, but only .9 s of clear intervals.
        self.step(11.8)
        self.assertTrue(self.flow.sealed)

    def test_prolonged_uncertainty_cannot_retain_progress(self):
        self.step(10.2)
        self.step(10.5)
        self.step(11.4, 'uncertain')
        self.assertEqual(0, self.flow.strip_absent_hits)
        self.assertEqual(0., self.flow.strip_clear_seconds)

    def test_each_blocker_is_visible_and_resets_progress(self):
        cases = [('packet_not_detected', dict(packet_context_ok=False)),
                 ('strip_area_not_visible', dict(strip_view_ok=False)),
                 ('hand_covering_strip_area', dict(hands_clear=False)),
                 ('packet_moving', dict(removal_ready=False)),
                 ('strip_still_detected', dict(strip='present'))]
        t = 11.
        for reason, changes in cases:
            self.step(t)
            self.step(t+.2, **changes)
            self.assertEqual(reason, self.flow.sealing_diagnostic['blocker'])
            self.assertEqual(0, self.flow.strip_absent_hits)
            self.assertFalse(self.flow.sealed)
            t += 1

    def test_attachment_occlusion_resets_progress_and_retains_warning(self):
        self.step(10.2)
        self.step(10.5, hands_clear=False)
        self.assertEqual(0, self.flow.strip_absent_hits)
        self.assertEqual('hand_covering_strip_area', self.flow.sealing_diagnostic['blocker'])
        self.assertTrue(self.flow.sealing_diagnostic['hand_warning'])
        for t in (10.8, 11.1, 11.4):
            self.step(t, hands_clear=False)
        self.assertFalse(self.flow.sealed)
        self.assertEqual(4, self.flow.sealing_hand_warning['observations'])
        for t in (11.7, 12., 12.3, 12.6, 12.9):
            self.step(t)
        self.assertTrue(self.flow.sealed)

    def test_observation_gap_is_recorded_and_resets_clear_time(self):
        self.step(10.2)
        self.step(10.6)
        self.step(14.)
        self.assertTrue(self.flow.sealing_diagnostic['observation_gap_reset'])
        self.assertEqual(1, self.flow.strip_absent_hits)
        self.assertEqual(0., self.flow.strip_clear_seconds)

    def test_motion_resets_progress_and_clear_settled_packet_can_retry(self):
        self.step(10.2)
        self.step(10.5, removal_ready=False)
        self.assertEqual(0, self.flow.strip_absent_hits)
        self.assertEqual('packet_moving', self.flow.sealing_diagnostic['blocker'])
        for t in (10.8, 11.1, 11.4, 11.7, 12.):
            self.step(t, removal_ready=True)
        self.assertTrue(self.flow.strip_removed)

    def test_landmark_adapter_uses_purity_thresholds_and_increasing_timestamps(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from standalone.models import LandmarkHands
        points = np.tile([50., 50.], (21, 1)).astype(np.float32)
        points[0], points[9] = (50, 60), (50, 40)
        hand = SimpleNamespace(points=points, scale=20., box=(45, 40, 55, 60))
        with patch('engines.sealing.hand_tracking.HandKeypoints') as factory:
            factory.return_value.detect.return_value = [hand]
            adapter = LandmarkHands()
            frame = np.zeros((100, 100, 3), np.uint8)
            first = adapter.detect(frame)
            adapter.detect(frame)
            self.assertTrue(factory.call_args.kwargs['video'])
            self.assertEqual(.25, factory.call_args.kwargs['detection_confidence'])
            calls = factory.return_value.detect.call_args_list
            self.assertLess(calls[0].kwargs['timestamp_ms'], calls[1].kwargs['timestamp_ms'])
            self.assertGreater(np.count_nonzero(first[0].mask), 0)
            adapter.close()
            factory.return_value.close.assert_called_once()

    def models(self):
        models = PacketModels.__new__(PacketModels)
        mask = np.full((100, 100), 255, np.uint8)
        zone = np.zeros_like(mask)
        zone[5:15, 5:95] = 255
        loose = np.zeros_like(mask)
        loose[70:80, 5:95] = 255
        models.bag = Mock(detect=Mock(return_value=[Detection(mask, (0, 0, 100, 100), .9)]))
        models.strip = Mock(detect=Mock(return_value=[Detection(loose, (5, 70, 95, 80), .9)]))
        models.hand = Mock(detect=Mock(return_value=[]))
        models.filter_bag = Mock(return_value=.9)
        models.filter_strip = Mock(return_value=.9)
        return models, mask, zone, loose

    def test_detached_strip_away_from_attachment_zone_does_not_block(self):
        models, mask, zone, loose = self.models()
        result = models.inspect(np.zeros((100, 100, 3), np.uint8), mask, inspection_zone=zone)
        self.assertEqual('absent', result[1])
        self.assertEqual('outside_attachment_area', models.last_diagnostic['strip_candidates'][0]['decision'])
        models.strip.detect.return_value = [Detection(zone, (5, 5, 95, 15), .9)]
        self.assertEqual('present', models.inspect(np.zeros((100, 100, 3), np.uint8), mask, inspection_zone=zone)[1])

    def test_hand_overlap_measurement_distinguishes_nearby_and_distant_hands(self):
        models, mask, zone, loose = self.models()
        frame = np.zeros((100, 100, 3), np.uint8)
        models.hand.detect.return_value = [Detection(loose, (5, 70, 95, 80), .9)]
        self.assertTrue(models.inspect(frame, mask, inspection_zone=zone)[2])
        models.hand.detect.return_value = [Detection(zone, (5, 5, 95, 15), .9)]
        self.assertFalse(models.inspect(frame, mask, inspection_zone=zone)[2])

    def test_log_keeps_source_time_and_only_one_scene_per_blocker(self):
        worker = PacketWorker(0, [])
        worker.flow = self.flow
        with tempfile.TemporaryDirectory() as tmp:
            worker.directory = Path(tmp)
            frame = np.zeros((100, 100, 3), np.uint8)
            for t in (12., 12.5):
                worker._log_sealing_diagnostic(frame, t, dict(timestamp=t, blocker='hand_covering_strip_area'))
            rows = [json.loads(row) for row in (Path(tmp)/'sealing_diagnostics.jsonl').read_text().splitlines()]
            self.assertEqual([12., 12.5], [r['source_timestamp'] for r in rows])
            self.assertEqual(1, len(list((Path(tmp)/'sealing_debug').glob('*.jpg'))))
            self.assertIn('scene', rows[0])
            self.assertNotIn('scene', rows[1])

    def test_slow_inference_resets_instead_of_adding_missing_clear_time(self):
        worker = PacketWorker(0, [])
        worker.flow = self.flow
        self.step(10.2)
        with tempfile.TemporaryDirectory() as tmp:
            worker.directory = Path(tmp)
            worker._log_sealing_diagnostic(None, 11., dict(timestamp=11., blocker='inference_too_slow'))
        self.assertEqual(0, self.flow.strip_absent_hits)
        self.assertEqual('inference_too_slow', self.flow.sealing_diagnostic['blocker'])

    def test_old_reference_loss_is_irrelevant_to_current_packet_check(self):
        self.step(10.2)
        self.step(10.4, reference_lost=True)
        self.assertEqual('checking_removal', self.flow.sealing_diagnostic['blocker'])
        self.assertEqual(2, self.flow.strip_absent_hits)

    def test_attachment_zone_maps_back_to_large_camera_frames(self):
        models, _, _, _ = self.models()
        models.bag.detect.side_effect = lambda frame: [Detection(np.full(frame.shape[:2], 255, np.uint8),
            (0, 0, frame.shape[1], frame.shape[0]), .9)]
        models.strip.detect.return_value = []
        frame = np.zeros((720, 1280, 3), np.uint8)
        roi = np.full(frame.shape[:2], 255, np.uint8)
        zone = np.zeros_like(roi)
        zone[50:100, 200:1100] = 255
        bag, strip, clear, strips, hands = models.inspect(frame, roi, inspection_zone=zone)
        self.assertEqual((720, 1280), bag.mask.shape)
        self.assertEqual((720, 1280), strips.shape)
        self.assertEqual((720, 1280), hands.shape)
        self.assertEqual('absent', strip)
        self.assertTrue(clear)
        self.assertEqual('tracked_attachment_area', models.last_diagnostic['inspection_scope'])
