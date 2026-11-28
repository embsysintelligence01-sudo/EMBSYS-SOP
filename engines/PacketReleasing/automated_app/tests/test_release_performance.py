"""Regression checks for unnecessary inference, fresh frames, and crop-only OCR."""

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from packet_cut.desktop.capture import LatestFrameCapture
from packet_cut.desktop.worker import PacketWorker
from packet_cut.desktop.automation import AutoPipelineConfig
from workflow_monitor.config import WorkflowMonitorConfig
from workflow_monitor.models import UnifiedModelManager
from packet_cut.desktop.ocr import preprocess_crop_for_ocr, run_ocr, run_number_ocr
import test_release_pass_by as release_fixtures
from test_desktop_start_stop_only import _frontside_frame, _intact_mask


class TestReleaseInferenceBudget(unittest.TestCase):
    def setUp(self):
        self.fixture = release_fixtures.TestReleasePassBy()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.pipeline = self.fixture.pipeline

    def test_number_detector_and_ocr_are_idle_after_id_lock(self):
        p = self.pipeline
        p._packet_id = '324811'
        p._frontside_verified = True
        p.model_manager.models['number'] = Mock(predict=Mock(side_effect=AssertionError('Number detector should be idle')))
        p.model_manager.models['ocr'] = Mock()
        p._ocr_runner = Mock(side_effect=AssertionError('OCR should be idle'))
        for _ in range(8):
            self.fixture.step(packet_mask=_intact_mask(), topstrip_visible=False)
        p.model_manager.models['number'].predict.assert_not_called()
        p._ocr_runner.assert_not_called()
        self.assertEqual(p.recovery.ocr_attempts, 0)

    def test_ocr_receives_only_number_crop_then_stops_after_two_reads(self):
        p = self.pipeline
        self.fixture.frame = _frontside_frame(_intact_mask())
        p.model_manager.models['ocr'] = Mock()
        runner = Mock(return_value=('324811', .99, [('324811', .99)]))
        p._ocr_runner = runner
        for _ in range(10):
            self.fixture.step(packet_mask=_intact_mask(), ocr_roi=(250, 170, 390, 220))
        self.assertEqual(p._packet_id, '324811')
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(p._ocr_timing["ocr_calls"], 2)
        report = p.stop_session(now_sec=self.fixture.now)
        self.assertEqual(report["operational_results"]["ocr_identification"]["timing"]["ocr_calls"], 2)
        for call in runner.call_args_list:
            self.assertEqual(call.args[0].shape, (50, 140, 3))

    def test_hand_occlusion_skips_jewel_inference_without_erasing_count(self):
        for _ in range(7):
            self.fixture.step(jewel_instances=self.fixture.jewels())
        p = self.pipeline
        p.model_manager.models['jewel'] = Mock()
        p.jewel_detector._predict_instances = Mock()
        for index in range(6):
            p.process_frame(self.fixture.frame, now_sec=2 + index * .2, overrides={
                'packet_mask': self.fixture.empty, 'hand_present': True, 'scissor_present': False})
        p.jewel_detector._predict_instances.assert_not_called()
        self.assertEqual(p._observer.last_clear_count, 2)
        self.assertIsNone(p._observer.visible_count)

    def test_scheduled_frames_are_not_counted_as_new_detection_evidence(self):
        p = self.pipeline
        p.model_manager.models['jewel'] = Mock()
        predict = Mock(return_value=(self.fixture.jewels(), True))
        p.jewel_detector._predict_instances = predict
        ov = {'packet_mask': self.fixture.empty, 'hand_present': False, 'scissor_present': False}
        for timestamp in (0, .05, .10, .15, .20):
            p.process_frame(self.fixture.frame, now_sec=timestamp, overrides=ov)
        self.assertEqual(predict.call_count, 1)
        self.assertIsNone(p._observer.last_clear_count)
        for timestamp in (.3, .6, .9):
            p.process_frame(self.fixture.frame, now_sec=timestamp, overrides=ov)
        self.assertEqual(predict.call_count, 4)
        self.assertEqual(p._observer.last_clear_count, 2)

    def test_completed_cut_checks_stop_scissor_model(self):
        p = self.pipeline
        p._cut_result['state'] = 'CUT_CONFIRMED'
        p._observer.checks['cutting_activity']['status'] = 'VERIFIED'
        p.model_manager.models['scissor'] = Mock(predict=Mock(side_effect=AssertionError('Scissor detector should be idle')))
        for _ in range(5):
            p.process_frame(self.fixture.frame, overrides={
                'packet_mask': _intact_mask(), 'hand_present': True, 'jewel_instances': []})
        p.model_manager.models['scissor'].predict.assert_not_called()


class TestCropOcrBudget(unittest.TestCase):
    def test_one_pass_attempt_does_not_run_three_fallback_inferences(self):
        engine = Mock()
        engine.ocr.return_value = []
        with patch('packet_cut.desktop.ocr.get_ocr_engine', return_value=engine):
            result = run_ocr(np.full((20, 100, 3), 200, np.uint8), max_passes=1)
        self.assertIsNone(result[0])
        self.assertEqual(engine.ocr.call_count, 1)

    def test_direct_recognition_skips_text_detection_for_readable_number(self):
        recognizer = Mock()
        recognizer.predict.return_value = [{"rec_text": "324871", "rec_score": .99}]
        with patch('packet_cut.desktop.ocr.get_number_recognizer', return_value=recognizer), \
                patch('packet_cut.desktop.ocr.get_ocr_engine') as full_engine:
            result = run_number_ocr(np.full((20, 100, 3), 200, np.uint8))
        self.assertEqual(result[:2], ('324871', .99))
        full_engine.assert_not_called()
        recognizer.predict.assert_called_once()

    def test_failed_direct_read_only_uses_fallback_when_scheduled(self):
        recognizer = Mock()
        recognizer.predict.return_value = [{"rec_text": "32", "rec_score": .3}]
        with patch('packet_cut.desktop.ocr.get_number_recognizer', return_value=recognizer), \
                patch('packet_cut.desktop.ocr.run_ocr', return_value=('324871', .98, [])) as fallback:
            run_number_ocr(np.full((20, 100, 3), 200, np.uint8))
            fallback.assert_not_called()
            result = run_number_ocr(np.full((20, 100, 3), 200, np.uint8), allow_detection_fallback=True)
            self.assertEqual(result[0], '324871')
            self.assertEqual(fallback.call_args.kwargs, {'max_passes': 1})

    def test_empty_crop_does_not_initialize_recognizer(self):
        with patch('packet_cut.desktop.ocr.get_number_recognizer') as recognizer:
            self.assertIsNone(run_number_ocr(np.empty((0, 0, 3), np.uint8))[0])
            recognizer.assert_not_called()

    def test_number_crop_upscaling_is_bounded(self):
        crop = np.full((400, 2000, 3), 200, np.uint8)
        processed = preprocess_crop_for_ocr(crop)
        self.assertLessEqual(processed.shape[1], 960)
        self.assertLessEqual(processed.shape[0], 256)


class FakeCamera:
    def __init__(self):
        self.frames = 0
        self.released = False

    def read(self):
        time.sleep(.02)
        if self.released:
            return False, None
        self.frames += 1
        return True, np.full((24, 32, 3), self.frames % 255, np.uint8)

    def release(self):
        self.released = True


class TestLatestFrameCapture(unittest.TestCase):
    def test_preview_continues_during_slow_analysis_and_next_frame_is_recent(self):
        camera = FakeCamera()
        previews = []
        capture = LatestFrameCapture(camera, lambda frame: previews.append(int(frame[0, 0, 0])))
        capture.start()
        try:
            first = capture.next_frame(0, timeout=1)
            self.assertIsNotNone(first)
            time.sleep(.3)  # Simulated slow inference; camera must keep draining.
            recent = capture.next_frame(first[0], timeout=1)
            self.assertGreaterEqual(recent[0] - first[0], 5)
            self.assertLessEqual(camera.frames - recent[0], 1)
            self.assertGreaterEqual(len(previews), 3)
            duplicate = capture.next_frame(recent[0], timeout=.001)
            self.assertTrue(duplicate is None or duplicate[0] > recent[0])
        finally:
            capture.stop()
            capture.join(1)
        self.assertFalse(capture.is_alive())
        self.assertTrue(camera.released)

    def test_worker_preview_runs_while_analysis_is_busy(self):
        from PyQt6.QtCore import Qt
        camera = FakeCamera()
        manager = UnifiedModelManager(WorkflowMonitorConfig.load())
        manager.loaded = {'packet': 'test'}
        manager.ensure_loaded = Mock(return_value={'loaded': dict(manager.loaded), 'device': 'cpu'})
        worker = PacketWorker(model_manager=manager, pipeline_config=AutoPipelineConfig(save_outputs=False))
        worker._open_camera = Mock(return_value=camera)
        preview, analyses, status, detections = [], [], [], []
        worker.frame_ready.connect(lambda frame: preview.append(frame), Qt.ConnectionType.DirectConnection)
        worker.system_monitoring_updated.connect(status.append)
        worker.analysis_frame_ready.connect(lambda image, timestamp, data: detections.append((timestamp, data)), Qt.ConnectionType.DirectConnection)
        fake = Mock()
        fake._cut_model_filename = 'cut-seg.pt'
        fake._instruction = 'Observing release'
        fake.stage = SimpleNamespace(value='RELEASE')
        fake._final_report = {}  # Report serialization is tested separately.

        def analyse(frame, now_sec=None):
            analyses.append(now_sec)
            time.sleep(.2)
            if len(analyses) == 2:
                worker.stop()
            return SimpleNamespace(annotated_frame=frame.copy(), system_monitoring={}, live_monitoring={
                'hand_detected': True, 'scissor_detected': True, 'cutting_detection_fresh': True},
                                   compliance_monitoring={}, completed=False)
        fake.process_frame.side_effect = analyse
        with patch('packet_cut.desktop.worker.AutomatedPacketPipeline', return_value=fake):
            worker.run()
        self.assertEqual(len(analyses), 2)
        self.assertTrue(all(timestamp is not None for timestamp in analyses))
        self.assertGreaterEqual(len(preview), 4)
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0][0], analyses[0])
        self.assertTrue(detections[0][1]['scissor_detected'])
        self.assertTrue(any('LIVE' in item.get('camera_status', '') for item in status))
        self.assertTrue(camera.released)


if __name__ == '__main__':
    unittest.main()
