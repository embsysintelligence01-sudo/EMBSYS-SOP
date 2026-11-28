"""Release observations must remain independent of missed earlier checks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from packet_cut.desktop.automation import AutoPipelineConfig, AutomatedPacketPipeline
from packet_cut.desktop.state_machine import AutoStage
from packet_cut.desktop.worker import PacketWorker
from workflow_monitor.config import WorkflowMonitorConfig
from workflow_monitor.models import UnifiedModelManager
from test_desktop_start_stop_only import (_fast_cfg, _raw_inst, _testbed_frame, _intact_mask,
                                         _frontside_frame, _backside_frame, _detached_cut_mask)


class TestReleasePassBy(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        config, workflow, inspection = _fast_cfg(self.output.name)
        config.pass_by = True
        self.pipeline = AutomatedPacketPipeline(
            pipeline_config=config, workflow_config=workflow, inspection_config=inspection,
            model_manager=UnifiedModelManager(workflow),
        )
        self.pipeline.start_session(now_sec=0)
        self.pipeline.testbed_roi.set_roi((0, 0, 640, 480))
        self.frame = _testbed_frame()
        self.empty = np.zeros((480, 640), np.uint8)
        self.now = 0

    def step(self, **changes):
        self.now += .2
        overrides = {'packet_mask': self.empty, 'hand_present': False,
                     'scissor_present': False, 'jewel_instances': [],
                     'mouth_state_override': 'UNKNOWN'}
        overrides.update(changes)
        return self.pipeline.process_frame(self.frame, now_sec=self.now, overrides=overrides)

    def jewels(self, count=2):
        articles = []
        for index in range(count):
            mask = np.zeros_like(self.empty)
            cv2.circle(mask, (120 + 130 * index, 290), 18, 255, -1)
            articles.append(_raw_inst(mask))
        return articles

    def test_occupied_start_missing_ocr_and_strip_do_not_block_counting(self):
        self.step(packet_mask=_intact_mask(), topstrip_visible=True)
        for _ in range(7):
            snapshot = self.step(jewel_instances=self.jewels())
        self.assertEqual(self.pipeline.stage, AutoStage.RELEASE_OBSERVATION)
        self.assertIn('Retained clear: 2', snapshot.live_monitoring['jewellery_count'])
        report = self.pipeline.stop_session(now_sec=self.now)
        checks = {entry['key']: entry for entry in report['release_checks']}
        self.assertEqual(checks['initial_testbed']['status'], 'UNVERIFIED')
        self.assertEqual(checks['packet_number']['status'], 'UNVERIFIED')
        self.assertEqual(checks['seal']['result'], 'STRIP_VISIBLE')
        self.assertEqual(report['session_information']['packet_id'], 'UNRESOLVED')
        self.assertEqual(report['operational_results']['jewellery_detection_and_counting']['count'], 2)
        self.assertTrue(Path(self.pipeline._report_paths['pdf']).is_file())
        self.assertTrue(Path(self.pipeline._report_paths['html']).is_file())
        html = Path(self.pipeline._report_paths['html']).read_text()
        self.assertIn('Missing checks did not block', html)
        self.assertIn('Retained clear count', html)
        self.assertIn('Initial testbed not clear', html)
        import pypdfium2 as pdfium
        with pdfium.PdfDocument(self.pipeline._report_paths['pdf']) as document:
            pdf_text = '\n'.join(page.get_textpage().get_text_range() for page in document)
        self.assertIn('UNVERIFIED', pdf_text)
        self.assertIn('Retained clear count', pdf_text)

    def test_late_ocr_updates_identity_without_resetting_retained_count(self):
        for _ in range(7):
            self.step(jewel_instances=self.jewels())
        self.frame = _frontside_frame(_intact_mask())
        for _ in range(6):
            self.step(packet_mask=_intact_mask(), ocr_roi=(250, 170, 390, 220),
                      ocr_text=('324811', .98), topstrip_visible=False)
        self.assertEqual(self.pipeline._packet_id, '324811')
        self.assertTrue(self.pipeline._observer.checks['packet_number']['recovered_late'])
        self.assertEqual(self.pipeline._observer.last_clear_count, 2)
        self.assertFalse(self.pipeline._backside_verified, 'Missing strip alone is not proof of a backside or sealed packet')

    def test_clear_count_and_matching_media_survive_partial_and_full_removal(self):
        for _ in range(7):
            self.step(jewel_instances=self.jewels())
        first_image = self.pipeline._observer.last_clear_image
        for _ in range(6):
            self.step(jewel_instances=self.jewels(1))
        self.assertEqual(self.pipeline._observer.visible_count, 1)
        self.assertEqual(self.pipeline._observer.last_clear_count, 2)
        for _ in range(6):
            self.step()
        observer = self.pipeline._observer
        self.assertEqual(observer.visible_count, 0)
        self.assertEqual(observer.last_clear_count, 2)
        self.assertEqual(observer.last_clear_image, first_image)
        report = self.pipeline.stop_session(now_sec=self.now)
        result = report['operational_results']['jewellery_detection_and_counting']
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['current_visible_count'], 0)
        media = {entry['key']: entry for entry in report['evidence_media']}
        self.assertEqual(media['jewellery_count']['count'], 2)
        self.assertEqual(media['jewellery_count']['image_path'], result['evidence_image'])
        saved = json.loads(Path(self.pipeline._report_paths['json']).read_text())
        self.assertEqual(saved['flow_mode'], 'PASS_BY')

    def test_cut_evidence_can_be_verified_without_ocr_or_observed_scissors(self):
        for _ in range(12):
            self.step(packet_mask=_detached_cut_mask(), cut_eval={
                'is_cut_detected': True, 'class_name': 'cut', 'confidence': .98})
        self.assertEqual(self.pipeline._cut_result['state'], 'CUT_CONFIRMED')
        self.assertFalse(self.pipeline._cutting_activity_observed)
        self.assertEqual(self.pipeline._observer.checks['cutting_activity']['status'], 'UNVERIFIED')
        self.assertEqual(self.pipeline._observer.checks['cut_separation']['status'], 'VERIFIED')
        self.assertEqual(self.pipeline.stage, AutoStage.RELEASE_OBSERVATION)

    def test_no_count_evidence_is_unknown_not_a_verified_zero(self):
        for _ in range(7):
            self.step()
        result = self.pipeline.stop_session(now_sec=self.now)['operational_results']['jewellery_detection_and_counting']
        self.assertIsNone(result['count'])
        self.assertEqual(result['count_status'], 'NOT_VERIFIED')
        self.assertIsNone(result['evidence_image'])

    def test_hand_occlusion_keeps_evidence_and_reports_visible_count_unknown(self):
        for _ in range(7):
            self.step(jewel_instances=self.jewels())
        for _ in range(4):
            self.step(hand_present=True)
        result = self.pipeline.stop_session(now_sec=self.now)['operational_results']['jewellery_detection_and_counting']
        self.assertEqual(result['count'], 2)
        self.assertIsNone(result['current_visible_count'])
        self.assertEqual(result['count_status'], 'PROVISIONAL_WITH_UNCERTAINTY')

    def test_missing_models_do_not_turn_unavailable_observations_into_passes(self):
        for index in range(6):
            snapshot = self.pipeline.process_frame(self.frame, now_sec=index + 1)
        self.assertFalse(snapshot.completed)
        report = self.pipeline.stop_session(now_sec=7)
        self.assertFalse(report['operational_results']['empty_testbed_verified'])
        self.assertIsNone(report['operational_results']['jewellery_detection_and_counting']['count'])
        self.assertTrue(all(item['status'] == 'UNVERIFIED' for item in report['release_checks']))

    def test_missing_optional_models_do_not_block_readiness(self):
        config = WorkflowMonitorConfig.load()
        manager = UnifiedModelManager(config)
        manager.load_all_models = Mock(return_value={'loaded': {'packet': 'test'}, 'missing': {}, 'device': 'cpu'})
        worker = PacketWorker(prepare_only=True, model_manager=manager,
                              frame_provider=lambda: (True, self.frame.copy()),
                              pipeline_config=AutoPipelineConfig(pass_by=True))
        ready, errors = [], []
        worker.setup_ready.connect(lambda image, status: ready.append(status))
        worker.error.connect(errors.append)
        worker.run()
        self.assertFalse(errors)
        self.assertEqual(len(ready), 1)
        self.assertIn('Unavailable checks', ready[0])

    def test_scissor_inference_failure_does_not_disable_jewellery_count(self):
        self.pipeline.model_manager.models['scissor'] = Mock(predict=Mock(side_effect=RuntimeError('detector test')))
        packet = self.empty.copy()
        packet[80:180, 470:570] = 255
        for index in range(7):
            self.pipeline.process_frame(self.frame, now_sec=(index + 1) * .2, overrides={
                'packet_mask': packet, 'hand_present': False, 'jewel_instances': self.jewels()})
        self.assertGreater(self.pipeline.recovery.total_inference_errors, 0)
        self.assertEqual(self.pipeline._observer.last_clear_count, 2)
        self.assertEqual(self.pipeline._observer.checks['cutting_activity']['status'], 'UNVERIFIED')

    def test_brief_detection_does_not_become_a_retained_count(self):
        for _ in range(2):
            self.step(jewel_instances=self.jewels())
        self.assertIsNone(self.pipeline._observer.last_clear_count)

    def test_cutting_action_has_markings_image_and_clip_without_cut_confirmation(self):
        hand, scissors = self.empty.copy(), self.empty.copy()
        hand[180:320, 80:200] = 255
        scissors[120:160, 420:590] = 255
        for _ in range(3):
            snapshot = self.step(packet_mask=_intact_mask(), hand_present=True, hand_mask=hand,
                                 scissor_present=True, scissor_mask=scissors)
        self.assertTrue(snapshot.live_monitoring['hand_detected'])
        self.assertTrue(snapshot.live_monitoring['scissor_detected'])
        self.assertIn('CUTTING ACTIVITY DETECTED', snapshot.live_monitoring['cutting_verification'])
        self.assertFalse(np.array_equal(snapshot.annotated_frame[220, 120], self.frame[220, 120]))
        self.assertFalse(np.array_equal(snapshot.annotated_frame[140, 450], self.frame[140, 450]))
        report = self.pipeline.stop_session(now_sec=self.now)
        self.assertEqual(report['operational_results']['cutting_activity']['status'], 'CUTTING_OBSERVED')
        media = {entry['key']: entry for entry in report['evidence_media']}
        self.assertTrue(Path(media['cutting_activity']['image_path']).is_file())
        self.assertEqual(media['cutting_activity']['video_status'], 'SAVED')
        self.assertNotEqual(report['operational_results']['cutting_verification']['state'], 'CUT_CONFIRMED')
        self.assertIsNotNone(self.pipeline._observer.checks['cutting_activity']['evidence_image'])

    def test_partial_cut_keeps_candidate_evidence_without_promoting_result(self):
        for _ in range(2):
            self.step(packet_mask=_detached_cut_mask(), cut_eval={
                'is_cut_detected': True, 'class_name': 'cut', 'confidence': .899})
        report = self.pipeline.stop_session(now_sec=self.now)
        cut = report['operational_results']['cutting_verification']
        self.assertEqual(cut['state'], 'CUT_UNVERIFIED')
        self.assertEqual(cut['stable_frames'], 2)
        self.assertIn('full confirmation pending', cut['reason'])
        self.assertTrue(Path(cut['evidence_image']).is_file())
        media = {entry['key']: entry for entry in report['evidence_media']}
        self.assertEqual(media['cut_packet']['result'], 'CUT_UNVERIFIED')
        self.assertEqual(media['cut_packet']['video_status'], 'SAVED')
        self.assertEqual(self.pipeline._observer.checks['cut_separation']['status'], 'UNVERIFIED')

    def test_detection_inset_is_labelled_recent_analysis_without_replacing_live_frame(self):
        import time
        from PyQt6.QtGui import QColor, QImage
        from PyQt6.QtWidgets import QApplication
        from packet_cut.desktop.widgets import MainWindow
        app = QApplication.instance() or QApplication([])
        window = MainWindow(pipeline_config=AutoPipelineConfig(save_outputs=False))
        def close_window():
            window.close()
            app.processEvents()
        self.addCleanup(close_window)
        live = QImage(320, 240, QImage.Format.Format_RGB888)
        live.fill(QColor('#000000'))
        analysis = live.copy()
        analysis.fill(QColor('#22cc55'))
        window._on_frame_ready(live)
        window._on_detection_frame(analysis, time.monotonic(), {'hand_detected': True, 'scissor_detected': True})
        self.assertIsNotNone(window._detection_pixmap)
        self.assertEqual(window._last_pixmap.toImage().pixelColor(20, 60).name(), '#000000')
        self.assertGreater(window.video.pixmap().toImage().pixelColor(20, 60).green(), 100)
        window._detection_time = time.monotonic() - 2
        window._render_pixmap()
        self.assertEqual(window.video.pixmap().toImage().pixelColor(20, 60).name(), '#000000')

    def test_mouth_can_be_observed_without_cut_or_ocr_verification(self):
        for _ in range(12):
            self.step(packet_mask=_intact_mask(), mouth_state_override='CLOSED')
        self.assertEqual(self.pipeline._mouth_status, 'CLOSED')
        self.assertEqual(self.pipeline._observer.checks['packet_mouth']['status'], 'VERIFIED')
        self.assertNotEqual(self.pipeline._cut_result['state'], 'CUT_CONFIRMED')
        self.assertFalse(self.pipeline._frontside_verified)

    def test_backside_and_strip_can_be_verified_when_ocr_engine_is_unavailable(self):
        self.frame = _frontside_frame(_intact_mask())
        for _ in range(4):
            self.step(packet_mask=_intact_mask(), ocr_roi=(250, 170, 390, 220), topstrip_visible=False)
        self.assertTrue(self.pipeline._observer.front_view_observed)
        self.assertEqual(self.pipeline._packet_id, 'UNRESOLVED')
        self.frame = _backside_frame(_intact_mask())
        for _ in range(5):
            self.step(packet_mask=_intact_mask(), ocr_roi=None, topstrip_visible=False)
        self.assertTrue(self.pipeline._backside_verified)
        self.assertEqual(self.pipeline._observer.checks['seal']['status'], 'VERIFIED')
        self.assertEqual(self.pipeline._packet_id, 'UNRESOLVED')

    def test_occupied_roi_can_be_selected_without_empty_background_calibration(self):
        import time
        from PyQt6.QtCore import QPoint, Qt, QTimer
        from PyQt6.QtTest import QTest
        from PyQt6.QtWidgets import QApplication, QCheckBox, QDialogButtonBox
        from packet_cut.desktop.widgets import MainWindow, ROISelectionLabel
        config = WorkflowMonitorConfig.load()
        manager = UnifiedModelManager(config)
        manager.load_all_models = Mock(return_value={'loaded': {}, 'missing': {}, 'device': 'cpu'})
        app = QApplication.instance() or QApplication([])
        window = MainWindow(model_manager=manager, pipeline_config=AutoPipelineConfig(save_outputs=False),
                            frame_provider=lambda: (True, _frontside_frame(_intact_mask())))
        def close_window():
            window.close()
            app.processEvents()
        self.addCleanup(close_window)
        window.check_btn.click()
        deadline = time.monotonic() + 3
        while not window.check_btn.isEnabled() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        self.assertTrue(window.roi_btn.isEnabled())
        selected = []

        def choose():
            dialog = QApplication.activeModalWidget()
            try:
                checkbox = dialog.findChild(QCheckBox)
                selected.append(checkbox is not None and not checkbox.isChecked())
                preview = dialog.findChild(ROISelectionLabel)
                QTest.mousePress(preview, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
                QTest.mouseRelease(preview, Qt.MouseButton.LeftButton, pos=QPoint(400, 400))
                dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
            finally:
                if dialog.isVisible():
                    dialog.reject()
        QTimer.singleShot(0, choose)
        window.roi_btn.click()
        self.assertEqual(selected, [True])
        self.assertIsNone(window._empty_reference)
        self.assertTrue(window.start_btn.isEnabled())
        self.assertEqual(window.retry_btn.text(), 'Refresh Pending Checks')


if __name__ == '__main__':
    unittest.main()
