"""Real codec, bounded capture, failure recovery and portable-report checks."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from workflow_monitor.session_media import SessionMediaRecorder
from workflow_monitor.visual_report import render_visual_report


class TestSessionMedia(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recorder = SessionMediaRecorder(self.root / 'clips', 'SESSION', '2026-11-26T10:00:00+00:00', 0)
        self.frame = np.full((240, 320, 3), 150, np.uint8)

    def test_bounded_event_clip_decodes_and_preserves_real_duration(self):
        for index in range(101):
            timestamp = index / 5
            self.recorder.record_frame(self.frame, timestamp)
            if index == 50:
                self.recorder.event('cut_packet', 'cut.jpg', {'packet_id': '324811', 'result': 'CUT_CONFIRMED'})
        events = self.recorder.finish(20)
        event = events[0]
        self.assertEqual(event['video_status'], 'SAVED', self.recorder.errors)
        self.assertLessEqual(len(self.recorder.buffer), 26)
        self.assertLessEqual(event['duration_seconds'], 7.2)
        self.assertGreaterEqual(event['duration_seconds'], 7)
        self.assertEqual(event['packet_id'], '324811')
        cap = cv2.VideoCapture(event['video_path'])
        self.addCleanup(cap.release)
        self.assertTrue(cap.isOpened())
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), event['encoded_frames'])
        self.assertTrue(cap.read()[0])
        self.assertEqual(len(list((self.root / 'clips').iterdir())), 1)

    def test_stop_flushes_latest_count_without_full_post_event_wait(self):
        self.recorder.record_frame(self.frame, 0)
        self.recorder.record_frame(self.frame, 1)
        self.recorder.event('jewellery_count', 'one.jpg', {'count': 1, 'result': 'PROVISIONAL'})
        self.recorder.record_frame(self.frame, 1.2)
        self.recorder.event('jewellery_count', 'two.jpg', {'count': 2, 'result': 'PROVISIONAL'})
        self.recorder.record_frame(self.frame, 1.4)
        events = self.recorder.finish(1.4)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['count'], 2)
        self.assertEqual(events[0]['image_path'], 'two.jpg')
        self.assertEqual(events[0]['video_status'], 'SAVED')
        self.assertLessEqual(events[0]['duration_seconds'], 1.6)

    def test_codec_failure_is_reported_without_raising(self):
        self.recorder.record_frame(self.frame, 0)
        self.recorder.event('cut_packet', 'cut.jpg', {'result': 'CUT_CONFIRMED'})
        with patch('workflow_monitor.session_media.cv2.VideoWriter', return_value=Mock(isOpened=lambda: False)):
            events = self.recorder.finish(.1)
        self.assertEqual(events[0]['video_status'], 'UNAVAILABLE')
        self.assertIsNone(events[0]['video_path'])
        self.assertTrue(self.recorder.errors)

    def test_video_encoding_waits_until_save_instead_of_stalling_capture(self):
        with patch('workflow_monitor.session_media.cv2.VideoWriter') as writer:
            self.recorder.record_frame(self.frame, 0)
            self.recorder.event('cut_packet', 'cut.jpg', {'result': 'CUT_CONFIRMED'})
            for index in range(1, 31):
                self.recorder.record_frame(self.frame, index / 5)
            writer.assert_not_called()
        self.assertEqual(self.recorder.events['cut_packet']['video_status'], 'READY_TO_SAVE')
        events = self.recorder.finish(6)
        self.assertEqual(events[0]['video_status'], 'SAVED')

    def test_disabled_capture_and_retried_count_have_no_stale_media(self):
        self.recorder.enabled = False
        self.recorder.record_frame(self.frame, 0)
        self.recorder.event('cut_packet', 'cut.jpg', {})
        self.assertEqual(self.recorder.finish(0), [])
        self.assertFalse((self.root / 'clips').exists())
        self.recorder.enabled = True
        self.recorder.record_frame(self.frame, 1)
        self.recorder.event('jewellery_count', 'one.jpg', {'count': 1})
        self.recorder.clear_event('jewellery_count')
        self.assertEqual(self.recorder.finish(2), [])


class TestVisualReport(unittest.TestCase):
    def test_historical_cutting_action_image_is_not_hidden_by_unverified_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'cutting.jpg'
            image.write_bytes(b'example')
            report = {
                'operational_results': {'cutting_activity_observed': True, 'cutting_verification': {'state': 'CUT_UNVERIFIED'}},
                'evidence_media': [],
                'evidence_images': [{'tag': 'stage4_cutting_observed', 'path': str(image)}],
            }
            html = render_visual_report(report, root)
            self.assertIn('CUTTING_OBSERVED', html)
            self.assertIn('src="cutting.jpg"', html)
            self.assertIn('CUT_UNVERIFIED', html)
            self.assertNotIn('<video ', html)

    def test_relative_assets_escaped_metadata_and_explicit_provisional_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'evidence').mkdir()
            (root / 'clips').mkdir()
            image = root / 'evidence' / 'jewels.jpg'
            image.write_bytes(b'example')
            clip = root / 'clips' / 'jewels.webm'
            clip.write_bytes(b'example')
            report = {
                'session_information': {'session_id': 'SESSION', 'packet_id': '<script>123</script>'},
                'operational_results': {'jewellery_detection_and_counting': {'count': 2, 'count_status': 'PROVISIONAL'}},
                'evidence_media': [{'key': 'jewellery_count', 'image_path': str(image), 'video_path': str(clip),
                                    'result': 'PROVISIONAL', 'count': 2, 'duration_seconds': 4}],
            }
            html = render_visual_report(report, root)
            self.assertIn('src="evidence/jewels.jpg"', html)
            self.assertIn('src="clips/jewels.webm"', html)
            self.assertIn('<video controls', html)
            self.assertIn('PROVISIONAL', html)
            self.assertIn('Count: 2', html)
            self.assertNotIn('<script>', html)
            self.assertNotIn(str(root), html)
            self.assertIn('Image not captured for this step.', html)

    def test_existing_report_gets_image_without_inventing_historical_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'cut.jpg'
            image.write_bytes(b'example')
            report = {'evidence_images': [{'tag': 'stage5_cut_confirmed', 'path': str(image)}]}
            html = render_visual_report(report, root)
            self.assertIn('src="cut.jpg"', html)
            self.assertNotIn('<video ', html)
            self.assertIn('NOT_RECORDED', html)
            # A reset event in a new report must not fall back to an older snapshot.
            report['evidence_media'] = []
            self.assertNotIn('src="cut.jpg"', render_visual_report(report, root))


class TestPdfReport(unittest.TestCase):
    def test_pdf_embeds_images_and_preserves_provisional_count(self):
        from workflow_monitor.pdf_report import save_pdf_report
        import pypdfium2 as pdfium
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'count.jpg'
            cv2.imwrite(str(image), np.full((240, 320, 3), 150, np.uint8))
            report = {
                'session_information': {'session_id': 'SESSION', 'packet_id': '324811',
                                        'stage_reached': '07B COUNT JEWELLERY'},
                'operational_results': {'jewellery_detection_and_counting': {'count': 2, 'count_status': 'PROVISIONAL'}},
                'evidence_media': [{'key': 'jewellery_count', 'image_path': str(image),
                                    'result': 'PROVISIONAL', 'count': 2}],
            }
            path = Path(save_pdf_report(report, root / 'report.pdf'))
            self.assertTrue(path.read_bytes().startswith(b'%PDF'))
            self.assertIn(b'/Subtype /Image', path.read_bytes())
            with pdfium.PdfDocument(str(path)) as document:
                text = '\n'.join(page.get_textpage().get_text_range() for page in document)
            self.assertIn('Packet Release', text)
            self.assertIn('Engine 3', text)
            self.assertIn('324811', text)
            self.assertIn('Count: 2', text)
            self.assertIn('PROVISIONAL', text)
            self.assertIn('Image not captured', text)

    def test_pdf_export_failure_does_not_discard_other_reports(self):
        import json
        from packet_cut.desktop.automation import AutomatedPacketPipeline, AutoPipelineConfig
        from workflow_monitor.config import WorkflowMonitorConfig
        from workflow_monitor.models import UnifiedModelManager
        with tempfile.TemporaryDirectory() as directory:
            config = WorkflowMonitorConfig.load()
            pipeline = AutomatedPacketPipeline(
                workflow_config=config, model_manager=UnifiedModelManager(config),
                pipeline_config=AutoPipelineConfig(output_dir=directory),
            )
            pipeline.start_session(now_sec=0)
            with patch('workflow_monitor.pdf_report.save_pdf_report', side_effect=OSError('PDF export test')):
                report = pipeline.stop_session(now_sec=1)
            self.assertIsNone(pipeline._report_paths['pdf'])
            for key in ('json', 'markdown', 'html'):
                self.assertTrue(Path(pipeline._report_paths[key]).is_file())
            saved = json.loads(Path(pipeline._report_paths['json']).read_text())
            self.assertIn('PDF export test', saved['report_export_errors'][0])
            self.assertEqual(saved['session_information']['session_id'], report['session_information']['session_id'])


if __name__ == '__main__':
    unittest.main()
