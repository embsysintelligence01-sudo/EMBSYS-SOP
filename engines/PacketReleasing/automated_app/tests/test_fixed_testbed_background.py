"""Regression checks for permanent markings versus newly placed objects."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from packet_cut.desktop.automation import AutoPipelineConfig, AutomatedPacketPipeline
from packet_cut.desktop.state_machine import AutoStage
from workflow_monitor.config import WorkflowMonitorConfig
from workflow_monitor.models import UnifiedModelManager


class TestFixedTestbedBackground(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        config = AutoPipelineConfig(pass_by=False, save_outputs=False, output_dir=self.output.name)
        workflow = WorkflowMonitorConfig.load()
        self.pipe = AutomatedPacketPipeline(
            pipeline_config=config, workflow_config=workflow,
            model_manager=UnifiedModelManager(workflow),
        )
        self.pipe.start_session(now_sec=0)
        self.pipe.testbed_roi.set_roi((0, 0, 320, 240))
        self.packet_mask = np.zeros((240, 320), np.uint8)
        self.packet_mask[65:185, 85:235] = 255

    @staticmethod
    def marked_surface(colour):
        frame = np.full((240, 320, 3), 180, np.uint8)
        cv2.rectangle(frame, (85, 65), (235, 185), colour, 4)
        cv2.putText(frame, 'BED', (10, 225), cv2.FONT_HERSHEY_SIMPLEX, .6, colour, 2)
        # A permanent dark marker as well as the coloured rectangle.
        frame[10:42, 12:44] = 10
        frame[17:35, 19:37] = 240
        frame[23:30, 25:31] = 10
        return frame

    def test_any_colour_markings_pass_even_when_model_calls_rectangle_packet(self):
        for colour in ((0, 0, 170), (0, 150, 0), (180, 0, 0), (0, 0, 0), (20, 160, 210)):
            with self.subTest(colour=colour):
                frame = self.marked_surface(colour)
                self.pipe.start_session(now_sec=0)
                self.pipe.set_empty_reference(frame)
                for index in range(5):
                    self.pipe.process_frame(frame, now_sec=index + 1,
                                            overrides={'packet_mask': self.packet_mask})
                self.assertEqual(self.pipe.stage, AutoStage.WAIT_FRONTSIDE_AND_OCR)
                info = self.pipe._run_packet_seg(frame, {'packet_mask': self.packet_mask}, [])
                self.assertEqual(info['total_area'], 0, 'Marking must stay ignored in later stages')

    def test_real_packet_added_to_calibrated_bed_still_blocks_start(self):
        reference = self.marked_surface((0, 0, 170))
        self.pipe.set_empty_reference(reference)
        occupied = reference.copy()
        occupied[self.packet_mask > 0] = (40, 70, 100)
        for index in range(6):
            self.pipe.process_frame(occupied, now_sec=index + 1,
                                    overrides={'packet_mask': self.packet_mask})
        self.assertEqual(self.pipe.stage, AutoStage.WAIT_EMPTY_TESTBED)
        info = self.pipe._run_packet_seg(occupied, {'packet_mask': self.packet_mask}, [])
        self.assertGreater(info['total_area'], 1200)

    def test_lighting_shift_and_sensor_noise_do_not_make_markings_foreign(self):
        reference = self.marked_surface((180, 0, 0))
        self.pipe.set_empty_reference(reference)
        noise = np.random.default_rng(7).integers(-2, 3, reference.shape)
        frame = np.clip(reference.astype(np.int16) + 12 + noise, 0, 255).astype(np.uint8)
        for index in range(5):
            self.pipe.process_frame(frame, now_sec=index + 1,
                                    overrides={'packet_mask': self.packet_mask})
        self.assertEqual(self.pipe.stage, AutoStage.WAIT_FRONTSIDE_AND_OCR)

    def test_new_foreign_object_blocks_even_without_packet_prediction(self):
        reference = self.marked_surface((0, 0, 0))
        self.pipe.set_empty_reference(reference)
        frame = reference.copy()
        frame[60:190, 50:280] = 20  # Covers most of the region; must not be lighting.
        for index in range(5):
            self.pipe.process_frame(frame, now_sec=index + 1,
                                    overrides={'packet_mask': np.zeros_like(self.packet_mask)})
        self.assertEqual(self.pipe.stage, AutoStage.WAIT_EMPTY_TESTBED)

    def test_uncalibrated_packet_is_not_automatically_learned_as_background(self):
        frame = self.marked_surface((0, 0, 0))
        for index in range(5):
            self.pipe.process_frame(frame, now_sec=index + 1,
                                    overrides={'packet_mask': self.packet_mask})
        self.assertEqual(self.pipe.stage, AutoStage.WAIT_EMPTY_TESTBED)


if __name__ == '__main__':
    unittest.main()
