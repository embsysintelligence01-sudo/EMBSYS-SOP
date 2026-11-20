import importlib
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from standalone.home_capture import HomeCaptureTrackerMixin


def detection(x=40, y=40):
    return SimpleNamespace(box=(x, y, x+80, y+60), c=np.array([x+40, y+30]))


class OriginalTracker:
    def try_lock(self, *args):
        self.original_calls += 1


class Tracker(HomeCaptureTrackerMixin, OriginalTracker):
    def __init__(self):
        self.fast_home_capture = True
        self.arm_start = None
        self.arm_frames = deque()
        self.original_calls = 0
        self.event = SimpleNamespace(emit=Mock())
        self.capture = Mock(side_effect=lambda frame, detections: detections)
        self.lock_jewels = Mock()

    def roi_dist(self, point):
        return 1

    def on_stone(self, point, margin=0):
        return False


class HomeCaptureTests(unittest.TestCase):
    def test_box_jitter_and_count_change_do_not_restart_one_second_timer(self):
        tracker = Tracker()
        frame = np.zeros((200, 300, 3), np.uint8)
        for t, x in ((0., 40), (.3, 56), (.65, 35)):
            current = [detection(x)]
            if t == .3:
                current.append(detection(190))  # One-frame false detection.
            tracker.try_lock(current, t, frame)
        tracker.lock_jewels.assert_not_called()
        final = detection(53)
        tracker.try_lock([final], 1., frame)
        self.assertEqual(0., tracker.arm_start)
        self.assertEqual([final], tracker.capture.call_args.args[1])
        tracker.lock_jewels.assert_called_once()

    def test_dropout_keeps_timer_but_cannot_capture_missing_gold(self):
        tracker = Tracker()
        frame = np.zeros((200, 300, 3), np.uint8)
        tracker.try_lock([detection()], 0., frame)
        tracker.try_lock([], .5, frame)
        tracker.try_lock([], 1., frame)
        tracker.lock_jewels.assert_not_called()
        tracker.try_lock([detection(50)], 1.1, frame)
        tracker.try_lock([detection(35)], 1.4, frame)
        self.assertEqual(0., tracker.arm_start)
        tracker.lock_jewels.assert_called_once()

    def test_one_frame_false_mask_at_deadline_is_not_assigned(self):
        tracker = Tracker()
        frame = np.zeros((200, 300, 3), np.uint8)
        for t in (0., .3, .6):
            tracker.try_lock([detection()], t, frame)
        tracker.try_lock([detection(), detection(190)], 1., frame)
        self.assertEqual(1, len(tracker.capture.call_args.args[1]))

    def test_original_mode_retains_its_assignment(self):
        tracker = Tracker()
        tracker.fast_home_capture = False
        tracker.try_lock([], 0., None)
        self.assertEqual(1, tracker.original_calls)
        tracker.capture.assert_not_called()

    def test_slow_inference_can_capture_on_second_observation(self):
        tracker = Tracker()
        frame = np.zeros((200, 300, 3), np.uint8)
        tracker.try_lock([detection()], 0., frame)
        tracker.try_lock([detection(55)], 1.4, frame)
        tracker.lock_jewels.assert_called_once()

    def test_both_scripts_freeze_first_clear_current_home_once(self):
        frame = np.full((200, 300, 3), 215, np.uint8)
        roi = np.full(frame.shape[:2], 255, np.uint8)
        for script in ('pickup_trigger_test', 'pickup_trigger_motion_class'):
            with self.subTest(script=script):
                engine = importlib.import_module(script).HomePositionEngine(Mock())
                track = SimpleNamespace(id=7, det_box=(40, 40, 120, 100),
                    det_origin=(40, 40), det_mask=np.full((60, 80), 255, np.uint8), det_seen=1.)
                engine.adopt_tracks(frame, [track], [], roi, 1.)
                self.assertEqual([7], [j.id for j in engine.jewels])
                home = engine.jewels[0].home_cog.copy()
                track.det_box = (60, 40, 140, 100)
                track.det_seen = 1.3
                engine.adopt_tracks(frame, [track], [], roi, 1.3)
                np.testing.assert_array_equal(home, engine.jewels[0].home_cog)

    def test_home_capture_still_waits_for_hand_release_and_fresh_mask(self):
        from pickup_trigger_test import HomePositionEngine
        frame = np.full((200, 300, 3), 215, np.uint8)
        roi = np.full(frame.shape[:2], 255, np.uint8)
        track = SimpleNamespace(id=7, det_box=(40, 40, 120, 100),
            det_origin=(40, 40), det_mask=np.full((60, 80), 255, np.uint8), det_seen=0.)
        engine = HomePositionEngine(Mock())
        engine.adopt_tracks(frame, [track], [], roi, 1.)
        self.assertFalse(engine.jewels)
        track.det_seen = 1.
        hand = SimpleNamespace(box=track.det_box, mask=roi)
        engine.adopt_tracks(frame, [track], [], roi, 1., [hand])
        self.assertFalse(engine.jewels)
        engine.adopt_tracks(frame, [track], [], roi, 1.)
        self.assertEqual(1, len(engine.jewels))
