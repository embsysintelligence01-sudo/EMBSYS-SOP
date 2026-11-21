"""Regression coverage for the main purity recorder's fixed-box start."""
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from engines.purity.box_trigger import BoxRecordingTrigger, StoneRecordingArea, pixel_box, draw_hand_preview
from test_sop_revision import contact, monitor_case, snap


SETTINGS = dict(points=((.50, .30), (.90, .80)), pinch_px=40, gold_fallback=False)


class StandaloneParityTests(TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        from pathlib import Path
        import sys
        path = Path(__file__).resolve().parents[1]/'pickup_trigger_box_standalone/pickup_trigger_test_box.py'
        spec = importlib.util.spec_from_file_location('embsys_standalone_reference', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        previous_path = sys.path[:]
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = previous_path
        cls.reference = module

    def test_identical_frame_start_times_for_covered_home_and_landmark_misses(self):
        import tempfile
        from unittest.mock import patch
        for scenario in ('covered', 'brief_miss', 'no_hand', 'still_home'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp, \
                    patch.object(self.reference, 'HAND_PINCH_PX', 40), \
                    patch.object(self.reference, 'PINCH_HAND', 'Any'), \
                    patch.object(self.reference, 'REC_GOLD_FALLBACK', False):
                frame = np.full((100, 160, 3), 235, np.uint8)
                points = np.full((21, 2), (110., 52.), np.float32)
                points[4], points[8] = (92.5, 52), (127.5, 52)
                hand = SimpleNamespace(points=points, handedness='Right', scale=100.)
                model = Mock()
                model.detect.return_value = [hand]
                home = BoxTriggerTests.detection(15)
                track = SimpleNamespace(id=1, state='OK', pos=np.array(home.c, float),
                                        vel=np.zeros(2), det_seen=0., det_box=home.box, box=home.box,
                                        det_mask=home.mask, det_origin=home.origin)
                tracker = SimpleNamespace(tracks=[track], roi_dist=lambda p: 100.,
                                          id_model=None, identify=None)
                events = []
                box = pixel_box(SETTINGS['points'], frame.shape)
                reference = self.reference.BoxRecorder(tmp, 10, events.append, hands=model, box=box)
                integrated = BoxRecordingTrigger(lambda: model)
                cues = []
                try:
                    times = (0., .1, .2, .3, .4, 1., 1.2, 1.4, 1.6, 1.8, 2.)
                    for ts in times:
                        away = ts >= 1 and scenario != 'still_home'
                        track.state = 'OCCLUDED' if away else 'OK'
                        if not away:
                            track.det_seen = ts
                        frame[20:30, 15:25] = (40, 70, 120) if away else (235, 235, 235)
                        missing = scenario == 'no_hand' or (scenario == 'brief_miss' and ts in (1.2, 1.6))
                        model.detect.return_value = [] if missing else [hand]
                        dets = [] if away else [home]
                        reference.update(tracker, dets, frame, ts)
                        sample = dict(id=1, state=track.state, pos=track.pos, in_hand=False,
                                      det_seen=track.det_seen, mask=track.det_mask, origin=track.det_origin)
                        cue = integrated.observe((0, SETTINGS['points']), frame, ts, [sample], dets, SETTINGS)
                        if cue is not None:
                            cues.append(cue)
                    starts = [e for e in events if e['state'] == 'REC_STARTED']
                    self.assertEqual(len(starts), len(cues))
                    if scenario in ('covered', 'brief_miss'):
                        self.assertEqual(1, len(starts))
                        self.assertAlmostEqual(starts[0]['time'], cues[0]['confirmed_at'])
                        self.assertTrue(starts[0]['provisional'])
                        self.assertTrue(cues[0]['provisional'])
                        self.assertIsNone(cues[0]['jewel_id'])
                    else:
                        self.assertEqual([], cues)
                finally:
                    reference.stop_all('test finished')
                    integrated.close()


class StoneAreaTests(TestCase):
    def test_twenty_percent_margin_clips_at_frame_edges(self):
        area = StoneRecordingArea()
        shape = (200, 400, 3)
        self.assertEqual((), area.update((10, 20, 310, 180), 1, shape, 20, False))
        points = area.update((10, 20, 310, 180), 1.5, shape, 20, False)
        self.assertEqual((0, 0, 370, 199), pixel_box(points, shape))

    def test_occlusion_and_active_cycle_keep_box_steady_then_idle_can_move_it(self):
        area = StoneRecordingArea()
        shape = (500, 800, 3)
        stone = (200, 100, 500, 300)
        area.update(stone, 1, shape, 20, False)
        points = area.update(stone, 1.5, shape, 20, False)
        self.assertEqual((140, 60, 560, 340), pixel_box(points, shape))
        self.assertEqual(points, area.update(None, 1.7, shape, 20, True))
        moved = (250, 150, 550, 350)
        self.assertEqual(points, area.update(moved, 2, shape, 40, True))
        self.assertEqual(points, area.update(moved, 2.2, shape, 20, False))
        updated = area.update(moved, 2.8, shape, 20, False)
        self.assertEqual((190, 110, 610, 390), pixel_box(updated, shape))


class BoxTriggerTests(TestCase):
    def setUp(self):
        self.model = Mock()
        self.model.detect.return_value = []
        self.trigger = BoxRecordingTrigger(lambda: self.model)
        self.frame = np.full((100, 160, 3), 235, np.uint8)
        self.home = self.detection(15)
        self.track = dict(id=1, state='OK', pos=self.home.c, in_hand=False,
                          det_seen=0, mask=self.home.mask, origin=self.home.origin)
        self.observe(0, [self.home])
        self.model.detect.reset_mock()

    @staticmethod
    def detection(x):
        return SimpleNamespace(mask=np.ones((10, 10), np.uint8), origin=(x, 20),
                               box=(x, 20, x+10, 30), c=(x+5, 25))

    def observe(self, ts, dets, settings=SETTINGS):
        key = (0, settings['points'], settings.get('hand', 'Any'))
        return self.trigger.observe(key, self.frame, ts, [self.track], dets, settings)

    def test_covered_home_can_start_provisional_clip_without_assigning_identity(self):
        self.frame[20:30, 15:25] = (40, 70, 120)
        self.track.update(state='OCCLUDED')
        self.pinch(35)
        for ts in (.2, .4, .6):
            self.assertIsNone(self.observe(ts, []))
        cue = self.observe(.8, [])
        self.assertIsNone(cue['jewel_id'])
        self.assertTrue(cue['provisional'])
        self.assertEqual([1], cue['candidate_ids'])
        self.assertIn('covered', self.trigger.preview['detail'])

    def test_short_landmark_miss_preserves_hold_but_expired_grip_cannot_trigger(self):
        self.frame[20:30, 15:25] = (40, 70, 120)
        self.track.update(state='OCCLUDED')
        self.pinch(35)
        hand = self.model.detect.return_value
        self.assertIsNone(self.observe(.2, []))
        self.model.detect.return_value = []
        self.assertIsNone(self.observe(.4, []))
        self.model.detect.return_value = hand
        self.assertIsNone(self.observe(.6, []))
        self.model.detect.return_value = []
        self.assertTrue(self.observe(.8, [])['provisional'])
        self.assertIsNone(self.observe(1.2, []))
        self.assertIsNone(self.trigger.hold)
        self.assertIsNone(self.observe(1.4, []))

    def test_ignored_left_hand_does_not_block_explicit_right_hand_gold_fallback(self):
        self.pinch(35)
        self.model.detect.return_value[0].handedness = 'Left'
        settings = dict(SETTINGS, hand='Right', gold_fallback=True)
        gold = self.detection(100)
        gold.c = (105, 51)
        for ts in (.2, .4, .6):
            self.track.update(pos=gold.c, det_seen=ts, mask=gold.mask)
            self.assertIsNone(self.observe(ts, [gold], settings))
        self.track['det_seen'] = .8
        cue = self.observe(.8, [gold], settings)
        self.assertEqual('FIXED_BOX_GOLD', cue['source'])
        self.assertEqual(1, cue['jewel_id'])

    def test_failed_hand_model_uses_gold_only_when_fallback_is_enabled(self):
        self.trigger.close()
        self.trigger.key = None
        self.trigger.hand_factory = Mock(side_effect=RuntimeError('test hand model unavailable'))
        self.assertIsNone(self.observe(0, [self.home]))
        gold = self.detection(100)
        gold.c = (105, 51)
        for ts in (.2, .4, .6, .8, 1.):
            self.track.update(pos=gold.c, det_seen=ts, mask=gold.mask)
            self.assertIsNone(self.observe(ts, [gold]))
        settings = dict(SETTINGS, gold_fallback=True)
        for ts in (1.2, 1.4, 1.6):
            self.track['det_seen'] = ts
            self.assertIsNone(self.observe(ts, [gold], settings))
        self.track['det_seen'] = 1.8
        cue = self.observe(1.8, [gold], settings)
        self.assertEqual('FIXED_BOX_GOLD', cue['source'])
        self.assertEqual('test hand model unavailable', cue['hand_model_error'])

    def test_right_selection_ignores_left_and_accepts_right_without_two_hand_ambiguity(self):
        self.track.update(state='OCCLUDED')
        self.pinch(35)
        left = self.model.detect.return_value[0]
        left.handedness = 'Left'
        settings = dict(SETTINGS, hand='Right')
        for ts in (.2, .4, .6, .8, 1., 1.2):
            self.assertIsNone(self.observe(ts, [], settings))
        self.assertFalse(self.trigger.preview['hands'][0]['selected'])
        right = SimpleNamespace(points=left.points.copy(), handedness='Right')
        self.model.detect.return_value = [left, right]
        for ts in (1.4, 1.6, 1.8):
            self.assertIsNone(self.observe(ts, [], settings))
        result = self.observe(2., [], settings)
        self.assertEqual(1, result['jewel_id'])
        self.assertEqual(['Right'], result['trigger_hands'])

    def test_any_selection_preserves_uncertainty_for_two_qualifying_hands(self):
        self.track.update(state='OCCLUDED')
        self.pinch(35)
        left = self.model.detect.return_value[0]
        left.handedness = 'Left'
        self.model.detect.return_value.append(SimpleNamespace(points=left.points.copy(), handedness='Right'))
        for ts in (.2, .4, .6):
            self.assertIsNone(self.observe(ts, []))
        result = self.observe(.8, [])
        self.assertIsNone(result['jewel_id'])
        self.assertEqual([1], result['candidate_ids'])

    def test_focus_frames_can_complete_pinch_without_fabricating_gold_sightings(self):
        import jewel_tracker_5 as app
        worker = app.Worker(0, [], continuous=True)
        worker.mode = 'LOCKED'
        worker.set_recording_settings(SETTINGS)
        self.track.update(state='OCCLUDED')
        tracked = SimpleNamespace(id=1, state='OCCLUDED', pos=np.array((110, 52)),
                                  det_seen=0, in_hand=True, det_mask=self.home.mask,
                                  det_origin=self.home.origin, acid_done=False)
        worker.tracks = [tracked]
        self.pinch(35)
        result = None
        for ts in (.2, .4, .6, .8, 1., 1.2):
            tracked.det_seen = ts  # Existing focus bookkeeping may refresh this.
            worker._sample_recording_box(self.frame, ts, [], fresh_detections=False)
            frame, observed, tracks, dets = worker.box_sample
            self.assertEqual(-float('inf'), tracks[0]['det_seen'])
            cue = self.trigger.observe((0, SETTINGS['points'], 'Any'), frame, observed,
                                       tracks, dets, SETTINGS)
            if cue is not None:
                result = cue
        self.assertEqual(1, result['jewel_id'])
        self.assertEqual('FIXED_BOX_PINCH', result['source'])

    def test_area_update_preserves_home_and_batch_reset_clears_it(self):
        home = self.trigger.homes[1]
        self.trigger.prepare((0, ((.4, .2), (.8, .7))))
        self.assertIs(home, self.trigger.homes[1])
        self.trigger.prepare((1, ((.4, .2), (.8, .7))))
        self.assertEqual({}, self.trigger.homes)

    def test_home_is_learned_while_waiting_for_settled_area_without_starting_clip(self):
        self.trigger.homes.clear()
        pending = dict(SETTINGS, points=(), home_exclusion=(80, 30, 145, 80))
        self.assertIsNone(self.observe(0, [self.home], pending))
        home = self.trigger.homes[1]
        self.model.detect.assert_not_called()
        self.track.update(state='OCCLUDED')
        self.pinch(35)
        for ts in (.2, .4, .6):
            self.assertIsNone(self.observe(ts, [], SETTINGS))
        self.assertIs(home, self.trigger.homes[1])
        self.assertEqual(1, self.observe(.8, [], SETTINGS)['jewel_id'])

    def pinch(self, gap):
        # At this frame size the padded inference crop starts at (0, 0).
        points = np.full((21, 2), (110., 52.))
        points[4] = (110-gap/2, 52)
        points[8] = (110+gap/2, 52)
        self.model.detect.return_value = [SimpleNamespace(points=points)]

    def test_home_still_occupied_never_starts_recording(self):
        self.pinch(35)
        for ts in (.2, .4, .6, .8, 1.):
            self.assertIsNone(self.observe(ts, [self.home]))
        self.model.detect.assert_called()
        self.assertEqual(1, len(self.trigger.preview['hands']))
        self.assertEqual(0, self.trigger.preview['hold_seconds'])
        self.assertEqual('Waiting for jewel pickup', self.trigger.preview['state'])

    def test_landmarks_are_preview_only_and_expired_results_are_hidden(self):
        self.pinch(35)
        self.observe(.2, [self.home])
        original = self.frame.copy()
        rendered = draw_hand_preview(self.frame, .3, self.trigger.preview)
        self.assertFalse(np.array_equal(rendered, original))
        np.testing.assert_array_equal(self.frame, original)
        self.assertIs(self.frame, draw_hand_preview(self.frame, 1., self.trigger.preview))
        self.assertEqual(35, self.trigger.preview['hands'][0]['distance_px'])

    def test_grip_outside_stone_corner_starts_inside_extended_area(self):
        area = StoneRecordingArea()
        area.update((90, 40, 130, 70), 0, self.frame.shape, 20, False)
        points = area.update((90, 40, 130, 70), .5, self.frame.shape, 20, False)
        settings = dict(SETTINGS, points=points)
        self.track.update(state='OCCLUDED')
        self.pinch(8)
        self.model.detect.return_value[0].points[:] = (134, 72)
        self.model.detect.return_value[0].points[4] = (130, 72)
        self.model.detect.return_value[0].points[8] = (138, 72)
        # The hand crop begins at x=0, y=0 for these bounds.
        for ts in (.2, .4, .6):
            self.assertIsNone(self.observe(ts, [], settings))
        self.assertEqual(1, self.observe(.8, [], settings)['jewel_id'])

    def test_hidden_bangle_with_adjusted_wide_pinch_starts_once(self):
        self.track.update(state='OCCLUDED')
        self.pinch(50)
        self.model.detect.return_value[0].points[8, 0] = 139
        for ts in (.2, .4, .6, .8, 1., 1.2):
            self.assertIsNone(self.observe(ts, []))
        settings = dict(SETTINGS, pinch_px=60)
        for ts in (1.4, 1.6, 1.8):
            self.assertIsNone(self.observe(ts, [], settings))
        result = self.observe(2., [], settings)
        self.assertEqual(1, result['jewel_id'])
        self.assertEqual('FIXED_BOX_PINCH', result['source'])
        self.assertIsNone(self.observe(2.2, [], settings))

    def test_stream_gap_restarts_observation_window(self):
        self.track.update(state='OCCLUDED')
        self.pinch(35)
        for ts in (.2, .4, .6, 2., 2.2, 2.4):
            self.assertIsNone(self.observe(ts, []))
        self.assertEqual(1, self.observe(2.6, [])['jewel_id'])

    def test_optional_gold_fallback_requires_real_detection_and_clear_box(self):
        self.model.detect.return_value = []
        gold = self.detection(100)
        gold.c = (105, 51)
        for ts in (.2, .4, .6, .8, 1., 1.2):
            self.track.update(pos=gold.c, det_seen=ts, mask=gold.mask)
            self.assertIsNone(self.observe(ts, [gold]))
        settings = dict(SETTINGS, gold_fallback=True)
        for ts in (1.4, 1.6, 1.8):
            self.track['det_seen'] = ts
            self.assertIsNone(self.observe(ts, [], settings))
        for ts in (2., 2.2, 2.4):
            self.track['det_seen'] = ts
            self.assertIsNone(self.observe(ts, [gold], settings))
        self.track['det_seen'] = 2.6
        self.assertEqual('FIXED_BOX_GOLD', self.observe(2.6, [gold], settings)['source'])


    def test_open_hand_blocks_gold_fallback(self):
        self.pinch(60)
        settings = dict(SETTINGS, gold_fallback=True)
        gold = self.detection(100)
        gold.c = (105, 51)
        for ts in (.2, .4, .6, .8, 1., 1.2, 1.4):
            self.track.update(pos=gold.c, det_seen=ts, mask=gold.mask)
            self.assertIsNone(self.observe(ts, [gold], settings))
        self.model.detect.assert_called()


class RecordingCycleTests(TestCase):
    def test_model_load_failure_reports_unavailable_without_stopping_capture(self):
        with monitor_case() as (monitor, events):
            monitor.box_trigger.hand_factory = Mock(side_effect=RuntimeError('test unavailable'))
            sample = dict(snap(100), box_settings=dict(SETTINGS, points=(), gold_fallback=True))
            monitor._box_cue(sample)
            try:
                monitor.box_job[0].result(timeout=2)
                monitor._box_cue(sample)
                self.assertFalse(monitor.stop_event.is_set())
                self.assertIsNone(monitor.box_error)
                self.assertTrue(any(kind == 'hand_model' and p['message'] == 'Hand model: unavailable'
                                    for kind, p in events))
                self.assertFalse(any(kind == 'hand_model' and p['message'] == 'Hand model: ready'
                                     for kind, p in events))
            finally:
                monitor.box_pool.shutdown(wait=True, cancel_futures=True)
                monitor.box_trigger.close()

    def test_background_home_checks_do_not_prevent_initial_stone_area_calibration(self):
        with monitor_case() as (monitor, events):
            monitor._box_cue = Mock(return_value=None)
            monitor.box_job = (Mock(), (), False)
            settings = dict(margin_percent=20, pinch_px=40, gold_fallback=False)
            for ts in (100, 100.3, 100.6):
                monitor.step(dict(snap(ts), box_settings=settings, stone_detected=True))
            self.assertTrue(monitor.recording_points)

    def test_area_stays_fixed_during_pinch_attempt_before_recording_starts(self):
        with monitor_case() as (monitor, events):
            monitor._box_cue = Mock(return_value=None)
            settings = dict(margin_percent=20, pinch_px=40, gold_fallback=False)
            for ts in (100, 100.3, 100.6):
                monitor.step(dict(snap(ts), box_settings=settings, stone_detected=True))
            points = monitor.recording_points
            self.assertTrue(points)
            monitor.box_attempt_pending = True
            for ts in (100.9, 101.2, 101.5):
                monitor.step(dict(snap(ts), box_settings=settings, stone_detected=True,
                                  stone=(90, 70, 230, 210)))
                self.assertEqual(points, monitor.recording_points)
            monitor.box_attempt_pending = False
            for ts in (101.8, 102.4):
                monitor.step(dict(snap(ts), box_settings=settings, stone_detected=True,
                                  stone=(90, 70, 230, 210)))
            self.assertNotEqual(points, monitor.recording_points)

    def test_previous_unassigned_cycle_does_not_block_later_unassigned_recording(self):
        with monitor_case() as (monitor, events):
            monitor.captured_ids.add(None)
            monitor._box_cue = Mock(return_value=dict(
                jewel_id=None, source='FIXED_BOX_PINCH', candidate_ids=[1, 2],
                timestamp=99.5, confirmed_at=100))
            monitor.step(dict(snap(100), box_settings=SETTINGS))
            self.assertIsNotNone(monitor.active)
            self.assertIsNone(monitor.active['jewel_id'])
            self.assertEqual('MISSED_OR_AMBIGUOUS', monitor.active['identity_status'])

    def test_auto_stone_area_needs_no_drawn_box_and_stone_motion_cannot_start_cycle(self):
        with monitor_case() as (monitor, events):
            monitor._box_cue = Mock(return_value=None)
            settings = dict(margin_percent=20, pinch_px=40, gold_fallback=False)
            for ts in (100, 100.3, 100.6):
                monitor.step(dict(snap(ts), box_settings=settings, stone_detected=True))
            effective = monitor._box_cue.call_args.args[0]['box_settings']
            self.assertTrue(effective['points'])
            for ts in (100.9, 101.2, 101.5):
                monitor.step(dict(snap(ts), stone=(60, 70, 240, 220),
                                  box_settings=settings, stone_detected=True))
            self.assertIsNone(monitor.active)
            self.assertTrue(any(kind == 'recording_area' and p['points'] for kind, p in events))

    def test_hand_model_loads_in_background_before_a_box_or_pickup(self):
        with monitor_case() as (monitor, events):
            model = Mock()
            monitor.box_trigger.hand_factory = Mock(return_value=model)
            s = dict(snap(100), mode='ARMING', box_settings=dict(SETTINGS, points=()))
            self.assertIsNone(monitor._box_cue(s))
            try:
                monitor.box_job[0].result(timeout=2)
                self.assertIsNone(monitor._box_cue(s))
                monitor.box_trigger.hand_factory.assert_called_once_with()
                model.detect.assert_not_called()
                self.assertTrue(monitor.box_ready)
                self.assertEqual(['Hand model: loading…', 'Hand model: ready'],
                                 [p['message'] for kind, p in events if kind == 'hand_model'])
            finally:
                monitor.box_pool.shutdown(wait=True, cancel_futures=True)
                monitor.box_trigger.close()

    def test_stone_roi_shift_is_ignored_then_box_events_open_j1_and_j2(self):
        with monitor_case() as (monitor, events):
            def box_snap(ts, held):
                sample = snap(ts, held=held, contact=contact(ts, confirmed=True))
                return dict(sample, box_settings=SETTINGS,
                            rubbing_trigger=dict(timestamp=ts, confirmed_at=ts, frames=2))
            monitor._box_cue = Mock(return_value=None)
            for ts in (100, 100.2, 100.5):
                s = box_snap(ts, held=1)
                s['stone'] = (50, 60, 230, 210)
                monitor.step(s)
            self.assertIsNone(monitor.active)

            def cue(jid, ts):
                return dict(jewel_id=jid, source='FIXED_BOX_PINCH', timestamp=ts-.5, confirmed_at=ts)

            monitor._box_cue.return_value = cue(1, 101)
            monitor.step(dict(box_snap(101, 1), contact=None))
            j1 = monitor.active
            self.assertEqual(1, j1['jewel_id'])
            self.assertEqual(97, monitor.journal.pins[j1['id']][0])
            self.assertFalse(j1['live_rub'])
            self.assertIsNone(j1['baseline_at'])

            monitor._box_cue.return_value = cue(None, 101.5)
            monitor.step(dict(box_snap(101.5, 2), contact=None))
            self.assertIs(j1, monitor.active)

            monitor._queue_finalization = Mock()
            monitor._box_cue.return_value = cue(2, 102)
            monitor.step(dict(box_snap(102, 2), contact=None))
            monitor._queue_finalization.assert_called_once_with(j1)
            self.assertIsNone(j1['acid_onset'])
            self.assertEqual('NEXT_JEWEL_PICKUP', j1['close_reason'])
            self.assertEqual(2, monitor.active['jewel_id'])
