import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tests import test_home_rubbing as fixtures
from standalone.home_rubbing import HomeRubbingFlow
from pickup_trigger_test import make_engine,CandidateClips


class PickupOnlyTests(unittest.TestCase):
    update=fixtures.HomeRubbingTests.update
    def setUp(self):
        fixtures.HomeRubbingTests.setUp(self)
        self.flow=HomeRubbingFlow(self.events.append,pickup_only=True)

    def pick(self):
        self.update(present=True,stone=False)
        for x in (50,60,70):result=self.update(x,40,stone=False)
        return result

    def place(self,x=20,y=20):
        for _ in range(5):
            result=self.update(x,y,hand=False,near=False,present=x==20 and y==20,stone=False)
        return result

    def test_pickup_verifies_without_stone_or_rubbing(self):
        result=self.pick()
        self.assertEqual('PICKUP_VERIFIED',result['state'])
        self.assertEqual('VERIFIED',result['pickup_status'])
        self.assertEqual('DISABLED',result['rubbing_status'])
        self.assertIsNone(result['rubbing_at'])

    def test_verified_return_home_completes_instead_of_cancelling(self):
        self.pick();result=self.place()
        self.assertEqual('PICKUP_VERIFIED_RETURNED_HOME',result['state'])
        self.assertEqual('VERIFIED',result['pickup_status'])
        self.assertEqual('HOME',result['return_status'])
        self.assertFalse(result['active'])

    def test_occlusion_and_return_without_departure_is_unresolved(self):
        self.update(present=True,stone=False)
        self.update(40,40,visible=False,stone=False)
        result=self.place()
        self.assertEqual('PICKUP_UNRESOLVED_RETURNED_HOME',result['state'])
        self.assertEqual('UNRESOLVED',result['pickup_status'])
        self.assertEqual('HOME',result['return_status'])

    def test_missing_hand_and_gold_then_return_does_not_cancel(self):
        self.update(present=True,stone=False)
        self.update(visible=False,hand=False,near=False,stone=False)
        self.assertEqual('PICKUP_UNRESOLVED_RETURNED_HOME',self.place()['state'])

    def test_passing_hand_and_unchanged_jewel_cancels(self):
        self.update(present=True,stone=False)
        result=self.place()
        self.assertEqual('CANCELLED',result['state'])
        self.assertFalse(result['departure_verified'])

    def test_placement_elsewhere_on_bed_preserves_verified_pickup(self):
        self.pick();result=self.place(150,30)
        self.assertEqual('PICKUP_VERIFIED_RETURNED_BED',result['state'])
        self.assertEqual('VERIFIED',result['pickup_status'])

    def test_new_pickup_after_return_preserves_event_history(self):
        self.pick();first=self.place()
        self.update(20,20,hand=False,near=False,present=True,stone=False)
        second=self.update(present=True,stone=False)
        self.assertEqual(2,second['event_number'])
        self.assertEqual('PENDING',second['pickup_status'])
        self.assertEqual('VERIFIED',second['previous_event']['pickup_status'])
        self.assertEqual('VERIFIED',self.flow.history[1][0]['pickup_status'])
        self.assertEqual('HOME',first['return_status'])

    def test_rubbing_rule_does_not_run_in_pickup_only_mode(self):
        self.update(present=True)
        for x in (90,110)*6:result=self.update(x,100)
        self.assertIsNone(result['rubbing_at'])
        self.assertEqual(0,result['reversal_count'])
        self.assertEqual('DISABLED',result['rubbing_status'])

    def test_recording_closes_after_return_and_keeps_clip(self):
        frame=np.zeros((180,180,3),np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            clips=CandidateClips(Path(tmp),5);self.addCleanup(clips.close)
            self.pick();clips.update(frame,[self.jewel],self.t)
            self.assertIn(1,clips.writers)
            self.place();clips.update(frame,[self.jewel],self.t)
            self.assertNotIn(1,clips.writers)
            self.assertTrue(list(Path(tmp).glob('*.avi')))

    def test_both_factories_enable_pickup_only_for_home_and_aruco(self):
        import pickup_trigger_motion_class as motion
        for factory in (make_engine,motion.make_engine):
            for mode in ('home','cog'):
                engine=factory(mode,lambda e:None)
                self.assertTrue(engine.home_rubbing.pickup_only)
                self.assertTrue(engine.gold_bridge.flow.pickup_only)

    def test_stable_released_return_rearms_without_an_extra_idle_frame(self):
        self.pick()
        while self.jewel.workflow['active']:
            self.update(hand=False,near=False,present=True,stone=False)
        result=self.update(present=True,stone=False)
        self.assertEqual(2,result['event_number'])
        self.assertEqual('PENDING',result['pickup_status'])
        self.assertEqual('VERIFIED',result['previous_event']['pickup_status'])

    def unassigned_mask(self, key=9, x=25, y=25, named=None):
        self.engine.gold_masks=SimpleNamespace(tracks={key:dict(
            visible=True,state='GOLD_VISIBLE',ambiguous_mask=False,
            hand_candidates=0,jewel_id=named,cog=np.array([x,y],float))})

    def test_missing_id_and_hand_free_placement_stops_without_confirming_identity(self):
        self.pick()
        self.unassigned_mask()
        for _ in range(4):
            result=self.update(visible=False,hand=False,near=False,stone=False)
        self.assertEqual('RECORDING_STOPPED_RETURN_IDENTITY_PENDING',result['state'])
        self.assertFalse(result['active'])
        self.assertEqual('VERIFIED',result['pickup_status'])
        self.assertEqual('IDENTITY_PENDING',result['return_status'])
        self.assertIsNone(result['returned_at'])

    def test_missing_gold_and_missing_id_stop_recording_without_inventing_return(self):
        self.pick()
        for _ in range(5):
            result=self.update(visible=False,hand=False,near=False,stone=False)
        self.assertEqual('RECORDING_STOPPED_RETURN_UNVERIFIED',result['state'])
        self.assertEqual('VERIFIED',result['pickup_status'])
        self.assertEqual('UNVERIFIED',result['return_status'])
        self.assertFalse(result['active'])
        self.assertIsNone(result['returned_at'])

    def test_hand_near_bed_prevents_missing_id_recording_timeout(self):
        self.pick()
        self.unassigned_mask()
        for _ in range(10):
            result=self.update(visible=False,hand=True,near=True,stone=False)
        self.assertTrue(result['active'])

    def test_brief_hand_dropout_does_not_close_and_returning_hand_resets_timer(self):
        self.pick()
        for _ in range(3):
            self.assertTrue(self.update(visible=False,hand=False,near=False,stone=False)['active'])
        self.update(visible=False,hand=True,near=True,stone=False)
        for _ in range(3):
            self.assertTrue(self.update(visible=False,hand=False,near=False,stone=False)['active'])

    def test_other_named_jewel_cannot_be_used_as_pending_return(self):
        self.pick()
        self.unassigned_mask(named=2)
        for _ in range(5):
            result=self.update(visible=False,hand=False,near=False,stone=False)
        self.assertEqual('RECORDING_STOPPED_RETURN_UNVERIFIED',result['state'])
        self.assertFalse(result['return_identity_pending'])

    def test_absent_detection_never_confirms_an_unverified_pickup(self):
        self.update(present=True,stone=False)
        for _ in range(5):
            result=self.update(visible=False,hand=False,near=False,stone=False)
        self.assertEqual('UNRESOLVED',result['pickup_status'])
        self.assertFalse(result['departure_verified'])

    def test_later_same_id_resolves_saved_return_without_restarting_recording(self):
        self.pick()
        for _ in range(5):
            result=self.update(visible=False,hand=False,near=False,stone=False)
        stopped=result['recording_stopped_at']
        for _ in range(4):
            result=self.update(hand=False,near=False,present=True,stone=False)
            self.assertFalse(result['active'])
        self.assertEqual('PICKUP_VERIFIED_RETURNED_HOME',result['state'])
        self.assertEqual('HOME',result['return_status'])
        self.assertEqual(stopped,result['recording_stopped_at'])
        self.assertEqual(1,result['event_number'])

    def test_new_grip_after_fallback_stop_starts_separate_event(self):
        self.pick()
        for _ in range(5):
            self.update(visible=False,hand=False,near=False,stone=False)
        result=self.update(present=True,stone=False)
        self.assertEqual(2,result['event_number'])
        self.assertTrue(result['active'])
        self.assertEqual('VERIFIED',result['previous_event']['pickup_status'])

    def test_missing_id_stop_closes_writer_in_both_scripts(self):
        import pickup_trigger_motion_class as motion
        frame=np.zeros((180,180,3),np.uint8)
        for cls in (CandidateClips,motion.CandidateClips):
            with self.subTest(script=cls.__module__),tempfile.TemporaryDirectory() as tmp:
                self.setUp()
                clips=cls(Path(tmp),5)
                try:
                    self.pick();clips.update(frame,[self.jewel],self.t)
                    self.assertIn(1,clips.writers)
                    for _ in range(5):
                        self.update(visible=False,hand=False,near=False,stone=False)
                        clips.update(frame,[self.jewel],self.t)
                    self.assertNotIn(1,clips.writers)
                    self.assertEqual(1,len(list(Path(tmp).glob('*.avi'))))
                finally:
                    clips.close()
