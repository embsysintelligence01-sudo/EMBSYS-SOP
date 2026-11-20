import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from standalone.home_rubbing import HomeRubbingFlow, RubEvent


class HandCandidateTransferTests(unittest.TestCase):
    def setUp(self):
        self.flow = HomeRubbingFlow(lambda event: None, pickup_only=True)
        self.roi = np.full((180, 500), 255, np.uint8)
        self.old = self.jewel(2, (80, 30), near=False)
        self.earring = self.jewel(3, (20, 100), near=True)
        hand = SimpleNamespace(points=np.full((21, 2), (20, 100), np.float32), scale=40.)
        self.engine = SimpleNamespace(jewels=[self.old, self.earring], live_gold={},
            hand_tracks={4: dict(detection=hand, cog=np.array([20., 100.]))},
            home_guard=np.zeros_like(self.roi), global_uncertain=False)
        self.flow.events[2] = RubEvent(2, 0., hand_id=4, pickup_only=True)

    @staticmethod
    def jewel(jid, position, near=False):
        x, y = position
        return SimpleNamespace(id=jid, box=(x-10,y-10,x+10,y+10),
            home_cog=np.array(position,float), state='UNCERTAIN', contact_at=.1,
            associated_hand=4, metrics=dict(hand_jewel_proximity=near,present=False),
            current_cog=None, gold_quality=dict(accepted=False,reason='no_fresh_gold_detection'))

    def test_stale_owner_yields_hand_to_new_earring_candidate(self):
        self.flow.update(self.engine,None,self.roi,.25)
        self.assertEqual(4,self.flow.events[3].hand_id)
        self.assertIsNone(self.flow.events[2].hand_id)
        self.assertFalse(self.flow.events[2].departure_verified)
        self.assertEqual('CANCELLED',self.flow.events[2].state)
        self.assertTrue(self.old.workflow['discard_recording'])
        self.assertTrue(self.earring.workflow['active'])

    def test_transfer_does_not_depend_on_jewel_processing_order(self):
        self.engine.jewels.reverse()
        self.test_stale_owner_yields_hand_to_new_earring_candidate()

    def test_two_simultaneous_contacts_do_not_choose_a_new_owner(self):
        self.old.metrics['hand_jewel_proximity']=True
        self.flow.update(self.engine,None,self.roi,.25)
        self.assertNotIn(3,self.flow.events)
        self.assertFalse(self.flow.events[2].departure_verified)
        self.assertFalse(self.flow.events[2].discard_recording)

    def test_transfer_preserves_event_with_verified_departure(self):
        self.flow.events[2].departure_verified=True
        self.flow.update(self.engine,None,self.roi,.25)
        self.assertFalse(self.old.workflow['discard_recording'])
        self.assertEqual('VERIFIED',self.old.workflow['pickup_status'])

    def test_transfer_preserves_event_with_provisional_movement_evidence(self):
        self.flow.events[2].away_hits=1
        self.flow.update(self.engine,None,self.roi,.25)
        self.assertFalse(self.old.workflow['discard_recording'])
        self.assertNotEqual('CANCELLED',self.old.workflow['state'])

    def test_transfer_preserves_fresh_complete_displacement_not_yet_voted(self):
        self.flow.events[2].origin_cog=[80.,30.]
        self.old.current_cog=np.array([130.,30.])
        self.old.gold_quality=dict(accepted=True,partial=False)
        self.flow.update(self.engine,None,self.roi,.25)
        self.assertFalse(self.old.workflow['discard_recording'])

    def test_transfer_deletes_only_false_event_segments_and_images_in_both_scripts(self):
        from pickup_trigger_test import CandidateClips
        from pickup_trigger_motion_class import CandidateClips as MotionClips
        frame=np.zeros((180,500,3),np.uint8)
        for cls in (CandidateClips,MotionClips):
            with self.subTest(script=cls.__module__),tempfile.TemporaryDirectory() as tmp:
                self.setUp()
                folder=Path(tmp);clips=cls(folder,5)
                try:
                    # A previous real event for the same jewel must survive.
                    self.old.workflow=dict(active=True,event_number=0,state='PICKUP_VERIFIED')
                    clips.update(frame,[self.old],0.)
                    self.old.workflow['active']=False
                    clips.update(frame,[self.old],.2)
                    self.old.workflow=self.flow.events[2].snapshot()
                    clips.update(frame,[self.old],.4)
                    clips.update(frame,[self.old],3.)  # Gap starts a second segment.
                    self.assertEqual(3,len(list(folder.glob('J2*.avi'))))
                    self.flow.update(self.engine,None,self.roi,3.25)
                    clips.update(frame,self.engine.jewels,3.25)
                    self.assertNotIn(2,clips.writers)
                    self.assertEqual(['J2_candidate_001.avi'],[p.name for p in folder.glob('J2*.avi')])
                    self.assertEqual(['J2_candidate_001_start.jpg'],[p.name for p in folder.glob('J2*_start.jpg')])
                    self.assertTrue(self.old.workflow['recording_deleted'])
                    self.assertIn(3,clips.writers)
                    clips.update(frame,self.engine.jewels,3.5)
                    self.assertEqual(1,len(list(folder.glob('J3*.avi'))))
                finally:
                    clips.close()

    def show_displaced_earring(self, partial=False):
        # Regression: the recorded earring moved ~381 px while the old-home
        # background ring changed 20.6%, narrowly exceeding its 20% threshold.
        self.earring.metrics.update(hand_jewel_proximity=False,present=False,
            reliable=False,ring_changed=.206,bed_fraction=1.,edge_retention=0.)
        self.earring.current_cog=np.array([401.,100.])
        self.earring.gold_quality=dict(accepted=True,partial=partial,area_ratio=1.03)
        mask=np.zeros_like(self.roi);mask[90:110,391:411]=255
        self.engine.live_gold[3]=SimpleNamespace(mask=mask,box=(391,90,411,110))
        self.engine.hand_tracks={}

    def test_complete_same_id_movement_verifies_despite_home_background_failure(self):
        self.flow.update(self.engine,None,self.roi,.25)
        self.show_displaced_earring()
        for t in (.5,.75,1.):
            self.flow.update(self.engine,None,self.roi,t)
        self.assertEqual('VERIFIED',self.earring.workflow['pickup_status'])
        self.assertIsNotNone(self.earring.workflow['pickup_verified_at'])

    def test_deferred_contact_can_start_verification_after_hand_has_left(self):
        self.show_displaced_earring()
        for t in (.25,.5,.75):
            self.flow.update(self.engine,None,self.roi,t)
        self.assertEqual('VERIFIED',self.earring.workflow['pickup_status'])

    def test_partial_mask_shift_cannot_start_deferred_pickup(self):
        self.show_displaced_earring(partial=True)
        for t in (.25,.5,.75):
            self.flow.update(self.engine,None,self.roi,t)
        self.assertNotIn(3,self.flow.events)

    def test_missing_detector_output_alone_cannot_verify_departure(self):
        self.flow.update(self.engine,None,self.roi,.25)
        self.earring.metrics['hand_jewel_proximity']=False
        self.engine.hand_tracks={}
        for t in (.5,.75,1.):
            self.flow.update(self.engine,None,self.roi,t)
        self.assertFalse(self.earring.workflow['departure_verified'])

    def test_same_id_displacement_without_prior_contact_does_not_start_event(self):
        self.show_displaced_earring()
        self.earring.contact_at=None
        self.flow.update(self.engine,None,self.roi,.25)
        self.assertNotIn(3,self.flow.events)

    def test_delayed_id_recovery_verifies_even_after_recording_has_closed(self):
        self.flow.update(self.engine,None,self.roi,.25)
        self.engine.hand_tracks={}
        self.earring.metrics['hand_jewel_proximity']=False
        for t in (.5,.75,1.,1.25,1.5):
            self.flow.update(self.engine,None,self.roi,t)
        self.assertFalse(self.earring.workflow['active'])
        self.assertFalse(self.earring.workflow['departure_verified'])
        self.show_displaced_earring()
        for t in (1.75,2.,2.25,2.5):
            self.flow.update(self.engine,None,self.roi,t)
            self.assertFalse(self.earring.workflow['active'])
        self.assertEqual('PICKUP_VERIFIED_RETURNED_BED',self.earring.workflow['state'])
        self.assertEqual('VERIFIED',self.earring.workflow['pickup_status'])
