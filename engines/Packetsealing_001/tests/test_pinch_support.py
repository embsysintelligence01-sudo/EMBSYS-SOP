import unittest
from types import SimpleNamespace

import numpy as np

from standalone.hand_keypoints import pinch_measurement
from standalone.home_rubbing import HomeRubbingFlow, RubEvent


def hand_at(x=30, y=30, closed=True):
    points=np.full((21,2),(x,y),np.float32)
    points[0]=(x,y+50)
    points[5]=(x-15,y+5);points[17]=(x+15,y+5)
    points[8]=(x+(5 if closed else 35),y)
    for index in (12,16,20):points[index]=(x+40,y-20)
    return SimpleNamespace(points=points,scale=50.)


class PinchMeasurementTests(unittest.TestCase):
    def test_thumb_can_pinch_each_of_four_fingertips(self):
        for tip in (8,12,16,20):
            hand=hand_at(closed=False)
            hand.points[tip]=hand.points[4]+[5,0]
            result=pinch_measurement(hand)
            self.assertTrue(result['closed'])
            self.assertEqual(tip,result['finger_tip'])

    def test_normalization_is_independent_of_image_scale(self):
        hand=hand_at();first=pinch_measurement(hand)
        hand.points=hand.points*3+100
        self.assertAlmostEqual(first['distance_ratio'],pinch_measurement(hand)['distance_ratio'])

    def test_enter_and_exit_thresholds_have_hysteresis(self):
        hand=hand_at();hand.points[8]=hand.points[4]+[15,0]
        self.assertFalse(pinch_measurement(hand)['closed'])
        self.assertTrue(pinch_measurement(hand,was_closed=True)['closed'])

    def test_invalid_or_collapsed_landmarks_do_not_count_as_a_pinch(self):
        for points in (np.zeros((21,2)),np.full((21,2),np.nan),np.zeros((5,2))):
            self.assertFalse(pinch_measurement(SimpleNamespace(points=points))['valid'])


class LinkedPinchTests(unittest.TestCase):
    def setUp(self):
        self.events=[];self.flow=HomeRubbingFlow(self.events.append,pickup_only=True)
        self.roi=np.full((220,220),255,np.uint8);self.t=0.
        self.stone=(75,75,125,125)
        self.jewel=SimpleNamespace(id=3,box=(20,20,40,40),home_cog=np.array([30.,30.]),
            current_cog=None,associated_hand=7,contact_at=0.,state='UNCERTAIN',metrics={})
        self.engine=SimpleNamespace(jewels=[self.jewel],live_gold={},hand_tracks={},
            home_guard=np.zeros_like(self.roi),global_uncertain=False)

    def step(self,x=30,y=30,closed=True,near=True,hand=True,stone=True,hand_id=7):
        self.t+=.25
        self.jewel.metrics=dict(hand_jewel_proximity=near,present=False,home_covered=hand)
        self.jewel.associated_hand=hand_id if hand else None
        self.engine.hand_tracks={hand_id:dict(detection=hand_at(x,y,closed),cog=np.array([x,y],float))} if hand else {}
        self.flow.update(self.engine,self.stone if stone else None,self.roi,self.t)
        return getattr(self.jewel,'workflow',None)

    def grip(self):
        for _ in range(3):result=self.step()
        return result

    def test_proximity_starts_recording_before_pinch_confirmation(self):
        result=self.step(closed=False)
        self.assertTrue(result['active'])
        self.assertEqual('PROXIMITY_ONLY',result['grip_status'])

    def test_sustained_pinch_strengthens_grip_without_verifying_pickup(self):
        result=self.grip()
        self.assertTrue(result['pinch_grip_supported'])
        self.assertEqual('PINCH_SUPPORTED',result['grip_status'])
        self.assertFalse(result['departure_verified'])
        self.assertIsNone(result['pickup_verified_at'])

    def test_same_hand_pinch_at_stone_preserves_hidden_jewel_recording(self):
        self.grip()
        for _ in range(3):result=self.step(90,90,near=False)
        self.assertEqual('PINCH_AT_STONE',result['grip_status'])
        self.assertTrue(result['active'])
        self.assertFalse(result['departure_verified'])
        self.assertIsNone(result['rubbing_at'])
        self.assertEqual(3,result['jewel_id'])

    def test_stone_pinch_without_prior_jewel_interaction_creates_no_event(self):
        for _ in range(5):self.step(90,90,near=False)
        self.assertNotIn(3,self.flow.events)

    def test_missing_stone_does_not_claim_stone_action_or_block_recording(self):
        self.grip()
        for _ in range(4):result=self.step(90,90,near=False,stone=False)
        self.assertFalse(result['pinch_stone_supported'])
        self.assertTrue(result['active'])

    def test_single_frame_pinch_or_flicker_does_not_strengthen_grip(self):
        for closed in (True,False,True,False,True):result=self.step(closed=closed)
        self.assertFalse(result['pinch_grip_supported'])

    def test_hand_identity_change_restarts_pinch_observations(self):
        self.step();self.step()
        result=self.step(hand_id=9)
        self.assertFalse(result['pinch_grip_supported'])

    def test_another_hands_stone_pinch_is_not_borrowed_by_candidate(self):
        self.grip()
        self.jewel.metrics['hand_jewel_proximity']=False
        for _ in range(3):
            self.t+=.25
            self.engine.hand_tracks={
                7:dict(detection=hand_at(closed=False),cog=np.array([30.,30.])),
                9:dict(detection=hand_at(90,90),cog=np.array([90.,90.]))}
            self.flow.update(self.engine,self.stone,self.roi,self.t)
        self.assertFalse(self.jewel.workflow['pinch_stone_supported'])

    def test_pinch_supported_event_is_not_discarded_on_hand_transfer(self):
        self.grip()
        other=SimpleNamespace(id=1,associated_hand=7,metrics=dict(hand_jewel_proximity=True))
        self.jewel.metrics['hand_jewel_proximity']=False
        self.engine.jewels.append(other)
        status=dict(trusted=True,matrix=[[1.,0.,0.],[0.,1.,0.]])
        self.flow.assign_current_hands(self.engine,self.t+.25,status)
        event=self.flow.events[3]
        self.assertFalse(event.discard_recording)
        self.assertTrue(event.snapshot()['active'])
