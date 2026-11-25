"""A stale stone outline or pending mask cannot freeze independent monitoring."""
from concurrent.futures import Future
import unittest
from unittest.mock import Mock, patch
import numpy as np
import cv2

from continuous_testing import StonePose, RubReference
from test_continuous_testing import scene, STONE, QUAD, Detector, Photos
from test_acid_application import DETECTION
from test_sop_revision import monitor_case, snap, contact
from sop_evidence import contact_region, validate_mark


class StoneRecoveryTests(unittest.TestCase):
    def test_established_coordinates_do_not_expire_with_outline_age(self):
        pose=StonePose();pose.update(scene(),1,STONE)
        pose.uncertain=True
        np.testing.assert_allclose(QUAD,pose.monitoring_quad(STONE),atol=1)
        np.testing.assert_allclose(QUAD,pose.monitoring_quad(None),atol=1)
        self.assertEqual(1,pose.geometry_ts)

    def test_displaced_detector_box_requires_coordinate_recovery(self):
        pose=StonePose();pose.update(scene(),1,STONE)
        moved=tuple(v+100 if i%2==0 else v for i,v in enumerate(STONE))
        self.assertIsNone(pose.monitoring_quad(moved))

    def test_initial_outline_cannot_include_dark_object_beyond_detected_stone(self):
        frame=scene();frame[200:225,80:220]=65
        pose=StonePose()
        self.assertFalse(pose.update(frame,1,STONE))
        self.assertIsNone(pose.quad)

    def stuck(self):
        pose=StonePose()
        pose.quad=QUAD+[100,0]
        return pose

    def test_repeated_detector_supported_outline_recovers_from_stale_pose(self):
        pose=self.stuck()
        self.assertFalse(pose.update(scene(),1,STONE))
        self.assertFalse(pose.update(scene(),1.1,STONE))
        self.assertTrue(pose.update(scene(),1.21,STONE))
        self.assertEqual(1,pose.generation)
        np.testing.assert_allclose(QUAD,pose.quad,atol=1)
        self.assertEqual(1.21,pose.recovered_at)
        self.assertTrue(pose.update(scene(),1.4,STONE))
        self.assertEqual(1,pose.generation)

    def test_duplicate_timestamps_cannot_confirm_recovery(self):
        pose=self.stuck()
        for _ in range(10):self.assertFalse(pose.update(scene(),1,STONE))
        self.assertEqual(1,pose.recovery_frames)
        self.assertEqual(0,pose.generation)

    def test_coverage_and_frame_gap_restart_recovery(self):
        for missing in ('covered','gap'):
            with self.subTest(missing=missing):
                pose=self.stuck();pose.update(scene(),1,STONE)
                if missing=='covered':
                    covered=scene();covered[60:200,80:220]=220
                    self.assertFalse(pose.update(covered,1.1,STONE))
                ts=2 if missing=='gap' else 1.2
                self.assertFalse(pose.update(scene(),ts,STONE))
                self.assertFalse(pose.update(scene(),ts+.1,STONE))
                self.assertTrue(pose.update(scene(),ts+.21,STONE))


class IndependentFlowTests(unittest.TestCase):
    def test_mask_and_acid_use_established_coordinates_without_fresh_outline(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            image=scene(mark=True)
            a['contact_region']=contact_region(contact(100),QUAD)
            a['mask_candidate']=RubReference(Photos(),scene(),99,QUAD).after(Photos(),image,102,QUAD)
            with patch.object(m.pose,'update',return_value=False):
                for ts in (102,102.08,102.16):
                    m.step(snap(ts,held=1,frame=image))
                self.assertIsNotNone(a['mask'])
                self.assertEqual('ESTABLISHED_COORDINATES_STATIONARY_BED',a['stone_coordinate_mode'])
                wet=scene(mark=True,acid=True)
                m._acid_observation(102.3,a['id'],[DETECTION],(wet,QUAD,False,False,False))
                m._verify_application(a,wet,QUAD,102.34,False,False)
                self.assertEqual(102.3,a['acid_onset'])
                for ts in (103,104,105,106):
                    m.step(snap(ts,held=1,frame=wet))
                    self.assertTrue(m._pose_known(a,ts))
            self.assertEqual(100,m.pose.geometry_ts)

    def test_live_detector_context_uses_continuity_after_outline_expiry(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            m._baseline(a,scene(),100.2,QUAD)
            a['acid_baseline_checked']=True
            m.detector=Detector()
            with patch.object(m.pose,'update',return_value=False):
                m.step(snap(102,held=1))
            self.assertIsNotNone(m.acid_context[1])
            self.assertTrue(m.acid_context[3])
            self.assertTrue(m._pose_known(a,102))
            m.acid_job.result(timeout=5)

    def test_jewel_placement_starts_baseline_without_mask_or_clearance(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            source=self.outside_hand()
            with patch('engines.purity.workflow.baseline_roi_interference',return_value='CLOTH_OR_GLARE') as check:
                m.step(snap(100.1,frame=source))
            check.assert_not_called()
            self.assertTrue(m.pose.uncertain)
            self.assertEqual(100.1,a['baseline_at'])
            self.assertEqual('JEWEL_PLACED_IN_TEST_BED',a['baseline_source'])
            self.assertIsNone(a['returned'])  # independent return debounce
            self.assertIsNone(a['mask'])
            np.testing.assert_array_equal(source,a['return_frame'])
            self.assertTrue(any(kind=='baseline' for kind,_ in events))

    def test_placement_baseline_is_not_replaced_by_later_mask_or_camera_frame(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            m.step(snap(100.1))
            baseline=a['fresh_gate'].baseline.copy()
            a['mask']=RubReference(Photos(),scene(),99,QUAD).after(Photos(),scene(mark=True),100.2,QUAD)
            m.step(snap(100.3,frame=scene(mark=True)))
            self.assertEqual(100.1,a['baseline_at'])
            self.assertEqual('JEWEL_PLACED_IN_TEST_BED',a['baseline_source'])
            np.testing.assert_array_equal(baseline,a['fresh_gate'].baseline)
            self.assertEqual(1,len(a['baseline_history']))

    def test_preplacement_model_hit_does_not_poison_placement_baseline(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            a['prebaseline_acid_checks']=[dict(timestamp=100.05,detections=[DETECTION])]
            m.step(snap(100.1))
            image=scene(acid=True)
            m._acid_observation(100.3,a['id'],[DETECTION],(image,QUAD,True,True,False))
            m._verify_application(a,image,QUAD,100.34,True,True)
            self.assertEqual(100.3,a['acid_onset'])
            self.assertFalse(any(e.get('reason')=='BASELINE_CAPTURED_AFTER_FIRST_CANDIDATE'
                                 for e in a.get('acid_candidate_events',[])))

    def test_stale_jewel_detection_cannot_start_placement_baseline(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            current=snap(101)
            current['tracks'][0]['det_seen']=100.5
            m.step(current)
            self.assertIsNone(a['baseline_at'])

    def test_provisional_candidate_placement_captures_baseline_without_claiming_identity(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1))
            a=m.active
            a['jewel_id']=None
            a['recording_trigger']=dict(candidate_ids=[1],provisional=True,source='STONE_AREA_PINCH')
            m.step(snap(100.1))
            self.assertEqual('JEWEL_PLACED_IN_TEST_BED',a['baseline_source'])
            self.assertIsNone(a['jewel_id'])
            self.assertIsNone(a['returned'])
            self.assertEqual('PROVISIONAL',m._record(a)['recording_assignment'])

    def test_local_interference_is_logged_without_vetoing_post_rub_capture(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            a['live_rub']=True;a['last_rub']=100
            a['contact_region']=contact_region(contact(100),QUAD)
            with patch('engines.purity.workflow.baseline_roi_interference',return_value='CLOTH_OR_GLARE'):
                m.step(snap(100.8,held=1))
            self.assertEqual(100.8,a['baseline_at'])
            self.assertEqual('CLOTH_OR_GLARE',a['baseline_region_status'])
            self.assertTrue(any(kind=='clearance_advisory' and data.get('region_reason')=='CLOTH_OR_GLARE'
                                for kind,data in events))

    def test_mask_captured_after_acid_candidate_cannot_replace_baseline(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            m._baseline(a,scene(),100.2,QUAD)
            a['acid_candidate_events']=[dict(status='CANDIDATE_RECORDING',candidate_at=100.3,timestamp=100.3)]
            a['mask']=RubReference(Photos(),scene(),99,QUAD).after(Photos(),scene(mark=True),100.4,QUAD)
            m.step(snap(100.8,held=1,frame=scene(mark=True)))
            self.assertEqual(100.2,a['baseline_at'])
            self.assertEqual('POST_RUB_COMPARISON',a['baseline_source'])

    def test_verified_mark_frame_seeds_baseline_without_another_capture(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            source=scene(mark=True)
            a['contact_region']=contact_region(contact(100),QUAD)
            a['mask_candidate']=RubReference(Photos(),scene(),99,QUAD).after(Photos(),source,100.2,QUAD)
            for ts in (100.2,100.28,100.36):m._confirm_candidate(a,source,ts,QUAD)
            self.assertIsNotNone(a['mask'])
            m.step(snap(100.4,held=1,frame=source))
            self.assertEqual('VERIFIED_RUBBED_MASK_FRAME',a['baseline_source'])
            self.assertEqual(100.2,a['baseline_at'])
            np.testing.assert_array_equal(source,a['return_frame'])
            from acid_freshness import normalized
            np.testing.assert_array_equal(normalized(source,QUAD),a['fresh_gate'].baseline)

    def outside_hand(self, **kwargs):
        frame=scene(**kwargs)
        skin=cv2.cvtColor(np.uint8([[[10,180,180]]]),cv2.COLOR_HSV2BGR)[0,0]
        frame[70:90,90:210]=skin
        return frame

    def test_whole_stone_coverage_and_box_overlap_are_advisory_for_acid(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            a['live_rub']=True;a['last_rub']=100
            a['contact_region']=contact_region(contact(100),QUAD)
            current=snap(100.8,held=1,frame=self.outside_hand())
            current['tracks'][0]['box']=STONE  # broad jewellery box is advisory too
            m.step(current)
            self.assertTrue(m.pose.uncertain)
            self.assertEqual(100.8,a['baseline_at'])
            self.assertTrue(any(k=='clearance_advisory' for k,_ in events))
            image=self.outside_hand(acid=True)
            m._acid_observation(101,a['id'],[DETECTION],(image,QUAD,False,True,False))
            m._verify_application(a,image,QUAD,101.04,False,True)
            self.assertEqual(101,a['acid_onset'])

    def test_mask_detection_and_verification_ignore_hand_outside_mark(self):
        models=Photos();frame=self.outside_hand(mark=True)
        reference=RubReference(models,scene(),99,QUAD)
        candidate=reference.after(models,frame,100,QUAD)
        self.assertEqual('NEW_RUB',candidate['status'])
        region=contact_region(contact(100),QUAD)
        for ts in (100,100.08,100.16):
            state,_=validate_mark(candidate,frame,QUAD,ts,region)
        self.assertEqual('VERIFIED',state)

    def test_recorded_backup_cannot_promote_mark_from_unverified_old_alignment(self):
        from backup_recovery import recover
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            a.update(finished=101,reference_alignment_uncertain=True,performance_timings={})
            m.backup_worker=Mock()
            m.backup_worker.analyze.return_value=dict(status='VERIFIED',timings={},
                rub=dict(mask='unverified-old-reference.png'))
            recover(m,a,{})
            self.assertIsNone(a['mask'])
            self.assertEqual('REVIEW_REQUIRED',a['backup']['status'])
            self.assertNotIn('rub',a['backup'])
            self.assertIn('unverified_rub',a['backup'])

    def test_detector_runs_before_baseline_and_records_unverified_candidates(self):
        with monitor_case() as (m,events):
            m.detector=Detector()
            m.step(snap(100,held=1))
            a=m.active
            self.assertIsNone(a['baseline_at'])
            self.assertIsNotNone(m.acid_job)
            m.acid_job.result(timeout=5)
            m._acid_observation(100.1,a['id'],[DETECTION],(scene(acid=True),QUAD,True,True,False))
            self.assertIsNone(a['acid_onset'])
            self.assertEqual('MODEL_CANDIDATE_WAITING_FOR_BASELINE',a['prebaseline_acid_checks'][-1]['status'])
            self.assertTrue(any(kind=='acid_detection' and data.get('waiting_for_baseline') for kind,data in events))

    def test_pending_mask_and_unverified_return_do_not_delay_baseline(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            a['live_rub']=True;a['last_rub']=100
            m.mask_job=(Future(),a['id'])
            m.step(snap(100.8,held=1))
            self.assertEqual(100.8,a['baseline_at'])
            self.assertIsNone(a['returned'])
            self.assertIsNone(a['mask'])
            self.assertTrue(any(kind=='mask_decision' and data.get('pending') for kind,data in events))
            m.mask_job=None

    def test_stone_recovery_does_not_reuse_old_mark_alignment_or_stop_detector(self):
        with monitor_case() as (m,events):
            m.step(snap(100,held=1));a=m.active
            m.pose.quad=QUAD+[100,0]
            a['live_rub']=True;a['last_rub']=100
            m.detector=Detector()
            for ts in (100.2,100.3,100.41,100.8):
                m.step(snap(ts,held=1))
                if m.acid_job is not None:
                    m.acid_job.result(timeout=5)
            self.assertEqual(1,len(a['stone_pose_recoveries']))
            self.assertTrue(a['reference_alignment_uncertain'])
            self.assertIsNotNone(a['baseline_at'])
            self.assertIsNone(a['acid_onset'])
            self.assertIsNone(a['mask'])
            self.assertTrue(any(kind=='status' and 'recovered automatically' in data['message'] for kind,data in events))


if __name__=='__main__':unittest.main()
