"""Physical progress never asserts successful evidence or repairs an uncertain ID."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from concurrent.futures import Future
import numpy as np
from PyQt5 import QtCore, QtWidgets
import jewel_tracker_5 as original
from packet_stage import testing_snapshot, ready_for_packet, camera_released
from surveillance import purity_evidence


class SurveillanceHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from packet_stage import packet_module
        cls.packet = packet_module(original)

    def track(self, jid=1, **changes):
        values = dict(id=jid, rubbed=False, acid_done=False, acid_result=None,
            pos=np.array([40., 40.]), box=np.array([20.,20.,60.,60.]), det_seen=time.time(), state='OK')
        values.update(changes)
        return SimpleNamespace(**values)

    def test_missing_rubbing_mask_acid_and_invalid_classification_never_gate(self):
        for changes in ({}, {'rubbed':True}, {'acid_result':{'prediction':None, 'close_reason':'ACID_NOT_FOUND'}},
            {'rubbed':True, 'acid_done':True, 'acid_result':{'prediction':{'prediction':'invalid', 'confidence':.99}}},
            {'acid_result':{'backup':{'status':'PROCESSING', 'timed_out':True}}}):
            with self.subTest(changes=changes):
                worker=SimpleNamespace(isRunning=lambda:True, tracks=[self.track(**changes)],
                    mode='ARMING', id_model=SimpleNamespace(ready=False), continuous_cycle_id='active', acid_token='active')
                self.assertTrue(ready_for_packet(worker))
                snapshot=testing_snapshot(worker,'/tmp/surveillance')
                self.assertIn(snapshot['jewels'][0]['test_status'], ('INCOMPLETE','REVIEW_REQUIRED'))
                self.assertNotEqual('VERIFIED',snapshot['evidence_status'])
        self.assertTrue(ready_for_packet(SimpleNamespace(tracks=[], isRunning=lambda:False)))
        self.assertFalse(ready_for_packet(None))

    def test_prediction_without_verified_rubbing_and_mask_stays_review_required(self):
        record={'prediction':{'prediction':'gold_22k','confidence':.99},'sop_review_status':'VERIFIED'}
        evidence=purity_evidence(self.track(rubbed=False, acid_done=True, acid_result=record))
        self.assertEqual('REVIEW_REQUIRED',evidence['evidence_status'])
        self.assertEqual('UNCONFIRMED',evidence['mask_status'])
        self.assertEqual('gold_22k',evidence['prediction']['prediction'])

    def test_pending_recovery_never_reuses_a_previous_verified_label(self):
        record={'prediction':{'prediction':'gold_22k','confidence':.99},
            'sop_review_status':'VERIFIED','mask_evidence_status':'VERIFIED',
            'acid_target_status':'ACID_OVERLAPS_NEW_RUB','reaction_coverage_status':'COMPLETE',
            'backup':{'status':'PROCESSING'}}
        tr=self.track(rubbed=True,acid_done=True,acid_result=record)
        self.assertEqual('REVIEW_REQUIRED',purity_evidence(tr)['evidence_status'])
        record['backup']['status']='VERIFIED'
        self.assertEqual('VERIFIED',purity_evidence(tr)['evidence_status'])

    def test_unknown_and_stale_id_remain_explicit_and_snapshot_is_independent(self):
        tr=self.track(3, det_seen=time.time()-20, acid_result={'prediction':{'prediction':'invalid','confidence':.8}})
        worker=SimpleNamespace(tracks=[tr,self.track(None)])
        snapshot=testing_snapshot(worker,'/tmp/surveillance')
        self.assertEqual([3,None],[j['id'] for j in snapshot['jewels']])
        self.assertEqual(['UNCERTAIN','UNASSIGNED'],[j['identity_status'] for j in snapshot['jewels']])
        tr.acid_result['prediction']['prediction']='gold_22k'
        self.assertEqual('invalid', snapshot['jewels'][0]['prediction']['prediction'])

    def test_camera_release_event_cannot_override_live_acquisition(self):
        signal=threading.Event();signal.set()
        grabber=SimpleNamespace(is_alive=lambda:True)
        worker=SimpleNamespace(isRunning=lambda:False, packet_capture_stopped=signal, grabber=grabber)
        self.assertFalse(camera_released(worker))
        grabber.is_alive=lambda:False
        self.assertTrue(camera_released(worker))

    def test_bridge_keeps_nonsequential_ids_and_quarantines_uncertainty(self):
        from standalone.tracking import TrackingBridge
        tr=self.track(3, identity_uncertain=True)
        previous=SimpleNamespace(tracks=[self.track(1),tr,self.track(None)], model=Mock(),
            id_model=SimpleNamespace(ready=False), identity_extractor=object(),frame_wh=(640,480))
        bridge=TrackingBridge([(0,0),(640,0),(640,480)], previous)
        self.assertEqual([1,3],[t.id for t in bridge.tracks])
        self.assertEqual({3},bridge.uncertain_ids)
        self.assertNotIn(3,bridge.bench_anchors)
        self.assertFalse(bridge.continuous_identity[3])
        matched=[]
        with patch.object(bridge, 'detect', return_value=([], [])), patch.object(bridge, 'update_tracks',
                side_effect=lambda *args:matched.extend(tr.id for tr in bridge.tracks)):
            bridge.process(np.zeros((480,640,3),np.uint8),time.time())
        self.assertEqual([1],matched)
        bridge.reset_assignment()
        self.assertEqual([1,3],[t.id for t in bridge.tracks])
        bridge.tracks[0].pos[0]=400
        self.assertEqual(40,previous.tracks[0].pos[0])
        bridge.close()

    def test_camera_barrier_identity_snapshot_survives_model_loading_delay(self):
        from standalone.tracking import TrackingBridge
        previous=SimpleNamespace(tracks=[self.track(2,det_seen=time.time()-20)],model=Mock(),
            id_model=SimpleNamespace(ready=True),frame_wh=(640,480),packet_identity_statuses={2:'ASSIGNED'})
        bridge=TrackingBridge([(0,0),(640,0),(640,480)],previous)
        self.assertEqual([2],list(bridge.bench_anchors))
        self.assertTrue(bridge.continuous_identity[2])
        bridge.close()

    def test_manual_background_result_arrays_are_retained_without_verifying_them(self):
        import json
        import tempfile
        from surveillance import retain_auxiliary_result
        with tempfile.TemporaryDirectory() as tmp:
            image=np.arange(12,dtype=np.uint8).reshape(3,4)
            path=Path(retain_auxiliary_result({'mask':image,'status':'NEW_RUB'},tmp))
            result=json.loads(path.read_text())
            self.assertEqual('REVIEW_REQUIRED',result['evidence_status'])
            np.testing.assert_array_equal(image,np.load(path.parent/result['result']['mask']['array']))

    def test_missing_purity_gold_model_allows_unassigned_packet_surveillance(self):
        from standalone.tracking import TrackingBridge
        previous=SimpleNamespace(tracks=[],model=None,id_model=None,frame_wh=None)
        with patch('standalone.tracking.OpenVINOYolo',side_effect=RuntimeError('Unavailable')):
            bridge=TrackingBridge([(0,0),(640,0),(640,480)],previous)
        self.assertEqual('LOCKED',bridge.mode)
        self.assertTrue(bridge.unavailable_model)
        self.assertFalse(bridge.id_model.ready)
        self.assertEqual([],bridge.process(np.zeros((480,640,3),np.uint8),time.time()))
        self.assertFalse(bridge.pickup_event(time.time()).active)
        bridge.reset_assignment()
        self.assertEqual([],bridge.tracks)
        bridge.close()

    def test_active_capture_and_rub_future_drain_after_camera_handoff(self):
        with patch.object(QtCore.QTimer,'singleShot'):
            owner=original.MainWindow()
        busy=[True]; future=Future()
        previous=SimpleNamespace(isRunning=lambda:busy[0], mode='ARMING',tracks=[self.track(1)],
            identity_slic=True, packet_capture_stopped=threading.Event(),source=0,model=Mock(),
            roi=np.array([(0,0),(100,0),(100,100)]),continuous_monitor=SimpleNamespace(
                directory=Path('/tmp/test-evidence'),active={'jewel_id':1}),
            stop=Mock(), stone=SimpleNamespace(model=object()), rubber=object())
        owner.worker=previous
        owner.rub_job={'future':future,'kind':'after'}
        owner.acid_thread=SimpleNamespace(isRunning=lambda:False,cancel=Mock())
        stage=owner.packet_stage
        with patch('packet_stage.packet_module',return_value=self.packet):stage.begin()
        previous.stop.assert_called_once()
        stage.tick();self.assertIsNone(stage.window)
        previous.packet_capture_stopped.set()
        with patch.object(self.packet.MainWindow,'connect_camera'):
            with patch.object(owner.acid_models,'close',wraps=owner.acid_models.close) as close:
                stage.tick()
                self.assertIsNotNone(stage.window)
                self.assertFalse(stage.testing_ready.is_set())
                busy[0]=False
                stage.tick();close.assert_not_called()
                future.set_exception(RuntimeError('Missing rubbing mask'))
                stage.tick();close.assert_not_called()
                owner.acid_thread=None  # Its finished slot has consumed queued results.
                stage.tick();close.assert_called_once()
        self.assertTrue(stage.testing_ready.is_set())
        self.assertEqual('INCOMPLETE',stage.testing['evidence_status'])
        self.assertEqual('REVIEW_REQUIRED',stage.testing['pending_rub_job_status'])
        owner.close()
