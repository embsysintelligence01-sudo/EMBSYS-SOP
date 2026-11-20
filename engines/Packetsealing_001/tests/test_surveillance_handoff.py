"""Unknown identities, independent outcomes and recording survive the handoff."""
from pathlib import Path
import copy
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import packet_app
from standalone.app import PacketWorker
from standalone.flow import PackingFlow, Pickup
from standalone.evidence import export_session, ffmpeg_path
import test_evidence as fixtures


class SurveillanceExportTests(unittest.TestCase):
    def test_sealing_never_promotes_missing_or_invalid_purity(self):
        for status, prediction in [('INCOMPLETE',None),('REVIEW_REQUIRED',{'prediction':'invalid','confidence':.99})]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp)
                work,recorder,flow=fixtures.EvidenceTests().fixture(directory)
                testing=dict(directory=str(directory/'purity'), evidence_status=status, analysis_status='FINISHED',
                    jewels=[dict(id=1,rubbed=False,acid_done=False,prediction=prediction,
                        evidence_status=status, test_status=status,identity_status='UNCERTAIN')])
                before=copy.deepcopy(testing)
                flow.testing_session=testing
                flow.identity_statuses={1:'UNCERTAIN'}
                result=export_session(directory,work,recorder,flow,102.9,'Complete',ffmpeg_path())
                self.assertTrue(result['sealed'])
                self.assertEqual(status,result['evidence_statuses']['purity_testing'])
                self.assertEqual('REVIEW_REQUIRED',result['evidence_statuses']['packet_sealing'])
                self.assertEqual(before,testing)
                self.assertEqual(prediction,result['testing_session']['jewels'][0]['prediction'])
                self.assertTrue(result['raw_retained'])

    def test_zero_assigned_ids_records_and_exports_without_starting_a_new_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            roi=np.array([(0,0),(159,0),(159,119),(0,119)],np.int32).reshape(-1,1,2)
            tracker=SimpleNamespace(mode='LOCKED',tracks=[],id_model=SimpleNamespace(ready=False),
                identity_error='',roi=roi,raw_detections=[],bench_anchors={},detections=[],
                ensure_roi=lambda:None, process=lambda *a,**k:[], pickup_event=lambda *a,**k:Pickup(), close=lambda:None)
            worker=PacketWorker(0,roi.reshape(-1,2).tolist(),handoff=dict(
                worker=SimpleNamespace(camera=None),testing=dict(directory=str(root/'purity'),
                    evidence_status='INCOMPLETE',identity_status='UNASSIGNED',jewels=[])))
            frames=[];exports=[];errors=[];states=[]
            class Grabber:
                on_frame=None
                problem=''
                def start(self):pass
                def latest(self,after):
                    t=time.time()
                    frame=np.full((120,160,3),180,np.uint8)
                    self.on_frame(frame,t)
                    frames.append(t)
                    if len(frames)>=4:worker.running=False
                    time.sleep(.04)
                    return frame,t
                def stop(self):pass
                def join(self,timeout):pass
            models=Mock(inspect=Mock(return_value=(None,'unknown',True,None,None)))
            worker.exported.connect(exports.append);worker.failed.connect(errors.append);worker.state.connect(states.append)
            with patch('standalone.app.ROOT',root),patch('standalone.app.PacketModels',return_value=models),\
                 patch('standalone.app.TrackingBridge',return_value=tracker),patch('standalone.app.make_grabber',return_value=Grabber()):
                worker.run()
            self.assertEqual([],errors)
            self.assertEqual(1,len(exports))
            result=exports[0]
            self.assertEqual([],result['jewels'])
            self.assertTrue(result['unassigned_identity'])
            self.assertTrue(result['raw_retained'])
            self.assertEqual('INCOMPLETE',result['evidence_statuses']['purity_testing'])
            self.assertTrue(any(s['identity_status']=='UNASSIGNED' for s in states))
            self.assertTrue((Path(result['directory'])/'.working').is_dir())
            import cv2
            raw=cv2.VideoCapture(str(Path(result['directory'])/'.working/session_raw.avi'))
            try:
                ok,recorded=raw.read()
                self.assertTrue(ok, 'Unassigned surveillance must retain a decodable recording')
                self.assertEqual((120,160),recorded.shape[:2])
            finally:
                raw.release()
            self.assertNotIn('purity',Path(result['directory']).parts)
            self.assertEqual(roi.tolist(),tracker.roi.tolist())

    def test_testing_snapshot_waits_for_background_publication_lock(self):
        import threading
        attempted=threading.Event()
        lock=threading.Lock()
        class PublicationLock:
            def __enter__(self):
                attempted.set()
                lock.acquire()
            def __exit__(self,*args):lock.release()
        testing={'evidence_status':'INCOMPLETE','jewels':[]}
        worker=PacketWorker(0,[],handoff={'testing':testing,'testing_lock':PublicationLock()})
        snapshots=[]
        lock.acquire()
        thread=threading.Thread(target=lambda:snapshots.append(worker._testing_snapshot()))
        thread.start()
        self.assertTrue(attempted.wait(2))
        self.assertEqual([],snapshots)
        testing.update(evidence_status='REVIEW_REQUIRED',jewels=[{'id':3}])
        lock.release()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual('REVIEW_REQUIRED',snapshots[0]['evidence_status'])
        self.assertEqual(3,snapshots[0]['jewels'][0]['id'])
        testing['jewels'][0]['id']=9
        self.assertEqual(3,snapshots[0]['jewels'][0]['id'])

    def test_handoff_cannot_reset_assignment_or_renumber(self):
        worker=PacketWorker(0,[],handoff={'worker':object(),'testing':{'jewels':[]}})
        worker.tracker=Mock()
        worker.flow=PackingFlow([1,3,8])
        worker._restart_assignment()
        worker.tracker.reset_assignment.assert_not_called()
        self.assertEqual([1,3,8],list(worker.flow.jewels))
