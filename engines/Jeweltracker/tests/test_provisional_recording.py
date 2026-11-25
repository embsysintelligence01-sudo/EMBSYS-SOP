"""Candidate recordings remain useful without asserting a confirmed jewel identity."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import jewel_tracker_5 as app
from continuous_testing import recording_assignment
from audit_report import recording_label, audit_outcome, event_rows, render_report, jewel_summary_groups
from reportlab.pdfgen.canvas import Canvas
from packet_stage import testing_snapshot
from test_sop_revision import monitor_case, snap


def trigger(ids):
    return dict(candidate_ids=ids, provisional=True, jewel_id=None,
                source='STONE_AREA_PINCH', trigger_hands=['Left'])


class RecordingAssignmentTests(unittest.TestCase):
    def test_confirmed_candidate_continues_same_cycle_and_preserves_evidence(self):
        with monitor_case() as (m,events):
            m._new_cycle(None,snap(100),100,recording_trigger=trigger([2]))
            active=m.active
            baseline=np.ones((12,12,3),np.uint8)
            active.update(baseline=baseline,baseline_at=100.5,live_rub=True,
                          rubbing_status='CONFIRMED_LIVE')
            cue=dict(trigger([2]),jewel_id=2,provisional=False,confirmed_at=101.)
            with patch.object(m,'_box_cue',return_value=cue), patch.object(m,'_finish') as finish:
                m._select_box_cycle(snap(101))
            finish.assert_not_called()
            self.assertIs(active,m.active)
            self.assertEqual(2,active['jewel_id'])
            self.assertEqual(100,active['started'])
            self.assertIs(baseline,active['baseline'])
            self.assertTrue(active['live_rub'])
            self.assertEqual([2],active['recording_trigger']['candidate_ids'])
            self.assertEqual(cue,m._record(active)['identity_confirmation'])
            self.assertEqual('CONFIRMED',m.cycle_assignments[active['id']]['recording_assignment'])
            self.assertEqual(1,sum(kind=='begin' for kind,_ in events))
            self.assertEqual(1,sum(kind=='identity_confirmed' for kind,_ in events))

    def test_different_or_ambiguous_candidate_still_requires_a_new_cycle(self):
        for ids in ([1],[1,2],[]):
            with self.subTest(ids=ids),monitor_case() as (m,events):
                m._new_cycle(None,snap(100),100,recording_trigger=trigger(ids))
                cue=dict(trigger([2]),jewel_id=2,provisional=False,confirmed_at=101.)
                with patch.object(m,'_box_cue',return_value=cue),patch.object(m,'_finish') as finish,patch.object(m,'_new_cycle'):
                    m._select_box_cycle(dict(snap(101),box_settings={}))
                finish.assert_called_once()

    def test_pdf_groups_candidate_history_without_discarding_attempts(self):
        candidate=dict(jewel_id=None,recording_candidate_id=2,recording_assignment='PROVISIONAL',
                       started=100,finished=101)
        confirmed=dict(jewel_id=2,started=101,finished=120)
        unrelated=dict(jewel_id=None,started=99)
        records=[candidate,confirmed,unrelated]
        groups=jewel_summary_groups(records)
        self.assertEqual([(confirmed,[candidate]),(unrelated,[])],groups)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'report.pdf';canvas=Mock(wraps=Canvas(str(output)))
            with patch('audit_report.canvas.Canvas',return_value=canvas):
                render_report(output,root,[dict(id=2)],records,[],lambda p:'Unavailable')
            strings=[call.args[2] for call in canvas.drawString.call_args_list]
            self.assertEqual(1,strings.count('J2'))
            self.assertNotIn('J2?',strings)
            self.assertTrue(any('other attempts: 1 (1 provisional)' in s for s in strings))
            self.assertTrue(any('J2 (provisional recording) | Rubbing' in s for s in strings))
            self.assertEqual(3,len(records))
            self.assertFalse(any('Unassigned |' in s for s in strings))

    def test_report_labels_candidate_and_links_original_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);clip=root/'pickup.avi';clip.touch()
            record=dict(jewel_id=None,recording_candidate_id=1,recording_assignment='PROVISIONAL',
                        rubbing_clip=dict(path=str(clip)),issues=[])
            output=root/'report.pdf';canvas=Mock(wraps=Canvas(str(output)))
            with patch('audit_report.canvas.Canvas',return_value=canvas):
                render_report(output,root,[dict(id=1)],[record],[],lambda p:'Unavailable')
            strings=[call.args[2] for call in canvas.drawString.call_args_list]
            self.assertTrue(any('PROVISIONAL RECORDING' in text for text in strings))
            self.assertTrue(any('J1 (provisional recording)' in text for text in strings))
            self.assertIn(clip.as_uri(),[call.args[0] for call in canvas.linkURL.call_args_list])
            self.assertIsNone(record['jewel_id'])

    def test_only_one_provisional_candidate_gets_a_display_link(self):
        for ids, expected in (([1],1),([2],2),([],None),([1,2],None)):
            with self.subTest(ids=ids):
                assignment=recording_assignment(dict(jewel_id=None,recording_trigger=trigger(ids)))
                self.assertEqual(expected,assignment['recording_candidate_id'])
                self.assertEqual('PROVISIONAL' if expected else 'UNASSIGNED',assignment['recording_assignment'])
        self.assertEqual('CONFIRMED',recording_assignment(dict(jewel_id=2,recording_trigger=trigger([1])))['recording_assignment'])

    def test_begin_and_manifest_keep_the_candidate_separate_from_identity(self):
        with monitor_case() as (m,events):
            m._new_cycle(None,snap(100),100,recording_trigger=trigger([1]))
            begin=next(data for kind,data in events if kind=='begin')
            self.assertIsNone(begin['jewel_id'])
            self.assertEqual(1,begin['recording_candidate_id'])
            self.assertEqual('PROVISIONAL',begin['recording_assignment'])
            record=m._record(m.active)
            record['visibility']=[]
            self.assertIsNone(record['jewel_id'])
            self.assertEqual(1,record['recording_candidate_id'])
            self.assertTrue(record['recording_trigger']['provisional'])
            self.assertEqual('J1 (provisional recording)',recording_label(record))
            self.assertEqual('REVIEW REQUIRED',audit_outcome(record))
            self.assertTrue(any(row[0]=='Recording association' for row in event_rows(record)))


class ProvisionalRecordingUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        with patch.object(QtCore.QTimer,'singleShot'):
            self.window=app.MainWindow()
        self.worker=app.Worker(0,[(0,0),(399,0),(399,299),(0,299)],continuous=True)
        self.window.worker=self.worker
        for jid in (1,2):
            box=np.array([30.,30.,60.,60.])+jid*70
            self.worker.tracks.append(app.Track(jid,app.Det(box,(box[:2]+box[2:])/2,900,.9),0))
        self.worker.jewel_table.connect(self.window.show_table)
        self.worker.workflow.connect(self.window.continuous_update)
        self.assignment=recording_assignment(dict(jewel_id=None,recording_trigger=trigger([1])))

    def tearDown(self):
        self.window.worker=None
        self.window.close()

    def begin(self,token='candidate'):
        self.worker.continuous_event('begin',jewel_id=None,token=token,**self.assignment)

    def result(self,token='candidate'):
        return dict(jewel_id=None,cycle_id=token,revision=self.worker.flow_revision,
                    recording_trigger=trigger([1]),capture_complete=True,
                    rubbing_status='CONFIRMED_LIVE',close_reason='REACTION_COMPLETE',
                    prediction=dict(prediction='gold_22k',confidence=.95),record_path='/tmp/candidate/manifest.json')

    def test_candidate_row_shows_recording_but_has_no_confirmed_test_owner(self):
        self.begin()
        tr=self.worker.tracks[0]
        self.assertIn('PROVISIONAL',self.window.table.item(0,0).text())
        self.assertIn('Recording',self.window.table.item(0,4).text())
        self.assertIsNone(tr.acid_result)
        self.assertIsNone(self.worker.continuous_pending_id)
        self.assertFalse(tr.done)

    def test_identity_confirmation_keeps_cycle_and_accepts_its_final_result(self):
        self.begin()
        token=self.worker.continuous_cycle_id
        photo=np.zeros((10,10,3),np.uint8)
        self.worker.continuous_event('contact',jewel_id=None,token=token,photo=photo,
                                     timestamp=1.,confirmed=True,**self.assignment)
        self.worker.continuous_event('identity_confirmed',jewel_id=1,token=token,
                                     recording_assignment='CONFIRMED',recording_candidate_id=None)
        tr=self.worker.tracks[0]
        self.assertEqual(token,self.worker.continuous_cycle_id)
        self.assertEqual(1,self.worker.continuous_pending_id)
        self.assertIs(photo,tr.rub_img)
        self.assertIsNone(tr.provisional_recording)
        self.assertNotIn('PROVISIONAL',self.window.table.item(0,0).text())
        record=dict(self.result(token),jewel_id=1)
        self.worker.continuous_event('result',record=record)
        self.assertIs(record,tr.acid_result)
        self.assertTrue(tr.done)

    def test_candidate_result_and_clip_link_do_not_confirm_identity_or_handoff_prediction(self):
        self.begin()
        self.worker.continuous_event('captured',jewel_id=None,token='candidate',capture_complete=True,**self.assignment)
        self.worker.continuous_event('result',record=self.result())
        tr=self.worker.tracks[0]
        self.assertIsNone(tr.acid_result)
        self.assertFalse(tr.acid_done)
        self.assertFalse(tr.done)
        self.assertFalse(getattr(tr,'test_captured',False))
        self.assertIn('Candidate recording',self.window.table.item(0,5).text())
        self.assertIn('PROVISIONAL',self.window.table.item(0,0).text())
        self.assertIsNone(testing_snapshot(self.worker,'/tmp/session')['jewels'][0]['prediction'])
        with patch.object(QtGui.QDesktopServices,'openUrl') as opened:
            self.window.open_table_recording(0,5)
            self.assertEqual('/tmp/candidate',opened.call_args.args[0].toLocalFile())

    def test_old_candidate_result_cannot_replace_a_newer_cycle(self):
        self.begin('old')
        self.worker.continuous_event('captured',jewel_id=None,token='old',**self.assignment)
        self.begin('new')
        self.worker.continuous_event('result',record=self.result('old'))
        self.assertEqual('new',self.worker.tracks[0].provisional_recording['cycle_id'])
        self.assertIsNone(self.worker.tracks[0].provisional_recording['prediction'])

    def test_multiple_candidates_leave_all_jewel_rows_unassigned(self):
        assignment=recording_assignment(dict(jewel_id=None,recording_trigger=trigger([1,2])))
        self.worker.continuous_event('begin',jewel_id=None,token='ambiguous',**assignment)
        self.assertTrue(all(getattr(tr,'provisional_recording',None) is None for tr in self.worker.tracks))

    def test_confirmed_result_takes_precedence_over_candidate_display(self):
        self.begin()
        self.worker.continuous_event('captured',jewel_id=None,token='candidate',**self.assignment)
        self.worker.continuous_event('begin',jewel_id=1,token='confirmed')
        record=dict(self.result('confirmed'),jewel_id=1)
        record.pop('recording_trigger')
        self.worker.continuous_event('result',record=record)
        self.assertIs(record,self.worker.tracks[0].acid_result)
        self.assertNotIn('PROVISIONAL',self.window.table.item(0,0).text())


if __name__=='__main__':unittest.main()
