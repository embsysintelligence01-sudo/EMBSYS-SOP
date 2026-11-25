"""Handoff the completed testing session to packet sealing and number OCR."""
from shared.paths import PURITY_DATA, SEALING_DATA
from model_registry import resolve_model
import gc
import importlib
import os
from pathlib import Path
import sys
import time
import threading
import copy
from contextlib import nullcontext
from surveillance import purity_evidence, aggregate_purity, retain_auxiliary_result

from PyQt5 import QtCore


def packet_wait_reason(worker):
    # Evidence, gallery readiness and track count never authorize physical progress.
    if worker is None:
        return 'Connect a surveillance session before transferring its camera.'
    return ''


def camera_released(worker):
    signal = getattr(worker, 'packet_capture_stopped', None)
    released = (signal is not None and signal.is_set()) or not worker.isRunning()
    return released and not any(owner is not None and owner.is_alive() for owner in
        (getattr(worker, 'grabber', None), getattr(worker, 'stone', None),
         getattr(worker, 'id_training_thread', None)) if hasattr(owner, 'is_alive'))


def ready_for_packet(worker):
    return not packet_wait_reason(worker)


def testing_snapshot(worker, directory, final=False):
    monitor = getattr(worker, 'continuous_monitor', None)
    with getattr(worker, 'lock', nullcontext()):
        records = getattr(monitor, 'tests', ()) if final else ()
        if not isinstance(records, (list, tuple)):
            records = ()
        jewels = []
        for tr in getattr(worker, 'tracks', ()):
            snapshot_track = copy.copy(tr)
            # Final report data comes from committed analysis, even when Qt
            # has not yet delivered the corresponding UI result signal.
            named = [r for r in records if r.get('jewel_id') == tr.id]
            candidates = [r for r in records if r.get('jewel_id') is None
                          and r.get('recording_assignment') == 'PROVISIONAL'
                          and r.get('recording_candidate_id') == tr.id]
            if named:
                record = max(named, key=lambda r: r.get('started') or 0)
                snapshot_track.acid_result = record
                snapshot_track.acid_done = bool(record.get('capture_complete'))
                snapshot_track.rubbed = record.get('rubbing_status') in ('CONFIRMED_LIVE','RUBBING_RECOVERED_FROM_CLIP')
            if candidates:
                snapshot_track.provisional_recording = max(candidates, key=lambda r: r.get('started') or 0)
            jewels.append(purity_evidence(snapshot_track))
        if getattr(worker, 'mode', 'LOCKED') != 'LOCKED':
            for jewel in jewels:
                if jewel['identity_status'] == 'ASSIGNED':
                    jewel['identity_status'] = 'UNCERTAIN'
    for jewel in jewels:
        if final and not jewel['prediction'] and not jewel.get('recording_prediction') and not jewel['close_reason']:
            jewel['close_reason'] = 'Analysis finished without a material result.'
    return dict(directory=str(directory), jewels=jewels,
        evidence_status=aggregate_purity(jewels),
        identity_status='UNASSIGNED' if not jewels else
            'REVIEW_REQUIRED' if any(j['identity_status'] != 'ASSIGNED' for j in jewels) else 'ASSIGNED',
        analysis_status='FINISHED' if final else 'PROCESSING',
        analysis_errors=copy.deepcopy(getattr(monitor, 'finalization_errors', [])),
        workflow_stage='PACKET_SEALING')


def packet_module(original):
    root = Path(os.environ.get('JEWEL_PACKET_DIR', Path(original.SCRIPT_DIR).parent / 'Packetsealing_001'))
    required = ('bagnewmodel.onnx', 'fp_filter_resnet18_bag.onnx', 'stripmodel.onnx',
                'hsv_fp_filter_strip.pt', 'handseg model.onnx', 'pcnumber.onnx')
    missing = [name for name in required if not resolve_model(root / 'model onnx' / name).is_file()]
    if missing:
        raise RuntimeError('Packet model files missing in '+str(root)+': '+', '.join(missing))
    if not (Path(__file__).resolve().parents[1] / 'sealing' / 'workflow.py').is_file():
        raise RuntimeError('Packet application missing: '+str(root))
    # Running this script as __main__ must not import a second copy of the tracker.
    sys.modules.setdefault('jewel_tracker_5', original)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module('standalone.app')
    module.ffmpeg_path()  # Fail before stopping testing if video export is unavailable.
    return module


class PacketStage(QtCore.QObject):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.window = None
        self.pending = None
        self.draining = None
        self.gold_owner = None
        self.testing = None
        self.testing_ready = threading.Event()
        self.testing_lock = threading.RLock()
        self.retained_auxiliary = []
        self.module = None
        self.reset_requested = False
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    @property
    def active(self):
        return self.pending is not None or self.window is not None or self.draining is not None

    def begin(self):
        owner = self.owner
        if self.window is not None:
            self.window.show()
            self.window.raise_()
            return
        if self.pending is not None or self.draining is not None or not ready_for_packet(owner.worker):
            return
        worker = owner.worker
        if owner.reset_timer.isActive():
            return  # Reset owns lifecycle until its recording has drained.
        try:
            self.module = packet_module(sys.modules[owner.__class__.__module__])
            if not worker.identity_slic:
                raise RuntimeError('Packet tracking needs SLIC. Start with run_tracker_slic.sh.')
        except Exception as exc:
            owner.add_log('Packet sealing unavailable: '+str(exc), 'alert')
            return
        # End live/manual acquisition; queued recorded-video/PDF work may drain.
        if owner.acid_thread is not None:
            owner.acid_thread.cancel()
        self.pending = worker
        self.gold_owner = worker
        worker.preserve_gold_model = True
        self.testing_ready.clear()
        owner.btn_packet.setEnabled(False)
        owner.rub_timer.stop()
        self.lock_controls()
        owner.lbl_status.setText('Finishing camera handoff; jewel analysis will continue in the background…')
        worker.stop()

    def lock_controls(self):
        owner = self.owner
        for control in (owner.btn_conn, owner.cam_mode, owner.btn_reset, owner.btn_roi,
                        owner.btn_stone, owner.btn_test, owner.btn_clear, owner.btn_af):
            control.setEnabled(False)

        for name in ('stone_grip_margin', 'pinch_distance', 'pinch_hand', 'box_gold_fallback'):
            control = getattr(owner, name, None)
            if control is not None:
                control.setEnabled(False)

    def tick(self):
        owner = self.owner
        if self.draining is not None and not self.draining.isRunning() and self.auxiliary_finished():
            self.finish_analysis()
        capture_stopped = self.pending is not None and camera_released(self.pending)
        if capture_stopped:
            previous, self.pending = self.pending, None
            self.draining = previous
            try:
                monitor = getattr(previous, 'continuous_monitor', None)
                directory = getattr(monitor, 'directory', None)
                if directory is None:
                    directory = PURITY_DATA / 'captures' / 'sessions' / ('handoff_' + __import__('uuid').uuid4().hex)
                    directory.mkdir(parents=True, exist_ok=True)
                self.testing = testing_snapshot(previous, directory)
                previous.packet_identity_statuses = {j['id']: j['identity_status'] for j in self.testing['jewels']}
                testing = self.testing
                self.window = self.module.MainWindow(handoff=dict(worker=previous, testing=testing, testing_ready=self.testing_ready, testing_lock=self.testing_lock))
                roi = getattr(previous, 'roi', None)
                self.window.roi = roi.tolist() if roi is not None else owner.zones.get('roi', [])
                self.window.roi_override = True
                self.window.source.setText(str(previous.source))
                self.window.setAttribute(QtCore.Qt.WA_QuitOnClose, False)
                self.window.session_exported.connect(self.exported)
                self.window.show()
                self.window.connect_camera()
                if not previous.isRunning() and self.auxiliary_finished():
                    self.finish_analysis()
                if self.draining is None:
                    owner.worker = None
                owner.lbl_status.setText('Packet sealing / OCR active. Pending jewel analysis continues; incomplete tests stay incomplete.')
                owner.phase_label.setText('PROCESS: Packet sealing → packet number OCR')
                owner.btn_reset.setEnabled(True)
            except Exception as exc:
                if self.window is not None and (self.window.worker is None or not self.window.worker.isRunning()):
                    self.window.close()
                    self.window.deleteLater()
                    self.window = None
                if self.draining is None:
                    owner.worker = None
                owner.set_running(False)
                owner.btn_conn.setEnabled(self.draining is None)
                owner.cam_mode.setEnabled(self.draining is None)
                owner.btn_af.setEnabled(self.draining is None)
                if self.draining is None:
                    owner.rub_timer.start(200)
                owner.add_log('Packet handoff failed; testing evidence retained: '+str(exc), 'alert')
        if self.reset_requested and self.window is not None:
            worker = self.window.worker
            if (worker is None or not worker.isRunning()) and self.draining is None:
                self.release_gold()
                self.window.close()
                self.window.deleteLater()
                self.window = None
                self.reset_requested = False
                owner.set_running(False)
                owner.btn_conn.setEnabled(True)
                owner.cam_mode.setEnabled(True)
                owner.btn_af.setEnabled(True)
                owner.rub_timer.start(200)
                owner.toggle_connect()
        reason = packet_wait_reason(owner.worker)
        owner.packet_status.setText(('Packet stage active; pending jewel analysis continues.' if self.draining is not None else
                                    'Packet stage active; jewel outcomes retained.') if self.window is not None else
                                   'Switching camera; background results are retained.' if self.pending is not None else
                                   reason or 'Surveillance handoff available. Missing/invalid evidence stays incomplete or requires review.')
        owner.btn_packet.setToolTip(reason or 'Continue surveillance regardless of evidence outcome; preserve identities and wait only for safe camera release.')
        owner.btn_packet.setText('Open packet sealing / OCR' if self.window is not None else 'Packet Sealing + OCR')
        owner.btn_packet.setEnabled(self.window is not None or (self.pending is None and self.draining is None
            and not owner.reset_timer.isActive() and ready_for_packet(owner.worker)))

    def auxiliary_finished(self):
        owner = self.owner
        thread = owner.acid_thread
        job = owner.rub_job
        # The Qt finished slot must also have consumed queued acid result signals.
        return thread is None and (job is None or job['future'].done())

    def finish_analysis(self):
        previous = self.draining
        if previous is None:
            return
        owner = self.owner
        if self.testing is not None:
            with self.testing_lock:
                identity_at_handoff = {j['id']: j['identity_status'] for j in self.testing['jewels']}
                final = testing_snapshot(previous, self.testing['directory'], final=True)
                for jewel in final['jewels']:
                    jewel['identity_status'] = identity_at_handoff.get(jewel['id'], 'UNASSIGNED')
                final['identity_status'] = self.testing['identity_status']
                self.testing.update(final)
                job = owner.rub_job
                if job is not None:
                    try:
                        value = job['future'].result()
                        self.retained_auxiliary.append(value)
                        self.testing['auxiliary_evidence'] = retain_auxiliary_result(value, self.testing['directory'])
                        self.retained_auxiliary.pop()
                        self.testing['pending_rub_job_status'] = 'FINISHED_UNCONFIRMED'
                    except Exception:
                        self.testing['pending_rub_job_status'] = 'REVIEW_REQUIRED'
                self.testing['auxiliary_evidence_requires_review'] = job is not None
                if self.testing['analysis_errors'] or job is not None:
                    if self.testing['evidence_status'] == 'VERIFIED':
                        self.testing['evidence_status'] = 'REVIEW_REQUIRED'
        previous.continuous_monitor = None
        if previous.stone is not None:
            previous.stone.model = None
        previous.stone = previous.rubber = None
        owner.rub_models.close()
        owner.acid_models.close()
        owner.acid_thread = owner.acid_owner = None
        owner.rub_cycle = owner.rub_cleanup = owner.rub_job = None
        owner.rub_cycle_key = owner.rub_display_result = None
        self.draining = None
        if owner.worker is previous:
            owner.worker = None
        self.testing_ready.set()
        if self.window is None and self.pending is None:
            self.release_gold()
            owner.set_running(False)
            owner.btn_conn.setEnabled(True)
            owner.cam_mode.setEnabled(True)
            owner.btn_af.setEnabled(True)
            owner.rub_timer.start(200)
        gc.collect()

    def release_gold(self):
        if self.gold_owner is not None:
            if self.gold_owner.model is not None:
                self.gold_owner.model.close()
            self.gold_owner = None

    def exported(self, result):
        owner = self.owner
        from PyQt5 import QtCore
        path = str(Path(result['directory']) / result['report'])
        url = QtCore.QUrl.fromLocalFile(path).toString()
        owner.report_label.setText(f'<a href="{url}">Jewel testing + packet sealing + OCR report</a>')
        number = result.get('packet_number', {})
        sealing = 'Sealed (inferred)' if result['sealed'] else 'Sealing incomplete'
        label = 'Packet number: '+number['text'] if number.get('status') == 'confirmed' else 'Packet number not confirmed'
        owner.phase_label.setText(sealing+' | '+label)
        owner.add_log('Packet evidence saved: '+str(result['directory']), 'info')

    def reset(self):
        if self.pending is not None:
            return
        if self.window is not None:
            self.reset_requested = True
            self.owner.btn_reset.setEnabled(False)
            self.window.finish_session()

    def close(self):
        if self.pending is not None:
            return False
        if self.window is not None:
            worker = self.window.worker
            if worker is not None and worker.isRunning():
                self.window.finish_session()
                self.owner.lbl_status.setText('Saving packet evidence. Close again after export finishes.')
                return False
            self.window.close()
        if self.draining is not None:
            self.owner.lbl_status.setText('Waiting for pending jewel analysis and evidence to finish…')
            return False
        self.release_gold()
        return True
