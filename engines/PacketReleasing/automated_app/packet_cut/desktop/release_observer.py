"""Independent release checks: absent evidence never gates another observer."""

from copy import deepcopy
import time
from datetime import datetime, timedelta

import cv2
import numpy as np

from inspection.utils.image_utils import evaluate_frame_quality, draw_mask_overlay
from workflow_monitor.activity_log import ComplianceStatus
from .state_machine import AutoStage


CHECKS = {
    'initial_testbed': 'Initial testbed clearance',
    'packet_number': 'Packet number / OCR',
    'seal': 'Backside / seal strip',
    'cutting_activity': 'Cutting action',
    'cut_separation': 'Physical cut separation',
    'packet_mouth': 'Packet mouth',
    'gold_placement': 'Jewellery placement',
    'jewellery_count': 'Jewellery count',
}
MISSING_REASONS = {
    'initial_testbed': 'Initial testbed clearance not verified.',
    'packet_number': 'Packet ID not read: number not shown, unreadable, or OCR unavailable.',
    'seal': 'Backside / seal condition not verified.',
    'cutting_activity': 'Cutting action not observed.',
    'cut_separation': 'Physical separation of cut pieces not verified.',
    'packet_mouth': 'Packet mouth not presented clearly or not verified.',
    'gold_placement': 'Jewellery placement not observed.',
    'jewellery_count': 'Stable jewellery count not verified.',
}
STAGE_KEYS = {
    AutoStage.WAIT_FRONTSIDE_AND_OCR: 'packet_number',
    AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP: 'seal',
    AutoStage.WAIT_CUTTING_ACTIVITY: 'cutting_activity',
    AutoStage.VERIFY_PHYSICAL_CUT: 'cut_separation',
    AutoStage.WAIT_UPRIGHT_MOUTH: 'packet_mouth',
    AutoStage.WAIT_GOLD_PLACEMENT: 'gold_placement',
}
COUNTERS = (
    '_empty_streak', '_frontside_streak', '_frontside_seen_once', '_prev_pkt_centroid',
    '_prev_pkt_area', '_ocr_votes', '_ocr_best_conf', '_ocr_missing_streak',
    '_turn_motion_observed', '_backside_ready_streak', '_hand_streak', '_scissor_streak',
    '_cutting_both_streak', '_next_cut_sample_mono', '_cut_valid_streak',
    '_cut_analysis_count', '_cut_valid_samples_in_batch', '_cut_prev_sig',
    '_mouth_open_streak', '_mouth_analysis_count', '_mouth_open_samples_in_batch',
    '_mouth_closed_samples_in_batch', '_batch_gold_seen', '_open_streak_gold_confidence',
    '_mouth_closed_streak', '_outside_testbed_streak', '_uncertain_count_streak',
    '_count_candidate', '_count_stable_streak',
)


class ReleaseObserver:
    def __init__(self, pipeline):
        self.p = pipeline
        self.defaults = {name: deepcopy(getattr(pipeline, name)) for name in COUNTERS}
        self.counters = {}
        self.checks = {key: {'key': key, 'title': title, 'status': 'UNVERIFIED',
                             'result': 'NOT_OBSERVED', 'reason': MISSING_REASONS[key],
                             'observed_at': None, 'recovered_late': False}
                       for key, title in CHECKS.items()}
        self.now = pipeline.started_mono
        self.initial_frames = 0
        self.initial_occupied = False
        self.initial_clear_streak = 0
        self.last_clear_count = None
        self.last_clear_at = None
        self.last_clear_image = None
        self.visible_count = None
        self.count_streak = 0
        self.count_candidate = None
        self.count_uncertain = False
        self.last_evidence_time = float('-inf')
        self.front_visible_streak = 0
        self.front_view_observed = False
        self.last_run = {}
        self.last_count_frame = float('-inf')

    def due(self, key, interval, supplied=False):
        # Overrides are already fresh detections supplied by a replay/test caller.
        if supplied:
            return True
        if self.now - self.last_run.get(key, float('-inf')) + 1e-9 < interval:
            return False
        self.last_run[key] = self.now
        return True

    def timestamp(self):
        return (datetime.fromisoformat(self.p.started_at) + timedelta(seconds=self.now - self.p.started_mono)).isoformat()

    def note(self, key, result, reason, evidence=None, status='VERIFIED'):
        previous = self.checks[key]
        if previous['result'] == result and previous['status'] == status and previous['reason'] == reason:
            return
        keys = list(CHECKS)
        late = any(self.checks[k]['observed_at'] for k in keys[keys.index(key) + 1:])
        self.checks[key] = {**previous, 'status': status, 'result': result, 'reason': reason,
                            'observed_at': self.timestamp(), 'evidence_image': evidence,
                            'recovered_late': late or previous['recovered_late']}
        p = self.p
        p.activity_logger.log(p._packet_id, p.stage.value, key, CHECKS[key], reason,
                              ComplianceStatus.RECOVERED if late else ComplianceStatus.INFO,
                              evidence_image=evidence, metadata={'observation_status': status, 'late': late})
        p.state_machine.record_event(p._packet_id, reason, 'Release observers', 1.0,
                                     'LATE_OBSERVATION' if late else result, 'INFO', evidence)

    def transition_observed(self, context, result, reason, evidence):
        key = STAGE_KEYS.get(context)
        if key:
            self.note(key, result, reason, evidence)

    def _run(self, stage, handler, frame, ov, models):
        """Reuse physical validators with separate temporal state per check."""
        p = self.p
        if stage not in self.counters:
            self.counters[stage] = deepcopy(self.defaults)
        bank = self.counters[stage]
        for name, value in bank.items():
            setattr(p, name, value)
        p._observation_context = stage
        try:
            return handler(frame, self.now, ov, models)
        except Exception as exc:
            key = STAGE_KEYS.get(stage, 'jewellery_count')
            self.checks[key]['reason'] = f'Observation unavailable: {type(exc).__name__}: {exc}'
            p.recovery.last_error = self.checks[key]['reason']
            p.recovery.total_inference_errors += 1
            return None
        finally:
            self.counters[stage] = {name: getattr(p, name) for name in COUNTERS}
            p._observation_context = None

    def process(self, frame, ts, overrides, models):
        p = self.p
        self.now = ts
        ov = dict(overrides)
        # Run shared detectors once; independent validators reuse their masks.
        before = p.recovery.total_inference_errors
        packet_known = 'packet_mask' in ov or p.model_manager.has('packet')
        info = p._run_packet_seg(frame, ov, models) if packet_known else None
        packet_known = packet_known and before == p.recovery.total_inference_errors
        if packet_known:
            ov['packet_mask'] = info['union_mask']
        hand_known = 'hand_present' in ov or 'hand_mask' in ov or p.model_manager.has('hand')
        scissor_available = 'scissor_present' in ov or 'scissor_mask' in ov or p.model_manager.has('scissor')
        scissor_known = False
        hp = sp = False
        hm = sm = np.zeros(frame.shape[:2], dtype=np.uint8)
        if hand_known:
            before = p.recovery.total_inference_errors
            hp, hm, _, _, _ = p._run_hand_and_scissor(frame, ov, models, run_scissor=False, full_view=True)
            hand_known = before == p.recovery.total_inference_errors
        if hand_known:
            ov.update(hand_present=hp, hand_mask=hm)
        scissors_complete = (self.checks['cutting_activity']['status'] == 'VERIFIED'
                             and p._cut_result['state'] == 'CUT_CONFIRMED')
        scissor_supplied = 'scissor_present' in ov or 'scissor_mask' in ov
        scissor_scene = hp or (packet_known and bool(info['pieces']))
        if scissor_available and not scissors_complete and (scissor_scene or scissor_supplied) and self.due(
                'scissor', p.pipe_cfg.scissor_check_interval_sec, scissor_supplied):
            before = p.recovery.total_inference_errors
            scissor_input = {**ov, 'hand_present': hp, 'hand_mask': hm}
            _, _, sp, sm, _ = p._run_hand_and_scissor(frame, scissor_input, models, full_view=True)
            scissor_known = before == p.recovery.total_inference_errors
        if scissor_known:
            ov.update(scissor_present=sp, scissor_mask=sm)
        jewel_supplied = 'jewel_instances' in ov
        jewel_available = jewel_supplied or p.model_manager.has('jewel')
        jewel_known = jewel_supplied
        jewel_attempted = jewel_supplied
        # A hand-obscured view cannot establish a clear count. Do not spend a
        # jewellery inference on it; the previous clear evidence is retained.
        if not jewel_supplied and jewel_available and hand_known and not hp and self.due('jewel', p.pipe_cfg.jewel_check_interval_sec):
            jewel_attempted = True
            before = p.recovery.total_inference_errors
            detected, available = p._safe_infer('jeweltrack.onnx', lambda: p.jewel_detector._predict_instances(
                'jewel', frame, p.insp_cfg.jewellery_counting.jewel_conf,
                p.insp_cfg.jewellery_counting.min_article_area_px), ([], False), models)
            jewel_known = available and before == p.recovery.total_inference_errors
            if jewel_known:
                models.append('jeweltrack.onnx')
                ov['jewel_instances'] = detected
                ov.setdefault('mouth_jewel_instances', detected)

        # Only the initial view is assessed here; an occupied start never waits.
        if self.initial_frames < p.pipe_cfg.empty_testbed_frames and not self.initial_occupied:
            if packet_known and hand_known and (hp or info['total_area'] >= 1200):
                self.initial_occupied = True
                self.note('initial_testbed', 'INITIAL_NOT_CLEAR', 'Initial testbed not clear (occupied); observation continued.', status='UNVERIFIED')
            elif packet_known and hand_known and jewel_known:
                self.initial_frames += 1
                empty = p.testbed_roi.check_empty(frame, len(ov['jewel_instances']), hp, info['total_area'] >= 1200)
                quality = evaluate_frame_quality(frame, min_blur_variance=0)
                if not empty.raw_is_empty or not quality.is_acceptable:
                    self.initial_occupied = True
                    self.note('initial_testbed', 'INITIAL_NOT_CLEAR', 'Initial testbed not clear or not visible; observation continued.', status='UNVERIFIED')
                else:
                    self.initial_clear_streak += 1
                    if self.initial_clear_streak >= p.pipe_cfg.empty_testbed_frames:
                        p._empty_verified = True
                        p.procedure_monitor.on_testbed_empty_confirmed(p._packet_id, p.stage.value)
                        self.note('initial_testbed', 'EMPTY_OBSERVED', 'Initial testbed observed clear across fresh frames.')
            elif jewel_attempted or not packet_known or not hand_known:
                self.initial_clear_streak = 0
            if not p._empty_verified and not self.initial_occupied and ts - p.started_mono > max(2.5, p.pipe_cfg.empty_testbed_frames * p.pipe_cfg.jewel_check_interval_sec * 2):
                self.initial_occupied = True
                self.checks['initial_testbed']['reason'] = 'Initial clearance not verified in the starting view; observation continued.'

        if packet_known:
            # Seeing the printed number region is independent of reading its ID.
            # Keep a front appearance reference even if OCR is unavailable, so
            # a later turn/strip observation is not gated by OCR success.
            number_known = 'ocr_roi' in ov or p.model_manager.has('number')
            number_fresh = False
            # Both the number-region detector and text recognizer become idle
            # once two consistent readings have locked this session's ID.
            needs_number = not p._frontside_verified and (not self.front_view_observed or 'ocr_text' in ov or p.model_manager.has('ocr'))
            if number_known and needs_number and len(info['pieces']) == 1 and self.due(
                    'number', p.pipe_cfg.number_check_interval_sec, 'ocr_roi' in ov):
                number_fresh = True
                number_started = time.perf_counter()
                number_ok, number_box, number_conf = p._run_number_obb(frame, ov, models)
                p._ocr_timing["number_detector_calls"] += 1
                p._ocr_timing["number_detector_total_ms"] += (time.perf_counter() - number_started) * 1000
                ov['ocr_roi'] = number_box if number_ok else None
                ov['frontside_conf'] = number_conf
                belongs = False
                if number_ok and number_box is not None and len(info['pieces']) == 1:
                    x1, y1, x2, y2 = number_box
                    overlap = np.count_nonzero(info['union_mask'][y1:y2, x1:x2])
                    belongs = overlap / max((x2 - x1) * (y2 - y1), 1) >= p.pipe_cfg.frontside_min_packet_overlap
                self.front_visible_streak = self.front_visible_streak + 1 if belongs and hand_known and not hp else 0
                if self.front_visible_streak >= p.pipe_cfg.frontside_stable_frames and not self.front_view_observed:
                    p._frontside_ref_hist, p._frontside_grid_density = p._packet_appearance_signature(frame, info['union_mask'])
                    p._frontside_centroid = info['pieces'][0]['centroid']
                    self.front_view_observed = True
            if number_fresh and not p._frontside_verified and ('ocr_text' in ov or p.model_manager.has('ocr')):
                self._run(AutoStage.WAIT_FRONTSIDE_AND_OCR, p._handle_stage2_frontside_and_ocr, frame, ov, models)
            if not p._backside_verified and len(info['pieces']) == 1 and hand_known and not hp and self.due(
                    'strip', p.pipe_cfg.strip_check_interval_sec, 'topstrip_visible' in ov):
                strip, strip_mask = p._run_topstrip(frame, ov, models)
                if strip is not None:
                    ov['topstrip_visible'] = strip
                    ov['topstrip_mask'] = strip_mask
                if strip is True:
                    p._strip_result = 'STRIP_VISIBLE'
                    image = None
                    if self.checks['seal']['result'] != 'STRIP_VISIBLE':
                        image = p._save_evidence(frame, 'seal_strip_visible')
                    self.note('seal', 'STRIP_VISIBLE', 'Strip visible: OPEN / NOT SEALED under the configured strip rule.', image, status='OBSERVED')
                # Absence alone cannot establish sealing or the packet backside.
                elif strip is False:
                    p._strip_result = 'STRIP_ABSENT_SEAL_UNVERIFIED'
                    self.checks['seal']['reason'] = 'Strip not detected; backside/seal condition not yet verified.'
                if p._frontside_verified or self.front_view_observed or ov.get('backside_visually_verified'):
                    self._run(AutoStage.WAIT_TURN_AND_BACKSIDE_STRIP, p._handle_stage3_turn_and_backside, frame, ov, models)
            if hand_known and scissor_known:
                if self.checks['cutting_activity']['status'] != 'VERIFIED':
                    self._run(AutoStage.WAIT_CUTTING_ACTIVITY, p._handle_stage4_cutting_activity, frame, ov, models)
                    if p._cutting_activity_observed:
                        self.note('cutting_activity', 'CUTTING_OBSERVED', 'Hand and scissors observed together; other checks continued.', p._cutting_evidence)
                cut_view = len(info['pieces']) == 2 or (p._cutting_activity_observed and info['pieces'] and p._cut_evidence_score < 0)
                if (cut_view and p._cut_result['state'] != 'CUT_CONFIRMED'
                        and ('cut_eval' in ov or p.model_manager.has('cut'))):
                    self._run(AutoStage.VERIFY_PHYSICAL_CUT, p._handle_stage5_verify_cut, frame, ov, models)
            if p._mouth_status == 'UNKNOWN' and self.due('mouth', p.pipe_cfg.mouth_check_interval_sec, 'mouth_state_override' in ov):
                self._run(AutoStage.WAIT_UPRIGHT_MOUTH, p._handle_stage6_upright_mouth, frame, ov, models)

        count_snapshot = None
        if jewel_known and packet_known and hand_known:
            context = AutoStage.WAIT_JEWELLERY_AND_COUNT if self.checks['gold_placement']['observed_at'] else AutoStage.WAIT_GOLD_PLACEMENT
            count_snapshot = self._run(context, p._handle_stage7_jewellery_counting, frame, ov, models)
            self.visible_count = None if hp or count_snapshot is None else p._provisional_jewel_count
            if count_snapshot is not None and not hp:
                self.last_count_frame = ts
            self.count_uncertain = hp or count_snapshot is None or p._jewel_count_status == 'PROVISIONAL_WITH_UNCERTAINTY'
            if not self.count_uncertain and self.visible_count and count_snapshot is not None:
                self.count_streak = self.count_streak + 1 if self.count_candidate == self.visible_count else 1
                self.count_candidate = self.visible_count
                if self.count_streak >= p.pipe_cfg.pass_by_count_frames:
                    # Retain the clearest observed full set as articles are taken
                    # away for later work. This is simultaneous count, never a sum.
                    if (self.last_clear_count is None or self.visible_count >= self.last_clear_count) and (
                            self.last_clear_count != self.visible_count or ts - self.last_evidence_time >= 3):
                        self.last_clear_at = self.timestamp()
                        self.last_evidence_time = ts
                        self.last_clear_count = self.visible_count
                        p._observation_context = AutoStage.WAIT_JEWELLERY_AND_COUNT
                        try:
                            evidence = count_snapshot.annotated_frame.copy()
                            cv2.putText(evidence, f'Packet {p._packet_id} | Observed count: {self.visible_count}',
                                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (30, 160, 30), 2)
                            self.last_clear_image = p._save_evidence(evidence,
                                                                   f'stage7_provisional_{self.visible_count}')
                        finally:
                            p._observation_context = None
                        p._jewel_evidence = self.last_clear_image
                        self.note('jewellery_count', 'COUNT_OBSERVED', f'Last clear count: {self.last_clear_count} (provisional).',
                                  self.last_clear_image, status='OBSERVED')
            else:
                self.count_streak = 0
        elif jewel_available and not jewel_attempted and packet_known and hand_known:
            # No result is reused as a new observation or a confirmation frame.
            # Preserve the most recent display briefly between scheduled checks.
            if hp or ts - self.last_count_frame > 1.5:
                self.visible_count = None
                self.count_uncertain = True
                self.count_streak = 0
        else:
            self.visible_count = None
            self.count_uncertain = True
            self.count_streak = 0
            self.checks['jewellery_count']['reason'] = 'Count not verified: jewellery, packet exclusion, or hand visibility unavailable.'

        p._instruction = 'Observing packet release — continue your work. STOP saves available evidence and missing checks.'
        observed = [v for v in self.checks.values() if v['observed_at']]
        p._latest_observed_action = (max(observed, key=lambda item: item['observed_at'])['reason'] if observed else
                                    'Observing release; no required stage order.')
        p._packet_det_summary = 'Packet ID not shown / unreadable' if not p._frontside_verified else f'Packet ID {p._packet_id} read by OCR'
        visual = count_snapshot.annotated_frame if count_snapshot else frame.copy()
        for present, mask, colour, label in (
                (hand_known and hp, hm, (50, 50, 230), 'HAND'),
                (scissor_known and sp, sm, (230, 180, 40), 'SCISSORS')):
            if present and np.any(mask):
                visual = draw_mask_overlay(visual, mask, colour, alpha=.3)
                x, y, width, height = cv2.boundingRect(mask)
                cv2.rectangle(visual, (x, y), (x + width, y + height), colour, 2)
                cv2.putText(visual, label, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, .55, colour, 2)
        active = 'Checks · OCR off' if p._frontside_verified else 'Checks · crop OCR'
        snapshot = p._build_snapshot(visual, active, list(dict.fromkeys(models)), p._instruction, False, False)
        snapshot.live_monitoring.update(
            hand_detected=bool(hp) if hand_known else None,
            scissor_detected=bool(sp) if scissor_known else None,
            cutting_detection_fresh=bool(scissor_known or not hp or not scissor_available),
        )
        action = ('CUTTING ACTIVITY DETECTED' if scissor_known and hp and sp and p._cutting_activity_observed else
                  'Cutting observed this session' if p._cutting_activity_observed else 'Cutting action not observed')
        snapshot.live_monitoring['cutting_verification'] = f"{action}\nSeparation: {p._cut_result['state'].replace('_', ' ')}"
        return snapshot

    def count_result(self):
        return {
            'count': self.last_clear_count, 'provisional_count': self.last_clear_count,
            'last_clear_count': self.last_clear_count, 'last_clear_timestamp': self.last_clear_at,
            'current_visible_count': self.visible_count,
            'confirmed_unambiguous_count': self.last_clear_count,
            'current_unambiguous_count': self.p._confirmed_unambiguous_count if self.visible_count is not None else None,
            'count_status': ('NOT_VERIFIED' if self.last_clear_count is None else
                             'PROVISIONAL_WITH_UNCERTAINTY' if self.count_uncertain else 'PROVISIONAL'),
            'evidence_image': self.last_clear_image,
            'reason': 'Highest stable simultaneously visible count retained as articles leave view; not a cumulative inventory, final verified total, or proof all articles were removed.',
        }
