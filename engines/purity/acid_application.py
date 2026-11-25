"""Verify detector candidates with bounded stone-relative camera observations.

This is a visual persistence/occlusion heuristic, not a chemical measurement or
a needle model. Recording runs independently while candidates are provisional.
"""
import cv2
import numpy as np

from acid_freshness import DEST, normalized, footprint, changed_pixels


ACID_ROI_MARGIN = .10


def acid_roi_support(region):
    """Expand an acid footprint by 10% per side, independently of the stone."""
    region = (region > 0).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(region)
    if not w or not h:
        return region > 0
    px, py = max(1, int(np.ceil(w*ACID_ROI_MARGIN))), max(1, int(np.ceil(h*ACID_ROI_MARGIN)))
    return cv2.dilate(region, np.ones((2*py+1, 2*px+1), np.uint8)) > 0


def acid_roi_interference(image, support, baseline=None):
    """Shared live/replay interference checks confined to the acid support."""
    if image is None or not np.any(support):
        return 'ACID_ROI_UNAVAILABLE'
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin = (hsv[:,:,0] <= 14) & (hsv[:,:,1] >= 45) & (hsv[:,:,2] > 80)
    light = (hsv[:,:,1] < 40) & (hsv[:,:,2] > 150)
    if float(skin[support].mean()) > .2 or float(light[support].mean()) > .8:
        return 'HAND_OR_GLARE_OVER_CANDIDATE'
    _, tool = visible_patch(image, support)
    if baseline is not None:
        # Compare the bright feature's own pixels; blur halos from a neighbouring
        # wet patch must not turn an unchanged rubbed streak into a new needle.
        tool &= cv2.absdiff(baseline,image).max(axis=2)>15
    # An applicator elsewhere on the stone must not block this acid ROI.
    if AcidApplicationVerifier._tool_like(tool & support, support):
        return 'APPLICATOR_LIKE_CHANGE_REMAINS'
    return None


def baseline_roi_interference(image, support):
    """A stationary bright rubbed streak is valid baseline texture, not a tool."""
    if image is None or not np.any(support):return 'BASELINE_ACID_ROI_UNAVAILABLE'
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    skin=(hsv[:,:,0]<=14)&(hsv[:,:,1]>=45)&(hsv[:,:,2]>80)
    light=(hsv[:,:,1]<40)&(hsv[:,:,2]>150)
    if float(skin[support].mean())>.2:return 'BASELINE_HAND_OVER_ACID_ROI'
    if float(light[support].mean())>.8:return 'BASELINE_GLARE_OVER_ACID_ROI'
    return None


def visible_patch(image, support):
    """Current local dark/coloured patch; no comparison with pre-acid pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ring = (cv2.dilate(support.astype(np.uint8),np.ones((15,15),np.uint8))>0) & ~support
    background = float(np.median(gray[ring])) if np.any(ring) else float(np.median(gray))
    patch = support & ((gray.astype(float)<background-8) | (hsv[:,:,1]>45))
    patch[:5]=False;patch[-5:]=False;patch[:,:5]=False;patch[:,-5:]=False
    # Keep the whole connected applicator, including the portion outside the box.
    bright_tool = (gray.astype(float)>background+35) & (hsv[:,:,1]<40)
    return patch, bright_tool


class AcidApplicationVerifier:
    def __init__(self, baseline, old_mask, window=5., max_candidates=2, min_clear_frames=2,
                 baseline_alternatives=(), baseline_timestamp=None):
        self.baseline = baseline.copy()
        # Historical locations must not exclude evidence of a new application.
        self.window, self.max_candidates = window, max_candidates
        self.min_clear_frames = max(2, int(min_clear_frames))
        self.pending = []
        self.events = []
        self.last_ts = -float('inf')
        self.baseline_timestamp=baseline_timestamp
        self.baseline_alternatives=list(baseline_alternatives)[-12:]
        self.first_candidate_timestamp=None

    def _event(self, status, timestamp, candidate, **details):
        self.events.append(dict(status=status, timestamp=timestamp,
                                candidate_at=candidate['timestamp'], **details))
        self.events = self.events[-60:]

    def _drop(self, candidate):
        self.pending = [item for item in self.pending if item is not candidate]

    def offer(self, frame, quad, detection, timestamp):
        if self.first_candidate_timestamp is None or timestamp<self.first_candidate_timestamp:
            self.first_candidate_timestamp=timestamp
        region = footprint(detection['bbox'], quad) > 0
        for candidate in self.pending:
            overlap = np.count_nonzero(region & candidate['seed'])
            if overlap / max(1, min(region.sum(), candidate['seed'].sum())) >= .2:
                return False
        if len(self.pending) >= self.max_candidates:
            previous = self.pending.pop(0)
            self._event('CANDIDATE_REPLACED_UNVERIFIED', timestamp, previous)
        candidate = dict(timestamp=timestamp, detection=dict(detection), seed=region,
                         source_quad=np.asarray(quad, np.float32).copy(),
                         previous=None,
                         previous_ts=None, patch=None, first=None, count=0,
                         reason='CANDIDATE_RECORDING')
        self.pending.append(candidate)
        self._event('CANDIDATE_RECORDING', timestamp, candidate)
        return True

    @staticmethod
    def _tool_like(changed, seed):
        # A straight elongated change touching the candidate may be the needle
        # even while stationary. Motion settling alone must never confirm it.
        _, labels, stats, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), 8)
        # Inspect only components touching this candidate, not every texture
        # change elsewhere on the stone.
        for index in np.unique(labels[seed & changed]):
            if index == 0:
                continue
            if stats[index, cv2.CC_STAT_AREA] < 8:
                continue
            component = labels == index
            ys, xs = np.where(component)
            _, (a, b), _ = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
            long, short = max(a, b), max(1., min(a, b))
            if long >= 24 and long / short >= 6:
                return True
        return False

    def observe(self, frame, quad, timestamp, clear=True, geometry_known=True):
        if timestamp <= self.last_ts or not self.pending:
            return None
        self.last_ts = timestamp
        # Use available stone coordinates even when the pose/clear-view
        # heuristics are uncertain. They no longer veto acid candidates.
        image = normalized(frame, quad) if quad is not None else None
        for candidate in list(self.pending):
            if timestamp < candidate['timestamp']:
                continue
            if timestamp - candidate['timestamp'] > self.window:
                self._event('APPLICATION_OBSCURED_OR_NOT_PERSISTENT', timestamp, candidate,
                            reason=candidate['reason'])
                self._drop(candidate)
                continue
            support = acid_roi_support(candidate['seed'])
            patch = visible_patch(image,support)[0] if image is not None else None
            baseline=candidate.get('baseline',self.baseline)
            reason = acid_roi_interference(image, support, baseline=baseline)
            baseline_reason=baseline_roi_interference(baseline,support)
            baseline_ts=candidate.get('baseline_timestamp',self.baseline_timestamp)
            if baseline_ts is not None and baseline_ts>=self.first_candidate_timestamp:
                baseline_reason='BASELINE_CAPTURED_AFTER_FIRST_CANDIDATE'
            if baseline_reason is not None:
                if candidate.get('baseline_reason')!=baseline_reason:
                    self._event('BASELINE_ROI_REJECTED',timestamp,candidate,reason=baseline_reason,
                                baseline_timestamp=self.baseline_timestamp)
                    candidate['baseline_reason']=baseline_reason
                cutoff=self.first_candidate_timestamp
                choices=[item for item in self.baseline_alternatives
                         if item['timestamp']<cutoff-.25 and baseline_roi_interference(item['image'],support) is None]
                if choices:
                    complete_context=[item for item in choices if item['timestamp']<=cutoff-2]
                    chosen=(max(complete_context,key=lambda item:item['timestamp']) if complete_context
                            else min(choices,key=lambda item:item['timestamp']))
                    candidate['baseline']=baseline=chosen['image']
                    candidate['baseline_timestamp']=chosen['timestamp']
                    candidate['baseline_recovered']=True
                    candidate.update(count=0,first=None,previous=None,patch=None)
                    baseline_reason=None
                    reason=acid_roi_interference(image,support,baseline=baseline)
                    self._event('BASELINE_ROI_RECOVERED',timestamp,candidate,
                                baseline_timestamp=chosen['timestamp'],source='RECORDED_PRE_ACID_FRAME')
            if reason is None and baseline_reason is not None:
                reason=baseline_reason
            if image is None:
                # Coordinates are required to compare the same stone pixels.
                reason = 'STONE_COORDINATES_UNAVAILABLE'
            if reason is not None:
                candidate.update(count=0, first=None, previous=None, patch=None, reason=reason)
                continue
            if patch.sum() < 5:
                candidate.update(count=0, first=None, previous=None, patch=None,
                                 reason='NO_VISIBLE_PATCH')
                # A disappeared tool is not an acid application.
                if not np.any(patch) and timestamp - candidate['timestamp'] >= .1:
                    self._event('CANDIDATE_DISAPPEARED', timestamp, candidate)
                    self._drop(candidate)
                continue
            # A stable dry mark/residue already present in the baseline must
            # not become acid merely because the cropped model detects it.
            # Require a majority change; JPEG colour halos around an unchanged
            # mark can otherwise pass a small changed-pixel fraction.
            changed_patch = patch & changed_pixels(baseline, image)
            if changed_patch.sum() < 5 or changed_patch.sum()/max(1, patch.sum()) < .50:
                candidate.update(count=0, first=None, previous=None, patch=None,
                                 reason='NO_NEW_PATCH_SINCE_BASELINE')
                continue
            previous = candidate['previous']
            previous_ts = candidate['previous_ts']
            stable = False
            if previous is not None and 0 < timestamp - previous_ts <= .4:
                union = patch | candidate['patch']
                overlap = np.count_nonzero(patch & candidate['patch']) / max(1, union.sum())
                moving = float(changed_pixels(previous, image)[union].mean())
                stable = overlap >= .65 and moving <= .12
            if not stable:
                candidate['count'] = 1
                ys, xs = np.where(patch)
                # All four corners are needed for rotated/perspective stones.
                corners = np.float32([[[xs.min(), ys.min()], [xs.max()+1, ys.min()],
                                      [xs.max()+1, ys.max()+1], [xs.min(), ys.max()+1]]])
                corners = cv2.perspectiveTransform(corners, cv2.getPerspectiveTransform(
                    DEST, np.asarray(quad, np.float32)))[0]
                lo, hi = np.floor(corners.min(0)).astype(int), np.ceil(corners.max(0)).astype(int)
                lo = np.maximum(lo, 0); hi = np.minimum(hi, [frame.shape[1], frame.shape[0]])
                patch_bbox = tuple(map(int, [*lo, *hi]))
                # Persistence verifies the application; it must not shrink the
                # detector's full acid ROI to just its changed pixels. Map the
                # original detection into this exact supported source frame.
                x1, y1, x2, y2 = candidate['detection']['bbox']
                detector_corners = np.float32([[[x1, y1], [x2, y1],
                                               [x2, y2], [x1, y2]]])
                detector_corners = cv2.perspectiveTransform(detector_corners,
                    cv2.getPerspectiveTransform(candidate['source_quad'],
                                                np.asarray(quad, np.float32)))[0]
                lo = np.maximum(np.floor(detector_corners.min(0)).astype(int), 0)
                hi = np.minimum(np.ceil(detector_corners.max(0)).astype(int),
                                [frame.shape[1], frame.shape[0]])
                candidate['first'] = dict(timestamp=timestamp, source_frame=frame,
                                          quad=np.asarray(quad).copy(),
                                          verified_patch_bbox=patch_bbox,
                                          bbox=tuple(map(int, [*lo, *hi])))
            else:
                candidate['count'] += 1
            candidate.update(previous=image.copy(), previous_ts=timestamp,
                             patch=patch, reason='CHECKING_CLEAR_PATCH_PERSISTENCE')
            first = candidate['first']
            if candidate['count'] >= self.min_clear_frames and timestamp-first['timestamp'] >= .03:
                status = ('VISIBLE_PATCH_ONSET' if first['timestamp']-candidate['timestamp'] <= .15
                          else 'ONSET_OBSCURED_USING_FIRST_CLEAR_PATCH')
                if candidate['detection'].get('freshness_warning'):
                    status = 'PATCH_OBSERVATION_TIME_FRESH_APPLICATION_UNVERIFIED'
                best = dict(candidate['detection'], **first,
                            application_candidate_at=candidate['timestamp'],
                            application_verified_at=timestamp,
                            application_timing_status=status,
                            application_verification='DETECTOR_AND_VISIBLE_PERSISTENT_PATCH',
                            clear_patch_frames=candidate['count'])
                if candidate.get('baseline_recovered'):
                    best.update(recovered_baseline=candidate['baseline'],
                                recovered_baseline_timestamp=candidate['baseline_timestamp'])
                self._event('APPLICATION_VERIFIED', timestamp, candidate, onset=first['timestamp'])
                self.pending.clear()
                return first['timestamp'], best
        return None
