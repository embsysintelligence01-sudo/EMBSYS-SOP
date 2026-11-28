"""Fixed-box recording cues adapted from the standalone pickup test."""
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
from engines.sealing.hand_tracking import draw_hand

HOME_GOLD_FULL = 0.4
HOME_WHITE_EMPTY = 0.7
HOME_EMPTY_S = 0.3
HAND_HOLD_S = 0.45
HAND_CROP_MIN_PX = 720
BG_WHITE_S_MAX, BG_WHITE_V_MIN = 40, 170
HAND_CHECK_S = 0.15
MAX_SAMPLE_GAP = 0.65
HOLD_S = 0.5
RETURN_STABLE_S = 0.8
RETURN_MOVE_PX = 15


def draw_hand_preview(frame, ts, preview, source_shape=None):
    """Draw fresh diagnostics on a preview copy, leaving recorded pixels intact."""
    shape = frame.shape if source_shape is None else source_shape
    if (not preview or tuple(shape[:2]) != preview['shape']
            or not 0 <= ts-preview['timestamp'] <= MAX_SAMPLE_GAP):
        return frame
    image = frame.copy()
    scale = np.array([frame.shape[1] / shape[1], frame.shape[0] / shape[0]])
    for hand in preview['hands']:
        points = hand['points'] * scale
        draw_hand(image, SimpleNamespace(points=points), thickness=2)
        thumb, index = np.rint(points[[4, 8]]).astype(int)
        color = ((150, 150, 150) if not hand['selected'] else
                 (0, 220, 0) if hand['pinched'] and hand['in_area'] else (0, 165, 255))
        cv2.line(image, tuple(thumb), tuple(index), color, 3)
        middle = np.rint((thumb+index)/2).astype(int)
        cv2.circle(image, tuple(middle), 6, color, -1)
        label = f"{hand['handedness']} {hand['distance_px']:.0f} / {preview['pinch_px']} px"
        if not hand['selected']:
            label += " (ignored)"
        cv2.putText(image, label,
                    tuple(middle+(8, -10)), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    source = preview.get('trigger_source', 'PINCH')
    message = f"HAND {preview['selection']} | {source} HOLD {preview['hold_seconds']:.1f} / {HOLD_S:.1f} s | {preview['state']}"
    cv2.putText(image, preview.get('detail', ''), (15, image.shape[0]-55),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 255), 2)
    cv2.putText(image, message, (15, image.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX,
                .7, (0, 220, 255), 2)
    return image


def home_view(home, dets, frame):
    """'gold' (raw gold masks cover the home mask), 'empty' (its pixels show white bench) or 'covered'."""
    x1, y1, x2, y2 = home.box
    fh, fw = frame.shape[:2]
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(fw, x2), min(fh, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return 'covered'
    m = home.mask[cy1-y1:cy2-y1, cx1-x1:cx2-x1]
    n = int(np.count_nonzero(m))
    if n == 0:
        return 'covered'
    gold = np.zeros(m.shape, bool)
    for d in dets:
        dm = getattr(d, 'mask', None)
        if dm is None:
            continue
        ox, oy = (int(v) for v in getattr(d, 'origin', d.box[:2]))
        h, w = dm.shape[:2]
        ax1, ay1, ax2, ay2 = max(cx1, ox), max(cy1, oy), min(cx2, ox+w), min(cy2, oy+h)
        if ax2 > ax1 and ay2 > ay1:
            gold[ay1-cy1:ay2-cy1, ax1-cx1:ax2-cx1] |= np.asarray(dm)[ay1-oy:ay2-oy, ax1-ox:ax2-ox] > 0
    if np.count_nonzero(gold & m)/n >= HOME_GOLD_FULL:
        return 'gold'
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[cy1:cy2, cx1:cx2]), cv2.COLOR_BGR2HSV)[m]
    pad = 40   # "white" relative to the bench right around it (light differs across the bench)
    ring = cv2.cvtColor(np.ascontiguousarray(frame[max(0, cy1-pad):min(fh, cy2+pad), max(0, cx1-pad):min(fw, cx2+pad)]),
                        cv2.COLOR_BGR2HSV).reshape(-1, 3)
    bench = ring[(ring[:, 1] < BG_WHITE_S_MAX) & (ring[:, 2] > 90)]
    bench_v = float(np.median(bench[:, 2])) if len(bench) > 50 else float(BG_WHITE_V_MIN)
    white = (hsv[:, 1] < BG_WHITE_S_MAX) & (hsv[:, 2] > .85*bench_v)
    return 'empty' if float(white.mean()) >= HOME_WHITE_EMPTY else 'covered'


def pixel_box(points, shape):
    """Convert a saved pair of normalized corners to this camera frame."""
    if len(points) != 2:
        return None
    height, width = shape[:2]
    x, y = np.clip(np.asarray(points, dtype=float), 0, 1).T
    x1, x2 = sorted(round(v * (width - 1)) for v in x)
    y1, y2 = sorted(round(v * (height - 1)) for v in y)
    if min(x2 - x1, y2 - y1) < 16:
        return None
    return x1, y1, x2, y2


class StoneRecordingArea:
    """Hold a clear, settled stone's extended bounds through a jewel cycle."""

    def __init__(self):
        self.box = None
        self.candidate = None
        self.since = None
        self.last_ts = None
        self.shape = None

    def update(self, stone, ts, shape, margin_percent, active):
        if self.shape != shape[:2]:
            self.__init__()
            self.shape = shape[:2]
        if active or stone is None:
            self.candidate = self.since = None
        else:
            bounds = np.asarray(stone, dtype=float)
            if (self.candidate is None or np.max(np.abs(bounds-self.candidate)) > 6
                    or self.last_ts is None or ts-self.last_ts > MAX_SAMPLE_GAP):
                self.candidate, self.since = bounds.copy(), ts
            if ts-self.since >= HOLD_S:
                height, width = shape[:2]
                x1, y1, x2, y2 = self.candidate
                dx, dy = (x2-x1)*margin_percent/100, (y2-y1)*margin_percent/100
                self.box = (max(0, round(x1-dx)), max(0, round(y1-dy)),
                            min(width-1, round(x2+dx)), min(height-1, round(y2+dy)))
        self.last_ts = ts
        if self.box is None:
            return ()
        height, width = shape[:2]
        x1, y1, x2, y2 = self.box
        return ((x1/(width-1), y1/(height-1)), (x2/(width-1), y2/(height-1)))


@dataclass
class Home:
    box: tuple
    mask: np.ndarray
    empty_since: float | None = None
    confirmed_left: bool = False


class BoxRecordingTrigger:
    """Observe existing tracker IDs and frames. Own only the hand model and cue state."""

    def __init__(self, hand_factory):
        self.hand_factory = hand_factory
        self.hands = None
        self.hand_error = None
        self.key = None
        self.homes = {}
        self.hold = None
        self.latched = None
        self.grips = []
        self.last_ts = None
        self.last_hand_near = -float('inf')
        self.preview = None
        self.pickup = None
        self.pickup_sequence = 0
        self.resting = {}

    def close(self):
        if self.hands is not None:
            self.hands.close()
            self.hands = None

    def prepare(self, key):
        if key != self.key:
            if self.last_ts is not None:
                self.close()
            if self.key is None or key[0] != self.key[0]:
                self.homes.clear()
            self.key = key
            self.hold = None
            self.latched = None
            self.grips.clear()
            self.last_ts = None
            self.last_hand_near = -float('inf')
            self.preview = None
            self.pickup = None
            self.resting.clear()
            self.hand_error = None
        if self.hands is None and self.hand_error is None:
            try:
                self.hands = self.hand_factory()
            except Exception as exc:
                self.hand_error = str(exc)

    def discard_cue(self, cue):
        """Allow a fresh retry when the coordinator rejects an expired event."""
        if self.pickup is None or self.pickup['pickup_id'] != cue.get('pickup_id'):
            return
        if cue.get('event') == 'identity_confirmation':
            self.pickup['jewel_id'] = None
        else:
            self.pickup = None
            self.latched = self.hold = None

    def _recover_owner(self, ts, views, dets, inside, observed_at):
        """Resolve one existing recording from departure/arrival/return evidence.

        Unlike a new grip, this event must never create a second test cycle.
        Covered homes and the latest/moved track alone do not establish an ID.
        """
        pickup = self.pickup
        if pickup is None or pickup['jewel_id'] is not None:
            return None
        ids = pickup['candidate_ids']
        eligible = pickup['ownership_candidates']
        pickup['departed'].update(jid for jid in ids
                                  if jid in self.homes and self.homes[jid].confirmed_left)
        for jid in pickup['departed']:
            home = self.homes.get(jid)
            if home is not None and home.empty_since is not None:
                pickup['departure_times'].setdefault(jid, home.empty_since)
        if not pickup['single_hand']:
            return None
        # A confident gallery answer on raw gold inside the work area can
        # recover an arrival even when the position tracker lost that box.
        arrivals = set()
        for det in dets:
            result = getattr(det, 'idres', None)
            if (inside(det.c) and result is not None and result[0] in eligible
                    and result[1] >= .6 and views.get(result[0]) != 'gold'):
                arrivals.add(result[0])
        owner, reason = None, None
        if len(arrivals) > 1:
            return None
        if len(arrivals) == 1:
            owner = next(iter(arrivals))
            if owner not in ids and (pickup['departed']
                    or not all(views.get(jid) == 'gold' for jid in ids)
                    or ts-pickup['last_grip_at'] > HAND_HOLD_S):
                return None
            reason = 'recognized_gold_arrival'
        elif len(pickup['departed']) == 1:
            candidate = next(iter(pickup['departed']))
            if all(views.get(jid) == 'gold' for jid in eligible if jid != candidate):
                owner = candidate
                reason = ('home_departure_and_return' if views.get(owner) == 'gold'
                          else 'unique_home_departure')
        if owner is None:
            return None
        pickup['jewel_id'] = owner
        return dict(source=pickup['source'], event='identity_confirmation',
                    pickup_id=pickup['pickup_id'], jewel_id=owner,
                    candidate_ids=list(ids), timestamp=pickup['timestamp'],
                    confirmed_at=ts, provisional=False, ownership_reason=reason,
                    ownership_at=(observed_at if reason == 'recognized_gold_arrival'
                                  else pickup['departure_times'].get(owner, observed_at)),
                    ownership_evidence=dict(home_views=dict(views),
                        departed=sorted(pickup['departed']), recognized_arrivals=sorted(arrivals),
                        observed_at=observed_at),
                    trigger_hands=list(pickup['trigger_hands']))

    def observe(self, key, frame, ts, tracks, dets, settings, live_frame=None, live_ts=None):
        self.prepare(key)
        observation_ts = ts
        if live_ts is not None:
            if live_frame is None or live_frame.shape != frame.shape or live_ts < ts:
                return None
            if live_ts-ts > .5:
                # Hand recording can continue during a missed gold pass. Only
                # current home pixels remain evidence; old gold masks cannot
                # assert that a jewel is still at home or has arrived.
                frame, observation_ts, dets = live_frame, live_ts, ()
            ts = live_ts
        self.preview = dict(timestamp=ts, shape=tuple(frame.shape[:2]), hands=[],
                            pinch_px=settings['pinch_px'], hold_seconds=0.,
                            selection=settings.get('hand', 'Any'),
                            state='Waiting for stone area')
        box = pixel_box(settings['points'], frame.shape)
        home_box = box if box is not None else settings.get('home_exclusion')
        if home_box is None:
            return None
        previous, self.last_ts = self.last_ts, ts
        if previous is not None and ts - previous > MAX_SAMPLE_GAP:
            self.hold = None
            self.grips.clear()
            self.latched = None
            self.pickup = None
            self.resting.clear()
            for home in self.homes.values():
                home.empty_since = None
                home.confirmed_left = False
        x1, y1, x2, y2 = home_box

        def inside(point):
            x, y = point
            return x1 <= x <= x2 and y1 <= y <= y2

        candidates = []
        confirmed_ids = []
        home_states = []
        views = {}
        for track in tracks:
            jid = track['id']
            home = self.homes.get(jid)
            visible = track['state'] == 'OK' and not track['in_hand'] and track['det_seen'] == observation_ts
            mask = track['mask']
            if home is None:
                # A piece first seen in the work box cannot become a pickup.
                if visible and not inside(track['pos']) and mask is not None and np.any(mask):
                    x, y = map(int, track['origin'])
                    h, w = mask.shape[:2]
                    self.homes[jid] = Home((x, y, x+w, y+h), np.asarray(mask) > 0)
                elif track['in_hand'] or track['state'] in ('OCCLUDED', 'RUBBING', 'MISSING', 'LEFT_ROI'):
                    candidates.append(track)
                home_states.append(f"J{jid} home learned" if jid in self.homes else f"J{jid} home unknown")
                continue
            view = home_view(home, dets, frame)
            views[jid] = view
            home_states.append(f"J{jid} home {view}")
            if view == 'empty':
                if home.empty_since is None:
                    home.empty_since = observation_ts
            else:
                home.empty_since = None
            if view == 'gold':
                home.confirmed_left = False
            if home.empty_since is not None and observation_ts-home.empty_since + 1e-9 >= HOME_EMPTY_S:
                home.confirmed_left = True
            # Covered homes are provisional pickup candidates, as in the standalone.
            # A fresh jewel resting outside the work area cannot take this gesture.
            # A confirmed empty home outranks a stale/misattached position ID.
            moved = np.linalg.norm(np.asarray(track['pos']) -
                                   np.array([(home.box[0]+home.box[2])/2,
                                             (home.box[1]+home.box[3])/2])) > 40
            if view != 'gold' and (home.confirmed_left or moved or not visible or inside(track['pos'])):
                candidates.append(track)
                if home.confirmed_left:
                    confirmed_ids.append(jid)

        self.preview['detail'] = ', '.join(home_states) or 'No eligible jewel tracks'
        if box is None:
            self.hold = None
            return None
        recovery = self._recover_owner(ts, views, dets, inside, observation_ts)
        # The standalone uses a padded crop and measures in source pixels.
        px = max(int(1.5*(x2-x1)), (HAND_CROP_MIN_PX-(x2-x1))//2)
        py = max(int(1.5*(y2-y1)), (HAND_CROP_MIN_PX-(y2-y1))//2)
        height, width = frame.shape[:2]
        cx, cy = max(0, x1-px), max(0, y1-py)
        hand_frame = frame if live_frame is None else live_frame
        crop = hand_frame[cy:min(height, y2+py), cx:min(width, x2+px)]
        nearby = False
        grips = 0
        grip_labels = []
        hands = []
        if self.hands is not None and self.hand_error is None:
            try:
                hands = self.hands.detect(crop, timestamp_ms=round(ts*1000))
            except Exception as exc:
                self.hand_error = str(exc)
        for hand in hands:
            points = hand.points + (cx, cy)
            thumb, index = points[4], points[8]
            distance = float(np.linalg.norm(thumb-index))
            in_area = inside((thumb+index)/2)
            pinched = distance <= settings['pinch_px']
            handedness = getattr(hand, 'handedness', 'Unknown')
            selected = settings.get('hand', 'Any') in ('Any', handedness)
            if selected:
                nearby |= any(inside(point) for point in points)
            self.preview['hands'].append(dict(points=points.copy(), distance_px=distance,
                                              in_area=in_area, pinched=pinched,
                                              handedness=handedness, selected=selected))
            if in_area and pinched and selected:
                grips += 1
                grip_labels.append(handedness)

        if grip_labels:
            self.grips.append((ts, tuple(sorted(grip_labels))))
        self.grips = [(when, labels) for when, labels in self.grips if 0 <= ts-when <= HAND_HOLD_S]
        # Match the standalone's recent-pinch tolerance across missed/open frames.
        held = bool(self.grips)
        held_labels = self.grips[-1][1] if held else ()
        if self.pickup is not None and grip_labels:
            self.pickup['last_grip_at'] = ts
        self.preview['state'] = ('Waiting for jewel pickup' if not candidates else
                                 'Hold grip inside stone area' if not grips else 'Checking pickup')
        if candidates and not held and not any(hand['selected'] for hand in self.preview['hands']):
            self.preview['state'] = 'Selected hand not detected'
        if grips > 1:
            self.preview['state'] = 'Multiple qualifying hands; identity uncertain'
        if nearby:
            self.last_hand_near = ts
        # Re-learn a returned jewel at its new resting pose. Its confirmed trip
        # remains attached to the existing recording, not to a new grip.
        for track in tracks:
            jid = track['id']
            visible = track['state'] == 'OK' and not track['in_hand'] and track['det_seen'] == observation_ts
            at_other_home = any(other != jid and views.get(other) == 'gold'
                                and home.box[0] <= track['pos'][0] <= home.box[2]
                                and home.box[1] <= track['pos'][1] <= home.box[3]
                                for other, home in self.homes.items())
            if (not held and visible and not inside(track['pos'])
                    and self.pickup is not None and self.pickup['jewel_id'] == jid
                    and not at_other_home and track['mask'] is not None and np.any(track['mask'])):
                position = np.asarray(track['pos'])
                old = self.resting.get(jid)
                if old is None or np.linalg.norm(position-old[1]) > RETURN_MOVE_PX:
                    self.resting[jid] = ts, position.copy()
                elif ts-old[0] >= RETURN_STABLE_S:
                    x, y = map(int, track['origin'])
                    h, w = track['mask'].shape[:2]
                    self.homes[jid] = Home((x, y, x+w, y+h), np.asarray(track['mask']) > 0)
            else:
                self.resting.pop(jid, None)
        if recovery is not None:
            self.preview['state'] = 'Pickup identity recovered; continuing recording'
            return recovery
        if self.pickup is not None:
            quiet = not held and all(views.get(jid) == 'gold' for jid in self.pickup['ownership_candidates'])
            if quiet:
                since = self.pickup.setdefault('quiet_since', ts)
                if ts-since >= RETURN_STABLE_S:
                    self.pickup = None
                    self.latched = self.hold = None
            else:
                self.pickup.pop('quiet_since', None)
        if self.pickup is not None and self.pickup['jewel_id'] is not None:
            others = [track for track in candidates if track['id'] != self.pickup['jewel_id']]
            if others:
                candidates = others
        ids = tuple(sorted(track['id'] for track in candidates))
        cue = None
        if held and ids:
            cue = ('FIXED_BOX_PINCH', ids)
        elif (settings['gold_fallback'] and (self.hand_error is not None or ts-self.last_hand_near > 1.0)
              and len(candidates) == 1):
            track = candidates[0]
            # Focus mode can refresh a track timestamp without a new sighting.
            # Require the actual detection mask that tracking assigned this frame.
            if (track['det_seen'] == ts and track['state'] == 'OK' and inside(track['pos'])
                    and track['mask'] is not None and any(
                        det.mask is track['mask'] and inside(det.c) for det in dets)):
                cue = ('FIXED_BOX_GOLD', ids)

        if cue is None:
            self.hold = None
            self.latched = None
            return None
        if self.pickup is not None:
            original = set(self.pickup['candidate_ids'])
            # Candidate contraction, missed hands, and returning to the same
            # spot cannot open another cycle for this trip. A separate jewel
            # must complete its own grip window before it starts a new cycle.
            if set(ids).issubset(original) and any(views.get(jid) != 'gold' for jid in original):
                self.preview.update(hold_seconds=HOLD_S, state='Pickup trigger already recorded')
                return None
        if self.hold is None or self.hold[0] != cue:
            self.hold = cue, ts
        self.preview['trigger_source'] = 'GOLD' if cue[0] == 'FIXED_BOX_GOLD' else 'PINCH'
        self.preview['hold_seconds'] = min(HOLD_S, max(0., ts-self.hold[1]))
        hand_certain = held and all(len(labels) == 1 and labels == held_labels for _, labels in self.grips)
        named = (len(ids) == 1 and ids[0] in confirmed_ids
                 and (cue[0] == 'FIXED_BOX_GOLD' or hand_certain))
        if self.latched == cue:
            self.preview.update(hold_seconds=HOLD_S, state='Pickup trigger already recorded')
            return None
        if ts - self.hold[1] + 1e-9 < HOLD_S:
            return None
        self.latched = cue
        self.pickup_sequence += 1
        self.pickup = dict(pickup_id=f'{key[0]}:{self.pickup_sequence}',
                           candidate_ids=list(ids), jewel_id=ids[0] if named else None,
                           ownership_candidates=[track['id'] for track in tracks],
                           source=cue[0], timestamp=self.hold[1],
                           single_hand=hand_certain or cue[0] == 'FIXED_BOX_GOLD',
                           trigger_hands=list(held_labels),
                           departed=set(confirmed_ids), last_grip_at=ts,
                           departure_times={jid: self.homes[jid].empty_since for jid in confirmed_ids})
        self.preview['state'] = 'Pickup trigger confirmed' if named else 'Provisional recording; identity uncertain'
        return dict(source=cue[0], jewel_id=ids[0] if named else None,
                    event='pickup', pickup_id=self.pickup['pickup_id'],
                    ownership_candidates=list(self.pickup['ownership_candidates']),
                    candidate_ids=list(ids), timestamp=self.hold[1], confirmed_at=ts, box=list(box),
                    hand_selection=settings.get('hand', 'Any'), trigger_hands=list(held_labels),
                    provisional=not named, hand_model_error=self.hand_error)
