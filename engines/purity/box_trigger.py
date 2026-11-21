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


def draw_hand_preview(frame, ts, preview):
    """Draw fresh diagnostics on a preview copy, leaving recorded pixels intact."""
    if (not preview or tuple(frame.shape[:2]) != preview['shape']
            or not 0 <= ts-preview['timestamp'] <= MAX_SAMPLE_GAP):
        return frame
    image = frame.copy()
    for hand in preview['hands']:
        points = hand['points']
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
            self.hand_error = None
        if self.hands is None and self.hand_error is None:
            try:
                self.hands = self.hand_factory()
            except Exception as exc:
                self.hand_error = str(exc)

    def observe(self, key, frame, ts, tracks, dets, settings):
        self.prepare(key)
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
            for home in self.homes.values():
                home.empty_since = None
        x1, y1, x2, y2 = home_box

        def inside(point):
            x, y = point
            return x1 <= x <= x2 and y1 <= y <= y2

        candidates = []
        confirmed_ids = []
        home_states = []
        for track in tracks:
            jid = track['id']
            home = self.homes.get(jid)
            visible = track['state'] == 'OK' and not track['in_hand'] and track['det_seen'] == ts
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
            home_states.append(f"J{jid} home {view}")
            if view == 'empty':
                if home.empty_since is None:
                    home.empty_since = ts
            else:
                home.empty_since = None
            if view == 'gold':
                home.confirmed_left = False
            if home.empty_since is not None and ts-home.empty_since + 1e-9 >= HOME_EMPTY_S:
                home.confirmed_left = True
            # Covered homes are provisional pickup candidates, as in the standalone.
            # A fresh jewel resting outside the work area cannot take this gesture.
            if view != 'gold' and not (visible and not inside(track['pos'])):
                candidates.append(track)
                if home.confirmed_left:
                    confirmed_ids.append(jid)

        self.preview['detail'] = ', '.join(home_states) or 'No eligible jewel tracks'
        if box is None:
            self.hold = None
            return None
        # The standalone uses a padded crop and measures in source pixels.
        px = max(int(1.5*(x2-x1)), (HAND_CROP_MIN_PX-(x2-x1))//2)
        py = max(int(1.5*(y2-y1)), (HAND_CROP_MIN_PX-(y2-y1))//2)
        height, width = frame.shape[:2]
        cx, cy = max(0, x1-px), max(0, y1-py)
        crop = frame[cy:min(height, y2+py), cx:min(width, x2+px)]
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
        self.preview['state'] = ('Waiting for jewel pickup' if not candidates else
                                 'Hold grip inside stone area' if not grips else 'Checking pickup')
        if candidates and not held and not any(hand['selected'] for hand in self.preview['hands']):
            self.preview['state'] = 'Selected hand not detected'
        if grips > 1:
            self.preview['state'] = 'Multiple qualifying hands; identity uncertain'
        if nearby:
            self.last_hand_near = ts
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
        self.preview['state'] = 'Pickup trigger confirmed' if named else 'Provisional recording; identity uncertain'
        return dict(source=cue[0], jewel_id=ids[0] if named else None,
                    candidate_ids=list(ids), timestamp=self.hold[1], confirmed_at=ts, box=list(box),
                    hand_selection=settings.get('hand', 'Any'), trigger_hands=list(held_labels),
                    provisional=not named, hand_model_error=self.hand_error)
