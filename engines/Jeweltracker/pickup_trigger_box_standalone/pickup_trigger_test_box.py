#!/usr/bin/env python3
"""Pickup trigger test (box): per-jewel video recording with the jewel_tracker_hand.py trigger.

A clip is recorded for each jewel while it is being worked in the target box. Same rules as the
touchstone recording of jewel_tracker_hand.py, with the drawn box in place of the stone:
  * start  - the jewel was picked up off the bench AND, held for REC_START_S:
             1. (main) thumb tip and index tip pinched within HAND_PINCH_PX with the midpoint in the box, or
             2. (second) its gold detection / track is inside the box while NO hand is seen at the box.
             A hand carrying the jewel over the box without pinching, a mere lift, or a pass near the box
             does not start a clip. Without the hand model only rule 2 applies.
  * stop   - set back on the bench and lying still for REC_STOP_SETTLE_S ("placed back"), MISSING /
             LEFT_ROI for REC_STOP_GONE_S ("left"), or REC_MAX_S reached ("max length").
             "Lying still" tolerates a jumpy detection box (position spread over REST_WINDOW_S, not speed).
  * parallel: while one jewel records, another picked-up jewel starts its own clip at the same time when it
             is pinched in the box and its home shows it left, or it is surely seen in the box (tracked there
             or its detection matched / identified as it).
  * one clip per trip: after a clip ends no new one starts until the jewel has rested on the bench.
  * the last REC_PREROLL_FRAMES frames (kept only while a jewel is in hand) open the clip.
Clips are raw frames (no overlay) shrunk to REC_MAX_W wide; a red REC badge shows on the feed.

Draw the box on the video with "Draw box"; it is saved to rec_box.json. Without a saved box the
"stone" zone of the Jewel Tracker zones.json is used. The box must lie inside the bench ROI for the
gold trigger (the detector only reports jewels inside the ROI); the pinch trigger works anywhere.
Tracking is the original jewel_tracker_5 Worker (assignment, SLIC identity, matching, recovery) with
the stone/rubbing/acid workflows bypassed. Run with the project's .venv-slic Python.
--self-test checks the trigger state machine without models or a camera.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    # The editor's Run button / file associations often start the system Python. Its OpenCV has SLIC, so
    # tracking and the box trigger run, but it has no mediapipe: the hand-pinch trigger silently stays off.
    project_python = ROOT/'.venv-slic'/('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if project_python.is_file() and Path(sys.executable).resolve() != project_python.resolve():
        print(f'Launching with project Python: {project_python}', flush=True)
        raise SystemExit(subprocess.call([str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]]))
try:
    TIMEZONE = ZoneInfo('Asia/Kolkata')
except ZoneInfoNotFoundError:
    # India has no daylight-saving transition; Windows may lack tzdata.
    TIMEZONE = timezone(timedelta(hours=5, minutes=30), 'IST')
_local_tracker = ROOT/'jewel_tracker'      # standalone copy: the tracker lives next to this script
if not (_local_tracker/'jewel_tracker_5.py').is_file():
    _local_tracker = ROOT.parent/'jewel_tracker'
TRACKER = Path(os.environ.get('JEWEL_TRACKER_DIR',
                              _local_tracker if (_local_tracker/'jewel_tracker_5.py').is_file()
                              else _local_tracker/'jewel_tracker'))
sys.path.insert(0, str(TRACKER))
_standalone_parent = ROOT/'int_8_GOLD_JEWEL'
if (_standalone_parent/'standalone'/'__init__.py').is_file():
    sys.path.insert(0, str(_standalone_parent))

import cv2
import numpy as np

# ---- per-jewel recording trigger (values from jewel_tracker_hand.py) ----
REC_START_S = 0.5              # "picked up and inside the box" must hold this long before a clip starts
REC_STOP_SETTLE_S = 0.6        # lying still on the bench this long -> the clip stops (set down, not just passing)
# "Lying still" tolerates a jumpy detection box: every position seen in the last REST_WINDOW_S stays within
# max(REST_JITTER_PX, REST_JITTER_RATIO x jewel size) of their median. Placed back -> stop after about
# REST_WINDOW_S + REC_STOP_SETTLE_S (~1 s).
REST_WINDOW_S = 0.4
REST_JITTER_PX = 25
REST_JITTER_RATIO = 0.3
REST_GRACE_S = 0.5             # a placed-back jewel failing the still test this briefly keeps its settle timer
REST_INLIERS = 0.66            # share of those positions that must be within the radius (box jumps = outliers)
REST_FRESH_S = 0.5             # last detection at most this old (a missed frame or two at 5 FPS)
REST_JITTER_MAX_PX = 50        # the jitter radius never grows past this (a slowly carried bangle is not "still")
REST_HAND_MARGIN_PX = 30       # a gripping fingertip this close to the jewel's box -> it is being held, not lying there
HAND_GRIP_TIPS = (4, 8, 12)    # MediaPipe landmarks that hold a jewel: thumb, index and middle fingertips
REC_STOP_GONE_S = 1.5          # MISSING / outside the ROI this long -> the clip stops (a detection flicker is shorter)
REC_MAX_W = 960                # frames are shrunk to this width before writing (small, light files)
REC_MAX_S = 240.0              # safety cap: never record one jewel longer than this
REC_PREROLL_FRAMES = 10        # frames kept before the trigger and written first, so the clip shows the approach
REC_STOPPED_SHOW_S = 4.0       # the "REC STOPPED" badge stays this long after a clip ends
REC_BADGE_COLOR = (0, 0, 230)  # red (BGR)
BOX_COLOR = (0, 165, 255)      # target box overlay (BGR orange)
# Same as jewel_tracker_5 (kept local so --self-test needs no models).
BOX_MARGIN_PX = 40             # STONE_MARGIN_PX: "near the box" for the resting test
REID_AFTER_S = 0.5             # unseen longer than this -> off the bench / in the hand
ID_MIN_PROB = 0.6              # SLIC identity probability needed to credit a box detection to a jewel

# ---- hand pinch: the MAIN recording trigger; gold detection is second ----
# 1. PINCH (priority): MediaPipe thumb tip and index tip closer than HAND_PINCH_PX with their midpoint
#    inside the box, for REC_START_S, while a jewel is picked up -> clip starts.
#    A hand seen at the box but NOT pinched (jewel just carried / hovering over the box) -> no clip.
# 2. GOLD (second): a gold detection / track inside the box starts a clip only when NO hand is seen at
#    the box (the hand model missed it), or when the hand model is off / failed to load.
# The hand model runs only while a jewel is picked up, on a crop around the box, at most every HAND_CHECK_S.
HAND_PINCH_PX = 150            # PINCH TRIGGER DISTANCE: thumb tip - index tip <= this many px (full-frame px;
                               # the live value is printed on the feed as "gap ...px"). Tune it there.
                               # None -> automatic: the largest of the 3 rules below.
PINCH_HAND = 'Right'           # only this hand's pinch counts ('Right' | 'Left' | 'Any'); the other hand is
                               # ignored (no pinch, does not block gold). Uses MediaPipe's handedness label,
                               # shown on the feed as R / L - if your right hand shows "L", choose 'Left'.
PINCH_HANDS = ('Right', 'Left', 'Any')
REC_GOLD_FALLBACK = True       # False -> with the hand model on, ONLY a pinch starts a clip (gold never does)
HAND_NEAR_HOLD_S = 1.0         # a hand seen at the box blocks the gold trigger this long (bridges hand-model misses)
HAND_PINCH_MAX_RATIO = 0.35    # thumb tip - index tip distance / hand size (wrist -> middle knuckle) -> pinched
HAND_PINCH_MIN_PX = 15         # ... but never require the tips closer than this (small / far hands)
HAND_GRIP_SIZE_RATIO = 1.6     # ... or up to this many times the held jewel's width (bangle gripped from above:
                               # measured 220-420 px tip gap for a ~300-370 px bangle at 2560x1440)
HAND_CHECK_S = 0.15            # run the hand model at most this often
HAND_HOLD_S = 0.45             # a pinch seen in the box keeps counting this long (bridges a missed detection)
HAND_CROP_PAD = 1.5            # the hand model sees the box widened by this many box sizes on every side ...
HAND_CROP_MIN_PX = 720         # ... and at least this big: the palm (wrist side) must be in view for MediaPipe

# ---- home position: each jewel's resting spot on the bench (position + its segmented gold mask) ----
# Set when it first rests, re-set when it is placed back after a trip. Every frame its home is looked at,
# independent of the tracker's ids: "gold" (raw gold masks cover it: the jewel is still there), "empty"
# (uncovered, its pixels show white bench: it left) or "covered" (hand / arm over it: unknown).
#   HOME     -> not picked up: can never trigger a clip, whatever the tracker's ids do.
#   COVERED  -> candidate (provisional): may start a clip, which stays provisional.
#   LEFT     -> left its home: confirmed once the home is seen empty; "moved" (its track went away visibly,
#               identity not contradicting) is provisional until then, and undone if gold is seen at home.
# A provisional clip whose jewel turns out still at home is handed to the jewel that really left (renamed);
# if none is known yet it is kept UNASSIGNED and, when a jewel is next placed after moving from its home,
# saved as that jewel's clip (identity checked). Never placed -> saved as UNASSIGNED_... (not deleted).
HOME_GOLD_FULL = 0.4           # share of the home mask covered by raw gold masks -> the jewel is still at home
HOME_WHITE_EMPTY = 0.7         # share of the home mask showing white bench -> uncovered and empty (SPOT_WHITE)
HOME_EMPTY_S = 0.3             # empty this long -> it left (confirmed)
HOME_RESOLVE_S = 0.8           # a clip of a jewel seen still at home waits this long for the one that left
ORPHAN_QUIET_S = 1.0           # an unassigned (J?) clip ends once everything is back at rest this long: no jewel
                               # away from its home, none in a trip, no hand at the box
# Home POSE = its centre of gravity (mask centroid) + its mask. Set down again, it "moved from its home pose"
# when the COG shifted more than max(HOME_MOVED_PX, HOME_MOVED_RATIO x size) OR its mask overlaps the home
# mask less than HOME_SAME_IOU. (Fine on purpose - HOME_LEAVE_* is only the coarse "carried away" test.)
HOME_MOVED_PX = 15
HOME_MOVED_RATIO = 0.12
HOME_SAME_IOU = 0.6
HOME_LEAVE_PX = 40             # visibly moved away: fresh track farther than max(this, ratio x size) from home
HOME_LEAVE_RATIO = 0.6
BG_WHITE_S_MAX, BG_WHITE_V_MIN = 40, 170   # white bench, as jewel_tracker_5
HOME_COLORS = dict(HOME=(80, 200, 80), COVERED=(0, 200, 255), LEFT=(60, 60, 230))

BOX_FILE = ROOT/'rec_box.json'
SETTINGS_FILE = ROOT/'rec_settings.json'   # UI-set values (pinch distance), kept between runs


def set_pinch_px(value):
    """Pinch trigger distance from the UI; 0 / None -> automatic. Takes effect on the next frame."""
    global HAND_PINCH_PX
    HAND_PINCH_PX = int(value) if value else None


def set_pinch_hand(value):
    """Which hand's pinch counts, from the UI ('Right' | 'Left' | 'Any'). Takes effect on the next frame."""
    global PINCH_HAND
    PINCH_HAND = value if value in PINCH_HANDS else 'Right'


def pinch_text():
    hand = 'any hand' if PINCH_HAND == 'Any' else f'{PINCH_HAND.lower()} hand'
    return f'{hand}, ' + ('auto distance' if HAND_PINCH_PX is None else f'<= {HAND_PINCH_PX} px')


def hand_counts(hand):
    """This detected hand is the one whose pinch counts. A model without handedness (None) -> counts."""
    label = getattr(hand, 'handedness', None)
    return PINCH_HAND == 'Any' or label is None or label == PINCH_HAND


def make_hand_model(model_path):
    """standalone HandKeypoints plus MediaPipe's handedness label ('Right' / 'Left') on every hand."""
    from standalone.hand_keypoints import HandKeypoints

    class HandednessKeypoints(HandKeypoints):
        def detect(self, frame):
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self.model.detect(image)
            height, width = frame.shape[:2]
            hands = []
            for i, landmarks in enumerate(result.hand_landmarks):
                points = np.array([(p.x*width, p.y*height) for p in landmarks], np.float32)
                if not np.all(np.isfinite(points)):
                    continue
                points[:, 0] = np.clip(points[:, 0], 0, width-1)
                points[:, 1] = np.clip(points[:, 1], 0, height-1)
                label = ''
                if i < len(result.handedness) and result.handedness[i]:
                    label = result.handedness[i][0].category_name
                hands.append(SimpleNamespace(points=points, handedness=label,
                                             scale=max(1., float(np.linalg.norm(points[0]-points[9])))))
            return hands

    return HandednessKeypoints(model_path)


def json_default(value):
    """Convert NumPy values when writing session records."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def capture_devices():
    """Enumerate Linux capture nodes; exclude webcam metadata-only endpoints."""
    if not sys.platform.startswith('linux'):
        return []
    import fcntl
    devices = []
    for path in sorted(Path('/dev').glob('video[0-9]*')):
        try:
            index = int(path.name[5:])
            data = bytearray(104)  # struct v4l2_capability
            with path.open('rb', buffering=0) as handle:
                fcntl.ioctl(handle, 0x80685600, data, True)  # VIDIOC_QUERYCAP
            _, capabilities, device_caps = struct.unpack_from('III', data, 80)
            caps = device_caps if capabilities & 0x80000000 else capabilities
            if caps & (0x1 | 0x1000):  # VIDEO_CAPTURE / VIDEO_CAPTURE_MPLANE
                name = bytes(data[16:48]).split(b'\0')[0].decode(errors='replace')
                devices.append((index, name))
        except (OSError, ValueError):
            continue
    return devices


def small_frame(frame, width=REC_MAX_W):
    """The frame shrunk to `width` (display and clip frames; keeps files small and writing cheap)."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, max(1, round(h*width/w))), interpolation=cv2.INTER_AREA) if w > width else frame


def resolve_box(shape, reference_size, zones, cli_box=None):
    """Target box in capture pixels: --box, else rec_box.json, else the zones.json stone. (box, source)."""
    h, w = shape[:2]
    if cli_box is not None:
        return tuple(map(float, cli_box)), 'command line'
    saved = load_json(BOX_FILE)
    if saved.get('box') and saved.get('size'):
        sx, sy = w/saved['size'][0], h/saved['size'][1]
        b = saved['box']
        return (b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy), BOX_FILE.name
    stone = zones.get('stone') or []
    if len(stone) >= 3:
        points = np.asarray(stone, float)*[w/reference_size[0], h/reference_size[1]]
        (x1, y1), (x2, y2) = points.min(0), points.max(0)
        return (x1, y1, x2, y2), 'zones.json stone'
    return None, ''


@dataclass
class RecState:
    """Per-jewel recording state (the rec_* fields of jewel_tracker_hand.py tracks)."""
    picked: bool = False             # off the bench since it last rested: a trip is under way
    lock: bool = False               # a clip just ended: no new clip until the jewel rests (one clip per trip)
    rested: bool = False             # seen lying still on the bench at least once
    arm_since: float | None = None   # picked + inside the box since
    rest_since: float | None = None  # recording and lying still on the bench since
    rest_last: float = -math.inf     # last time it passed the still test (REST_GRACE_S bridges jumpy frames)
    hist: deque = field(default_factory=lambda: deque(maxlen=60))  # (t, pos) of fresh detections
    gone_since: float | None = None  # recording and MISSING / LEFT_ROI since
    writer: object = None
    path: Path | None = None
    start_t: float = 0.
    size: float | None = None        # max(w, h) of its box while resting on the bench (capture px)
    count: int = 0
    last_clip: str = ''
    home: object = None              # SimpleNamespace(pos, box, mask, size) of its resting spot, or None
    home_state: str = ''             # '' (no home yet) | 'HOME' | 'COVERED' | 'LEFT'
    left_by: str = ''                # 'empty' (confirmed: home seen empty) | 'moved' (track went away; provisional)
    left_t: float = -math.inf
    home_view: str = ''              # last look at the home: 'gold' | 'empty' | 'covered'
    empty_since: float | None = None
    provisional: bool = False        # the running clip started before its trip was confirmed
    pending_clip: Path | None = None  # saved while still provisional: renamed / deleted once resolved
    unresolved: bool = False         # seen still at home with a provisional clip: waiting to hand it over
    revert_t: float = 0.
    calm_t: float = -math.inf        # last frame it was lying still (a gap > REST_GRACE_S -> "set down" again)
    last_move: float = -math.inf     # last time it was set down at a NEW pose with no clip of its own
    box_gold_t: float = -math.inf    # last time gold credited to it was seen inside the box (it really went there)
    home_t: float = -math.inf        # when it was last set down at a NEW pose (its current home was made)

    def confirmed_left(self):
        return self.home_state == 'LEFT' and self.left_by == 'empty'


def make_home(tr):
    """Home of a resting track from its own segmented mask, or None (no mask -> no home logic)."""
    mask = getattr(tr, 'det_mask', None)
    if mask is None:
        return None
    mask = np.asarray(mask) > 0
    if not mask.any():
        return None
    ox, oy = (int(v) for v in getattr(tr, 'det_origin', tr.det_box[:2]))
    h, w = mask.shape[:2]
    return SimpleNamespace(pos=np.array(tr.pos, float), box=(ox, oy, ox+w, oy+h), mask=mask, size=float(max(w, h)))


def mask_iou(mask_a, origin_a, mask_b, origin_b):
    """IoU of two masks given in frame coordinates by their top-left origins."""
    a, b = np.asarray(mask_a) > 0, np.asarray(mask_b) > 0
    (ax, ay), (bx, by) = (tuple(int(v) for v in origin_a), tuple(int(v) for v in origin_b))
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+a.shape[1], bx+b.shape[1]), min(ay+a.shape[0], by+b.shape[0])
    inter = 0
    if x2 > x1 and y2 > y1:
        inter = int(np.count_nonzero(a[y1-ay:y2-ay, x1-ax:x2-ax] & b[y1-by:y2-by, x1-bx:x2-bx]))
    union = int(np.count_nonzero(a))+int(np.count_nonzero(b))-inter
    return inter/union if union else 0.


def pose_change(home, tr):
    """(moved, cog_shift_px, mask_iou) of a track now lying still, against its home pose."""
    shift = float(np.hypot(*(np.asarray(tr.pos, float)-home.pos)))
    mask = getattr(tr, 'det_mask', None)
    iou = 1.
    if mask is not None:
        iou = mask_iou(home.mask, home.box[:2], mask, getattr(tr, 'det_origin', tr.det_box[:2]))
    moved = shift > max(HOME_MOVED_PX, HOME_MOVED_RATIO*home.size) or iou < HOME_SAME_IOU
    return moved, shift, iou


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


class BoxRecorder:
    """Per-jewel background recording, the update_recording trigger of jewel_tracker_hand.py.

    `tracker` is the original Worker (tracks, roi_dist, id_model, identify); `dets` are its raw
    detections of this frame. `hands` is an optional HandKeypoints (detect(frame) -> points/scale).
    """
    def __init__(self, folder, fps, emit, hands=None, box=None):
        self.folder, self.fps, self.emit, self.hands = Path(folder), float(fps), emit, hands
        self.box = None
        self.set_box(box)
        self.rec = {}
        self.preroll = deque(maxlen=REC_PREROLL_FRAMES)
        self.stopped_note = None      # (text, t) of the last clip that ended, for the on-feed badge
        self.last_t = 0.
        self.hand_check_t = -math.inf
        self.hand_pinch_t = None
        self.hand_vis, self.hand_vis_t = [], -math.inf
        self.hand_near_t = -math.inf    # last time any hand landmark was seen at the box
        self.box_moved = False
        self.hand_error = ''
        self.grips = deque(maxlen=32)   # (t, thumb-index gap px, hand scale px) with the midpoint in the box
        self.vis_size = None            # largest picked-up jewel, for the overlay's grip colour
        self.orphans = []               # unassigned clips (provisional jewel was still at home): see update_orphans
        self.loose = []                 # raw detections no track took (MISSING jewel set down): see loose_still
        self.loose_owners = {}          # {track id: where its unclaimed gold lies} this frame (for the overlay)

    def set_box(self, box):
        self.box_moved = True
        if box is None:
            self.box = None
            return
        x1, x2 = sorted(int(round(v)) for v in (box[0], box[2]))
        y1, y2 = sorted(int(round(v)) for v in (box[1], box[3]))
        self.box = (x1, y1, x2, y2) if x2-x1 >= 4 and y2-y1 >= 4 else None

    def forget_box_jewels(self, tracks):
        """A box (re)drawn over jewels lying there must not count them as 'carried into the box':
        they lose their rest-on-bench credit until they are seen resting on the bench again."""
        for tr in tracks:
            rec = self.state(tr)
            if rec.writer is None and (self.on_box(tr.pos) or (rec.home is not None and self.on_box(rec.home.pos))):
                rec.rested = rec.picked = False
                rec.arm_since = None
                rec.home, rec.home_state, rec.left_by = None, '', ''

    def on_box(self, p, margin=BOX_MARGIN_PX):
        b = self.box
        return b is not None and b[0]-margin <= p[0] <= b[2]+margin and b[1]-margin <= p[1] <= b[3]+margin

    def state(self, tr):
        return self.rec.setdefault(tr.id, RecState())

    def resting(self, tr, t):
        """Seen just now, on the bench (outside the box) and lying still: most positions of the last
        REST_WINDOW_S (REST_INLIERS), and the latest one, within the jitter radius of their median. Position
        spread, not speed, and outliers allowed - a flickering / jumping detection box still counts as lying
        still. A missed detection or two (REST_FRESH_S) does not break it either."""
        if not (t-tr.det_seen <= REST_FRESH_S and tr.state == 'OK' and not self.on_box(tr.pos, margin=0)):
            return False
        if self.hand_on(tr.det_box, t):   # held still in the hand (hovering, about to go into the box) - not set down
            return False
        return self.still_points(self.state(tr).hist, t, tr.det_box)

    @staticmethod
    def still_points(hist, t, box):
        """(t, pos) history lying still: only the samples of the last REST_WINDOW_S (never older ones across a
        gap: something that just appeared elsewhere must first be seen there for the whole window), at least
        3 of them, most (REST_INLIERS) and the latest within the jitter radius of their median."""
        recent = [(when, p) for when, p in hist if t-when <= REST_WINDOW_S+1e-6]
        if len(recent) < 3 or t-min(when for when, _ in recent) < .75*REST_WINDOW_S:
            return False
        points = np.asarray([p for _, p in reversed(recent)], float)   # points[0] is the latest
        x1, y1, x2, y2 = map(float, box)
        radius = min(REST_JITTER_MAX_PX, max(REST_JITTER_PX, REST_JITTER_RATIO*max(x2-x1, y2-y1)))
        near = np.hypot(*(points-np.median(points, 0)).T) <= radius
        return bool(near[0] and near.mean() >= REST_INLIERS)

    def loose_still(self, tracker, dets, frame, t):
        """Set down while the tracker still calls it MISSING: the original tracker does not always re-attach a
        jewel's detection after it was hidden in the hand, so its raw gold box (grey on the feed) lies there
        unclaimed. Raw detections no fresh track took are followed here; one lying still (no hand on it, not on
        the home of a jewel lying at home) is credited to a jewel that is out of sight and in a trip:
        its identity when confident, else the only such jewel recording (or the only one in a trip).
        Returns {track id: pseudo track (pos, det_box, det_mask, det_origin) of where it was set down}."""
        taken = {tuple(map(int, tr.det_box)) for tr in tracker.tracks if tr.det_seen == t}
        free = [d for d in dets if tuple(map(int, d.box)) not in taken and not self.on_box(d.c, margin=0)]
        for cl in self.loose:
            cl.matched = False
        for d in free:
            open_ = [cl for cl in self.loose if not cl.matched]
            best = min(open_, key=lambda cl: float(np.hypot(*(cl.c-d.c))), default=None)
            if best is not None and float(np.hypot(*(best.c-d.c))) <= 2*REST_JITTER_MAX_PX:
                best.hist.append((t, np.array(d.c, float))); best.c, best.det, best.seen = np.array(d.c, float), d, t
                best.matched = True
            else:
                self.loose.append(SimpleNamespace(hist=deque([(t, np.array(d.c, float))], maxlen=30),
                                                  c=np.array(d.c, float), det=d, seen=t, matched=True))
        self.loose = [cl for cl in self.loose if t-cl.seen <= 1.]
        away = [tr for tr in tracker.tracks if t-tr.det_seen > REST_FRESH_S
                and (self.state(tr).writer is not None or self.state(tr).picked)]
        out = {}
        for cl in self.loose:
            if not away or t-cl.seen > REST_FRESH_S or not self.still_points(cl.hist, t, cl.det.box):
                continue
            d = cl.det
            if self.hand_on(d.box, t):
                continue                       # still held
            mask = getattr(d, 'mask', None)
            pool = [tr for tr in away if tr.id not in out]
            at_home = [tid for tid, rec in self.rec.items() if mask is not None and rec.home is not None
                       and rec.home_state == 'HOME'
                       and mask_iou(rec.home.mask, rec.home.box[:2], mask, d.origin) >= HOME_SAME_IOU]
            if any(tid not in {tr.id for tr in pool} for tid in at_home):
                continue                       # another jewel lying at its own home (its track merely lost)
            jid, p = self.identity(tracker, frame, d)
            owner = None
            if at_home:                        # the jewel already set down here (its new home): same one
                owner = next(tr for tr in pool if tr.id == at_home[0])
            elif jid is not None and p >= ID_MIN_PROB:
                owner = next((tr for tr in pool if tr.id == jid), None)   # identified: that jewel or nobody
            else:
                recording = [tr for tr in pool if self.state(tr).writer is not None]
                if len(recording) == 1:
                    owner = recording[0]
                elif not recording and len(pool) == 1:
                    owner = pool[0]
            if owner is not None:
                out[owner.id] = SimpleNamespace(id=owner.id, pos=np.array(d.c, float), det_box=d.box,
                                                det_mask=mask, det_origin=getattr(d, 'origin', d.box[:2]))
        return out

    def hand_on(self, box, t):
        """Any hand (either hand) seen in the last HAND_HOLD_S HOLDING `box`: a gripping fingertip (thumb, index,
        middle) on / right next to it. The palm or wrist merely passing over a jewel lying there (the hand
        reaching for the next jewel or into the box) is not holding it."""
        if not self.hand_vis or not 0. <= t-self.hand_vis_t <= HAND_HOLD_S:
            return False
        x1, y1, x2, y2 = (float(v) for v in box)
        m = REST_HAND_MARGIN_PX
        for entry in self.hand_vis:
            p = entry[0][list(HAND_GRIP_TIPS)] if len(entry[0]) > max(HAND_GRIP_TIPS) else entry[0]
            if np.any((p[:, 0] >= x1-m) & (p[:, 0] <= x2+m) & (p[:, 1] >= y1-m) & (p[:, 1] <= y2+m)):
                return True
        return False

    def inside_box(self, tr):
        """Its centre is actually inside the box (no margin). This is what starts a recording."""
        return self.on_box(tr.pos, margin=0)

    # ---- home position ----
    @staticmethod
    def leave_radius(home):
        return max(HOME_LEAVE_PX, HOME_LEAVE_RATIO*home.size)

    def on_other_home(self, tr):
        """Its track is ON the pose of another jewel that has not left its home (its mask matches that home):
        the tracker put this id on that jewel (e.g. it lay under the hand that lifted this one). Mask overlap,
        not the home's bounding box - a jewel set down next to (or inside the ring of) a bangle is no match."""
        mask = getattr(tr, 'det_mask', None)
        for tid, rec in self.rec.items():
            if tid == tr.id or rec.home is None or rec.home_state == 'LEFT':
                continue
            if mask is not None:
                if mask_iou(rec.home.mask, rec.home.box[:2], mask,
                            getattr(tr, 'det_origin', tr.det_box[:2])) >= HOME_SAME_IOU:
                    return True
            elif float(np.hypot(*(np.asarray(tr.pos, float)-rec.home.pos))) <= max(HOME_MOVED_PX,
                                                                                    HOME_MOVED_RATIO*rec.home.size):
                return True
        return False

    @staticmethod
    def matched_det(tr, dets, t):
        if tr.det_seen != t:
            return None
        box = tuple(map(int, tr.det_box))
        return next((d for d in dets if tuple(map(int, d.box)) == box), None)

    @staticmethod
    def identity(tracker, frame, d):
        """(jewel id, probability) from the SLIC identity model, or (None, 0) when unavailable."""
        model = getattr(tracker, 'id_model', None)
        if d is None or frame is None or model is None or not getattr(model, 'ready', False):
            return None, 0.
        try:
            jid, p = tracker.identify(frame, d)
            return jid, float(p or 0.)
        except Exception:
            return None, 0.

    def moved_away(self, tracker, tr, rec, dets, frame, t):
        """Its track, freshly detected, is clearly away from home - and the identity model does not say
        that detection is another jewel (an id swap with a jewel lying under the hand)."""
        if t-tr.det_seen >= .3 or float(np.hypot(*(tr.pos-rec.home.pos))) <= self.leave_radius(rec.home):
            return False
        jid, p = self.identity(tracker, frame, self.matched_det(tr, dets, t))
        return not (jid is not None and jid != tr.id and p >= ID_MIN_PROB)

    def update_home(self, tracker, tr, rec, dets, frame, t):
        """Home state from the look of the home (and, provisionally, the track). Returns 'reverted' when a
        candidate / provisionally-left jewel is seen still at home, 'confirmed' when it is seen to have left."""
        view = rec.home_view = home_view(rec.home, dets, frame)
        before, was_confirmed = rec.home_state, rec.confirmed_left()
        if rec.home_state == 'LEFT':
            if rec.left_by == 'moved':
                if view == 'empty':
                    rec.left_by = 'empty'
                elif view == 'gold':
                    rec.home_state, rec.left_by = 'HOME', ''   # the gold that moved away was another jewel
        else:
            if view == 'gold':
                rec.home_state, rec.empty_since = 'HOME', None
            elif view == 'empty':
                rec.empty_since = t if rec.empty_since is None else rec.empty_since
                rec.home_state = 'COVERED'
                if t-rec.empty_since >= HOME_EMPTY_S:
                    rec.home_state, rec.left_by, rec.left_t = 'LEFT', 'empty', t
            else:
                rec.home_state, rec.empty_since = 'COVERED', None
            if rec.home_state != 'LEFT' and view != 'gold' and self.moved_away(tracker, tr, rec, dets, frame, t):
                rec.home_state, rec.left_by, rec.left_t = 'LEFT', 'moved', t
        if rec.home_state == 'HOME' and before in ('COVERED', 'LEFT'):
            return 'reverted'
        if rec.confirmed_left() and not was_confirmed:
            return 'confirmed'
        return ''

    def set_home(self, tr, rec, t=None):
        home = make_home(tr)
        if home is not None:
            rec.home, rec.home_state, rec.left_by, rec.empty_since = home, 'HOME', '', None
            if t is not None:
                rec.home_t = t

    def final_path(self, path, tid):
        """The clip's file name with the jewel it finally belongs to (J<id>_...)."""
        rest = path.name.split('_', 1)[1] if '_' in path.name else path.name
        return path.with_name(f'J{tid}_{rest}')

    def rename_clip(self, path, tid):
        target = self.final_path(path, tid)
        if target != path:
            try:
                os.replace(path, target)
            except OSError:
                return path
        return target

    def take_over_target(self, exclude):
        """The jewel that really left its home and has no clip for this trip yet (most recently left)."""
        options = [(rec.left_t, tid) for tid, rec in self.rec.items()
                   if tid != exclude and rec.confirmed_left() and rec.writer is None and not rec.lock]
        return max(options)[1] if options else None

    def still_home(self, tid, t):
        """Jewel `tid` was a candidate but is seen still at its home: its clip (if any) was not about it.
        The clip waits up to HOME_RESOLVE_S for the jewel that really left to be confirmed (its home must be
        seen empty for HOME_EMPTY_S), then is handed over to it - or kept unassigned until a jewel is placed."""
        rec = self.rec[tid]
        rec.picked = rec.lock = False
        rec.arm_since = None
        if rec.writer is not None or rec.pending_clip is not None:
            rec.unresolved, rec.revert_t = True, t

    def others_away(self, tid):
        """Another jewel is in a trip right now: recording, picked up, or its home not showing it lying there."""
        return any(other != tid and (rec.writer is not None or rec.picked or rec.unresolved
                                     or (rec.home is not None and rec.home_state != 'HOME'))
                   for other, rec in self.rec.items())

    def went_and_back(self, tid):
        """Jewel `tid`, seen back at its home pose, really made the trip: gold credited to it was seen in the box
        during this clip and no other jewel is away to explain that gold. (Its home was covered the whole time -
        a forearm - so it was never seen empty; it was put back on the very spot it came from.)"""
        rec = self.rec[tid]
        return rec.box_gold_t >= rec.start_t-REC_START_S-.5 and not self.others_away(tid)

    def scene_quiet(self, t):
        """Everything back at rest: no jewel in a trip or away from its home, and no hand at the box."""
        if self.hand_vis and 0. <= t-self.hand_vis_t <= HAND_HOLD_S:
            return False
        if self.hands is not None and 0. <= t-self.hand_near_t <= HAND_NEAR_HOLD_S:
            return False
        return not any(rec.writer is not None or rec.picked or rec.unresolved
                       or (rec.home is not None and rec.home_state != 'HOME') for rec in self.rec.values())

    def resolve_pending(self, t):
        for tid, rec in list(self.rec.items()):
            if rec.unresolved and (self.take_over_target(tid) is not None or self.went_and_back(tid)
                                   or t-rec.revert_t >= HOME_RESOLVE_S):
                rec.unresolved = False
                self.resolve_still_home(tid, t)

    def resolve_still_home(self, tid, t):
        rec = self.rec[tid]
        rec.provisional = False
        if rec.writer is None and rec.pending_clip is None:
            return
        target = self.take_over_target(tid)
        if rec.writer is not None and target is not None:      # live clip -> hand it over
            other = self.rec[target]
            other.writer, other.path, other.start_t = rec.writer, rec.path, rec.start_t
            other.provisional, other.arm_since, other.rest_since, other.gone_since = False, None, None, None
            rec.writer = rec.path = None
            self.emit(dict(time=t, jewel=f'J{target}', state='REC_REASSIGNED', from_jewel=f'J{tid}',
                           reason=f'J{tid} still at its home; J{target} left its home'))
        elif target is None and self.went_and_back(tid):        # it went to the box and came back to its spot
            reason = 'placed back on its home spot (its gold was in the box)'
            if rec.writer is not None:
                self.stop(tid, reason, t)
            else:
                path, rec.pending_clip = rec.pending_clip, None
                rec.last_clip, rec.lock = path.name, True
                self.emit(dict(time=t, jewel=f'J{tid}', state='REC_CONFIRMED', clip=path.name, reason=reason))
        elif rec.writer is not None:                            # nobody confirmed gone yet -> keep it unassigned
            self.orphans.append(SimpleNamespace(writer=rec.writer, path=rec.path, start_t=rec.start_t,
                                                from_id=tid, since=t, quiet_since=None, left=set(),
                                                gold=rec.box_gold_t >= rec.start_t-REC_START_S-.5))
            rec.writer = rec.path = None
            self.emit(dict(time=t, jewel=f'J{tid}', state='REC_UNASSIGNED',
                           reason=f'J{tid} still at its home - clip kept; it goes to the jewel that moved '
                                  'from its home once that jewel is placed'))
            self.stopped_note = (f'REC J{tid} -> unassigned (J{tid} still at home)', t)
        else:                                                   # already saved while provisional
            path, rec.pending_clip = rec.pending_clip, None
            if target is not None:
                new = self.rename_clip(path, target)
                self.rec[target].lock, self.rec[target].last_clip = True, new.name
                self.emit(dict(time=t, jewel=f'J{target}', state='REC_REASSIGNED', from_jewel=f'J{tid}',
                               clip=new.name, reason=f'J{tid} still at its home; J{target} left its home'))
            else:
                rec.last_clip = ''
                self.orphans.append(SimpleNamespace(writer=None, path=path, start_t=t, from_id=tid, since=t,
                                                    quiet_since=None, left=set(), gold=False))
                self.emit(dict(time=t, jewel=f'J{tid}', state='REC_UNASSIGNED', clip=path.name,
                               reason=f'J{tid} still at its home - clip kept; it goes to the jewel that moved '
                                      'from its home once that jewel is placed'))

    # ---- unassigned clips: a provisional clip whose jewel was still at home ----
    def orphan_owner(self, tracker, tr, dets, frame, t):
        """Jewel just placed after moving from its home: the identity of its detection when confident,
        else its track id."""
        jid, p = self.identity(tracker, frame, self.matched_det(tr, dets, t))
        return jid if jid is not None and p >= ID_MIN_PROB and jid in self.rec else tr.id

    def assign_orphan(self, orphan, tid, t, reason):
        """Close the unassigned clip (if still recording) and save it as jewel `tid`'s clip."""
        if orphan.writer is not None:
            try:
                orphan.writer.release()
            except Exception:
                pass
            orphan.writer = None
        path = self.rename_clip(orphan.path, tid)
        self.rec[tid].last_clip = path.name
        self.emit(dict(time=t, jewel=f'J{tid}', state='REC_REASSIGNED', from_jewel=f'J{orphan.from_id}',
                       clip=path.name, reason=reason))
        self.emit(dict(time=t, jewel=f'J{tid}', state='REC_SAVED', reason=reason, clip=path.name,
                       duration_s=round(t-orphan.start_t, 3)))
        self.stopped_note = (f'REC SAVED J{tid} ({reason})', t)

    def save_orphan_unassigned(self, orphan, t, reason):
        if orphan.writer is not None:
            try:
                orphan.writer.release()
            except Exception:
                pass
            orphan.writer = None
        rest = orphan.path.name.split('_', 1)[1] if '_' in orphan.path.name else orphan.path.name
        target = orphan.path.with_name(f'UNASSIGNED_{rest}')
        try:
            os.replace(orphan.path, target)
        except OSError:
            target = orphan.path
        self.emit(dict(time=t, jewel='', state='REC_SAVED_UNASSIGNED', clip=target.name, reason=reason,
                       from_jewel=f'J{orphan.from_id}', duration_s=round(t-orphan.start_t, 3)))

    def update_orphans(self, tracker, placed, dets, frame, t):
        """Hand an unassigned clip to a jewel confirmed gone from its home (live), or - once a jewel is placed
        after moving from its home - save it as that jewel's clip. No placement within REC_MAX_S: saved as
        UNASSIGNED_..."""
        for orphan in list(self.orphans):
            target = self.take_over_target(orphan.from_id)
            if orphan.writer is not None and target is not None:
                other = self.rec[target]
                other.writer, other.path, other.start_t = orphan.writer, orphan.path, orphan.start_t
                other.provisional, other.arm_since, other.rest_since, other.gone_since = False, None, None, None
                self.orphans.remove(orphan)
                self.emit(dict(time=t, jewel=f'J{target}', state='REC_REASSIGNED', from_jewel=f'J{orphan.from_id}',
                               reason=f'J{target} left its home'))
                continue
            # A jewel set down at a new pose since this clip started - now, or already earlier (e.g. while the
            # wrong jewel's clip was still waiting): it is the one the clip shows.
            moved = [(self.rec[tr.id].last_move, 1, tr.id, tr) for tr in placed if tr.id != orphan.from_id]
            moved += [(rec.last_move, 0, tid, None) for tid, rec in self.rec.items()
                      if tid != orphan.from_id and rec.last_move >= orphan.start_t and rec.writer is None
                      and tid not in {tr.id for tr in placed}]
            if moved:
                _, _, tid, tr = max(moved, key=lambda m: m[:3])
                if tr is not None:
                    tid = self.orphan_owner(tracker, tr, dets, frame, t)
                placed[:] = [p for p in placed if p.id != tid]
                self.rec[tid].last_move = -math.inf     # used up: one clip per placement
                self.orphans.remove(orphan)
                self.assign_orphan(orphan, tid, t, f'placed - J{tid} moved from its home pose')
                continue
            # Nobody set down at a new pose, but everything is back at rest (a jewel put back on the very spot
            # it came from looks unmoved): end the clip instead of recording on - it goes to the one jewel that
            # left its home during it, else to the jewel whose gold was in the box, else UNASSIGNED_.
            orphan.left |= {tid for tid, rec in self.rec.items()
                            if tid != orphan.from_id and rec.home is not None and rec.home_state == 'LEFT'}
            quiet = self.scene_quiet(t)
            orphan.quiet_since = (t if orphan.quiet_since is None else orphan.quiet_since) if quiet else None
            if orphan.quiet_since is not None and t-orphan.quiet_since >= ORPHAN_QUIET_S:
                self.orphans.remove(orphan)
                if len(orphan.left) == 1:
                    tid = next(iter(orphan.left))
                    self.assign_orphan(orphan, tid, t, f'all back at rest - J{tid} left its home during the clip')
                elif not orphan.left and orphan.gold and orphan.from_id in self.rec:
                    self.assign_orphan(orphan, orphan.from_id, t,
                                       f'all back at rest - J{orphan.from_id} was in the box and is back on its spot')
                else:
                    self.save_orphan_unassigned(orphan, t, 'all back at rest - no jewel placed away from its home')
                continue
            if t-orphan.start_t >= REC_MAX_S:
                self.orphans.remove(orphan)
                self.save_orphan_unassigned(orphan, t, 'no jewel was placed away from its home')

    def arrivals(self, tracker, dets, frame, t):
        """(ids, sure): track ids with a RAW detection inside the box right now, each credited to the jewel it
        belongs to; `sure` are those credited by the tracker's own match or a confident SLIC identity (not a
        guess), needed to start a parallel clip. Only picked-up jewels are considered, so a jewel lying in the
        box cannot trigger another one."""
        if self.box is None or not dets:
            return set(), set()
        inside = [d for d in dets if self.on_box(d.c, margin=0) and tracker.roi_dist(d.c) > 0]
        picked = [tr for tr in tracker.tracks if self.state(tr).picked and not self.resting(tr, t)]
        if not inside or not picked:
            return set(), set()
        out, sure = set(), set()
        for d in inside:
            # Unlike the stone, the tracker keeps following jewels inside the box: a detection it
            # attached to a track this frame belongs to that jewel, picked up or not.
            box = tuple(map(int, d.box))
            matched = next((tr for tr in tracker.tracks if tr.det_seen == t and
                            tuple(map(int, tr.det_box)) == box), None)
            if matched is not None:
                if matched in picked:
                    out.add(matched.id); sure.add(matched.id)
                    continue
                mrec = self.state(matched)
                if mrec.home is None or mrec.home_state != 'HOME':
                    continue                  # e.g. lying in the box from the start: never an arrival
                # Matched to a jewel whose home still shows it lying there: the tracker's id is wrong for this
                # gold in the box - it is the jewel that really left. Credit it like an unmatched detection.
            # Gold reappeared in the box: its identity first (step 5 of the home logic).
            jid, p = self.identity(tracker, frame, d)
            if jid is not None and p >= ID_MIN_PROB:
                owner = next((tr for tr in picked if tr.id == jid), None)
                if owner is not None:
                    out.add(owner.id); sure.add(owner.id)
                    continue
                if self.rec.get(jid) is not None and self.rec[jid].home_state == 'HOME':
                    continue                  # confidently a jewel still lying at home: credit nobody
            if len(picked) == 1:              # one jewel in hand: the box detection is its arrival
                out.add(picked[0].id)
                continue
            # several in hand: a jewel confirmed to have left its home first, then the nearest
            owner = min(picked, key=lambda tr: (not (self.state(tr).home is None or self.state(tr).confirmed_left()),
                                                float(np.hypot(*(d.c-tr.pos)))))
            out.add(owner.id)
        return out, sure

    def seen_in_box(self, tr, t):
        """Tracked inside the box from a detection of this moment (not a stale last position)."""
        return t-tr.det_seen < .3 and self.inside_box(tr)

    def update(self, tracker, dets, frame, t):
        """Start a clip only when a picked-up jewel is brought inside the box (debounced), write while it is
        there, stop when it is set back on the bench, has gone, or hits the cap. One resize/write per frame."""
        if frame is None:
            return
        self.last_t = t
        tracks = list(tracker.tracks)
        for tr in tracks:
            self.state(tr)
        if self.box_moved:
            self.box_moved = False
            self.forget_box_jewels(tracks)
        for tr in tracks:
            hist = self.rec[tr.id].hist
            if t-tr.det_seen < .3 and (not hist or hist[-1][0] != t):
                hist.append((t, np.array(tr.pos, float)))
        picked = any(self.rec[tr.id].picked for tr in tracks)
        active = any(self.rec[tr.id].writer is not None for tr in tracks) or any(o.writer for o in self.orphans)
        small = None
        if picked or active:
            small = small_frame(frame)
            if picked:                        # fill the pre-roll only while something is being carried
                self.preroll.append(small)
        else:
            self.preroll.clear()
        dets = dets or []
        if any(rec.picked or rec.writer is not None for rec in self.rec.values()) or self.orphans:
            self.hand_pinch_in_box(frame, t, self.vis_size)   # fresh hands BEFORE the still test (held vs set down)
        resting, reverted, confirmed, placed = {}, [], [], []
        loose = self.loose_still(tracker, dets, frame, t)   # MISSING jewels set down: their unclaimed gold
        self.loose_owners = loose
        for tr in tracks:
            rec = self.rec[tr.id]
            here = loose.get(tr.id) or tr      # where it lies: its track, or its unclaimed raw gold box
            still = here is not tr or self.resting(tr, t)
            near_home = rec.home is not None and float(np.hypot(*(here.pos-rec.home.pos))) <= self.leave_radius(rec.home)
            if still and rec.home is not None:
                # On the pose of another jewel still lying at its home, or far away while its own home still
                # shows it lying there: that is the tracker's id on ANOTHER jewel, not this one resting.
                if self.on_other_home(here) or (not near_home and rec.home_state == 'HOME'):
                    still = False
            resting[tr.id] = still
            settling = still and t-rec.calm_t > REST_GRACE_S   # it has just come to rest (set down)
            if still:
                rec.calm_t = t
            new_pose = rec.home is None
            if settling and rec.home is not None:
                moved, shift, iou = pose_change(rec.home, here)   # COG + mask against the home it had before
                new_pose = moved
                if moved or rec.home_state == 'LEFT':
                    if rec.writer is None and not rec.lock:
                        placed.append(tr)      # set down after moving from its home pose, no clip of its own
                        rec.last_move = t
                    if moved:
                        self.emit(dict(time=t, jewel=f'J{tr.id}', state='HOME_MOVED', cog_shift_px=round(shift, 1),
                                       mask_iou=round(iou, 2), reason='set down at a new pose: new home'))
                    if moved and (rec.pending_clip is not None or (rec.writer is not None and rec.provisional)):
                        # Its provisional clip: it was carried and set down somewhere new - it really left.
                        clip = rec.pending_clip.name if rec.pending_clip is not None else rec.path.name
                        rec.pending_clip, rec.provisional = None, False
                        self.emit(dict(time=t, jewel=f'J{tr.id}', state='REC_CONFIRMED', clip=clip,
                                       reason=f'J{tr.id} set down away from its home: it really left'))
                elif not rec.confirmed_left() and rec.home_state in ('COVERED', 'LEFT'):
                    reverted.append(tr.id)     # same pose as its home, never seen gone: it did not leave
            if still:
                if rec.home is None or settling:
                    # first rest, or set down (after a trip / at a new spot / nudged); home_t only for a new pose
                    self.set_home(here, rec, t if new_pose else None)
                else:
                    rec.home_state, rec.left_by, rec.empty_since = 'HOME', '', None
                rec.picked = rec.lock = False  # settled on the bench: trip over, a new trip must pick it up again
                rec.rested = True
                x1, y1, x2, y2 = map(float, here.det_box)
                rec.size = max(x2-x1, y2-y1)   # its width lying flat: what a grip around it spans
            elif rec.home is not None:
                result = self.update_home(tracker, tr, rec, dets, frame, t)
                (reverted if result == 'reverted' else confirmed if result == 'confirmed' else []).append(tr.id)
                rec.picked = rec.home_state in ('COVERED', 'LEFT')   # only a jewel no longer seen at home
            elif (tr.state in ('OCCLUDED', 'RUBBING', 'MISSING') or t-tr.det_seen > REID_AFTER_S
                  # Box adaptation: the tracker keeps following a jewel carried into the box in plain view
                  # (over the stone it does not), so a jewel that rested on the bench and is now inside the
                  # box was picked up. A jewel lying in the box from the start never rested: no trigger.
                  or (rec.rested and self.inside_box(tr))):
                rec.picked = True              # off the bench / in the hand -> it was picked up (no home known)
        for tid in confirmed:
            rec = self.rec[tid]
            if rec.writer is not None and rec.provisional:
                rec.provisional = False
                self.emit(dict(time=t, jewel=f'J{tid}', state='REC_CONFIRMED', clip=rec.path.name,
                               reason=f'J{tid} home seen empty: it really left'))
            rec.pending_clip = None
        for tid in reverted:
            rec = self.rec[tid]
            if rec.home_t >= rec.start_t and (rec.writer is not None or rec.pending_clip is not None):
                # Its home is the spot it was SET DOWN on during this clip; a hand passing over it (covered,
                # id lost) and gold showing there again is that jewel still lying where it was put - not a
                # "never left" candidate: its clip is never handed to another jewel for that.
                rec.pending_clip = None        # (a live clip stops below: lying on its set-down spot)
                rec.picked = False
                continue
            self.still_home(tid, t)
        self.resolve_pending(t)
        self.update_orphans(tracker, placed, dets, frame, t)
        arrivals, sure = self.arrivals(tracker, dets, frame, t)
        for tr in tracks:                     # its gold really was in the box (see went_and_back)
            if tr.id in arrivals or (self.seen_in_box(tr, t) and (self.rec[tr.id].picked or self.rec[tr.id].writer)):
                self.rec[tr.id].box_gold_t = t
        sizes =[self.rec[tr.id].size or 0. for tr in tracks if self.rec[tr.id].picked and not resting[tr.id]]
        self.vis_size = max(sizes) if sizes else None
        if sizes:
            self.hand_pinch_in_box(frame, t, self.vis_size)  # throttled; hand at the box? + skeleton on the feed
        hands_on = self.hands is not None and not self.hand_error
        hand_near = hands_on and 0. <= t-self.hand_near_t <= HAND_NEAR_HOLD_S
        gold_allowed = not hands_on or (REC_GOLD_FALLBACK and not hand_near)
        running = any(self.rec[tr.id].writer is not None for tr in tracks)
        hand_id = self.pinch_owner(tracks, arrivals, sure, resting, frame, t)
        for tr in tracks:
            rec = self.rec[tr.id]
            if rec.writer is None:
                by_hand = tr.id == hand_id                                   # 1. pinch in the box
                if running:
                    # PARALLEL clip: another jewel is already being worked (so a hand is at the box anyway).
                    # This one starts its own clip as soon as it is surely in the box - tracked there, or its
                    # box detection matched / identified as it - never on a guess.
                    by_gold = not by_hand and (tr.id in sure or self.seen_in_box(tr, t))
                else:                                                        # 2. gold in the box, no hand seen
                    by_gold = not by_hand and gold_allowed and (tr.id in arrivals or self.inside_box(tr))
                start = rec.picked and (by_hand or by_gold) and not resting[tr.id] and not rec.lock
                rec.arm_since = (t if rec.arm_since is None else rec.arm_since) if start else None
                if rec.arm_since is not None and t-rec.arm_since >= REC_START_S:
                    self.start(tr, rec, t, 'pinched in the box' if by_hand else
                               'in the box - parallel clip' if running else
                               'gold in the box - no hand seen' if hands_on else 'brought into the box')
                continue  # the pre-roll already covers this frame; live writing starts next frame
            if rec.unresolved:                 # its clip is about to be handed over / unassigned: just keep writing
                if small is not None:
                    try:
                        rec.writer.write(small)
                    except Exception:
                        pass
                continue
            if resting[tr.id]:
                rec.rest_since = t if rec.rest_since is None else rec.rest_since
                rec.rest_last = t
            elif rec.rest_since is not None and t-rec.rest_last > REST_GRACE_S:
                rec.rest_since = None          # really moving again (not a jumpy frame or two)
            placed_back = (rec.rest_since is not None and t-rec.rest_since >= REC_STOP_SETTLE_S
                           and t-rec.rest_last <= REST_GRACE_S)
            # Set down at a new spot during this clip, and that spot shows its gold again with no fingers on it:
            # placed - even if a hand passed over it meanwhile (covered, its id lost or put on another jewel),
            # which breaks the continuous still test above. The hand going on to the next jewel does not keep
            # this clip running.
            placed_back = placed_back or (
                rec.home is not None and rec.home_t > rec.start_t and t-rec.home_t >= REC_STOP_SETTLE_S
                and rec.home_state == 'HOME' and (resting[tr.id] or rec.home_view == 'gold')
                and not self.hand_on(rec.home.box, t))
            gone_now = (tr.state in ('MISSING', 'LEFT_ROI')  # a detection flicker trips this for a frame or two only
                        and not resting[tr.id])              # MISSING to the tracker, but its gold lies there: placed
            if gone_now and tr.state == 'MISSING' and self.hand_pinch_in_box(frame, t, rec.size):
                gone_now = False  # out of sight but held in a pinch at the box: still being worked, keep recording
            rec.gone_since = (t if rec.gone_since is None else rec.gone_since) if gone_now else None
            gone = rec.gone_since is not None and t-rec.gone_since >= REC_STOP_GONE_S
            over = t-rec.start_t >= REC_MAX_S
            if placed_back or gone or over:
                self.stop(tr.id, 'placed back' if placed_back else 'left' if gone else 'max length', t)
                continue
            if small is not None:
                try:
                    rec.writer.write(small)
                except Exception:
                    pass
        for orphan in self.orphans:            # unassigned clips keep recording until they are given to a jewel
            if orphan.writer is not None and small is not None:
                try:
                    orphan.writer.write(small)
                except Exception:
                    pass
        live = {tr.id for tr in tracks}
        for tid in [tid for tid, rec in self.rec.items() if tid not in live and rec.writer is not None]:
            self.stop(tid, 'track removed', t)

    def pinch_owner(self, tracks, arrivals, sure, resting, frame, t):
        """Track id a thumb/index pinch in the box starts a clip for (the main trigger), or None.
        Candidates are picked-up jewels still waiting for their clip; one seen in the box wins, else the one
        seen most recently - the one just carried in. While another clip runs (parallel), the pinch may be the
        hand working that jewel, so only a waiting jewel surely seen in the box, or one whose home shows it
        left, can take it - never one whose home is merely covered or still holds it."""
        if self.hands is None or self.box is None:
            return None
        candidates = [tr for tr in tracks if self.rec[tr.id].picked and self.rec[tr.id].writer is None
                      and not resting[tr.id] and not self.rec[tr.id].lock]
        holding = [tr for tr in candidates if self.hand_pinch_in_box(frame, t, self.rec[tr.id].size)]
        if not holding:
            return None
        def rank(tr):   # left its home (confirmed) > provisionally left > home only covered; then most recent
            rec = self.rec[tr.id]
            level = 2 if rec.home is None or rec.confirmed_left() else 1 if rec.home_state == 'LEFT' else 0
            return level, max(tr.det_seen, rec.left_t)
        if any(self.rec[tr.id].writer is not None for tr in tracks):
            # PARALLEL: another clip runs, so the pinch may be the hand working that jewel. A waiting jewel
            # takes it when it is surely in the box, or when its home shows it has left (it is in a hand too).
            holding = [tr for tr in holding if tr.id in sure or self.seen_in_box(tr, t)
                       or self.rec[tr.id].home_state == 'LEFT']
            if not holding:
                return None
            return max(holding, key=rank).id
        seen = [tr for tr in holding if tr.id in arrivals or self.inside_box(tr)]
        return max(seen or holding, key=rank).id

    @staticmethod
    def grip_limit(scale, size=None):
        """Largest thumb-index tip gap that still holds a jewel `size` px wide. jewel_tracker_hand.py only
        allows a tight pinch (small piece hidden between the fingers); a bangle is held from above with the
        thumb under it and the fingers over it, so the tips are about the jewel's width apart.
        HAND_PINCH_PX set -> that fixed distance only."""
        if HAND_PINCH_PX is not None:
            return float(HAND_PINCH_PX)
        return max(HAND_PINCH_MIN_PX, HAND_PINCH_MAX_RATIO*scale, HAND_GRIP_SIZE_RATIO*(size or 0.))

    def hand_pinch_in_box(self, frame, t, size=None):
        """Thumb and index tips holding a jewel of `size` px with their midpoint inside the box, seen within
        the last HAND_HOLD_S. The hand model runs at most every HAND_CHECK_S; in between the last result holds."""
        if self.hands is None or frame is None or self.box is None:
            return False
        if t-self.hand_check_t >= HAND_CHECK_S or t < self.hand_check_t:
            self.hand_check_t = t
            self.detect_pinch(frame, t)
        held = any(0. <= t-when <= HAND_HOLD_S and gap <= self.grip_limit(scale, size)
                   for when, gap, scale in self.grips)
        if held:
            self.hand_pinch_t = t
        return held

    def detect_pinch(self, frame, t):
        """Hand landmarks on the box widened by HAND_CROP_PAD; a thumb/index pinch whose midpoint is inside the box."""
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = self.box
        px = max(int((x2-x1)*HAND_CROP_PAD), (HAND_CROP_MIN_PX-(x2-x1))//2)
        py = max(int((y2-y1)*HAND_CROP_PAD), (HAND_CROP_MIN_PX-(y2-y1))//2)
        cx1, cy1, cx2, cy2 = max(0, x1-px), max(0, y1-py), min(fw, x2+px), min(fh, y2+py)
        if cx2-cx1 < 32 or cy2-cy1 < 32:
            return
        try:
            hands = self.hands.detect(np.ascontiguousarray(frame[cy1:cy2, cx1:cx2]))
        except Exception as exc:
            if self.hand_error != str(exc):
                self.emit(dict(time=t, state='HAND_MODEL_ERROR', reason=f'{type(exc).__name__}: {exc}'))
            self.hand_error = str(exc)
            return
        self.hand_error = ''
        vis = []
        for hand in hands:
            points = hand.points+np.array((cx1, cy1), np.float32)  # crop -> full-frame px (no resize)
            thumb, index = points[4], points[8]
            mid = (thumb+index)/2.
            gap, scale = float(np.linalg.norm(thumb-index)), float(hand.scale)
            inside = self.on_box(mid, margin=0)
            counts = hand_counts(hand)          # PINCH_HAND only: the other hand (holding the stone, resting
            if counts:                          # near the box) neither pinches nor blocks the gold trigger
                if any(self.on_box(p) for p in points):   # the working hand at the box: pinch decides, not gold
                    self.hand_near_t = t
                if inside:                      # judged per jewel size in hand_pinch_in_box
                    self.grips.append((t, gap, scale))
            label = (getattr(hand, 'handedness', '') or '?')[:1]
            vis.append((points, mid, inside, gap, scale, counts, label))
        self.hand_vis, self.hand_vis_t = vis, t

    def start(self, tr, rec, t, reason):
        preroll = list(self.preroll)
        if not preroll:                       # nothing buffered yet: nothing to open with
            rec.arm_since = None
            return
        writer = None
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            h, w = preroll[-1].shape[:2]
            rec.count += 1
            stamp = datetime.now(TIMEZONE).strftime('%Y%m%d_%H%M%S')
            path = self.folder/f'J{tr.id}_{rec.count:03d}_{stamp}.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
            if not writer.isOpened():         # mp4v not available on this build: fall back to MJPG/AVI
                writer.release()
                path = path.with_suffix('.avi')
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (w, h))
            if not writer.isOpened():
                writer.release()
                rec.arm_since = None
                self.emit(dict(time=t, jewel=f'J{tr.id}', state='REC_FAILED',
                               reason='could not start recording (no video writer)'))
                return
            rec.writer, rec.path, rec.start_t = writer, path, t
            rec.rest_since = rec.gone_since = None
            rec.provisional = rec.home is not None and not rec.confirmed_left()
            for frame in preroll:             # the frames just before the trigger: the approach into the box
                if frame.shape[:2] == (h, w):
                    writer.write(frame)
            self.emit(dict(time=t, jewel=f'J{tr.id}', state='REC_STARTED', reason=reason,
                           clip=path.name, preroll_frames=len(preroll), provisional=rec.provisional))
        except Exception as exc:
            if writer is not None and rec.writer is not writer:
                writer.release()
            rec.arm_since = None
            self.emit(dict(time=t, jewel=f'J{tr.id}', state='REC_FAILED', reason=f'recording could not start: {exc}'))

    def stop(self, tid, reason, t):
        rec = self.rec.get(tid)
        if rec is None:
            return
        if rec.writer is not None:
            try:
                rec.writer.release()
            except Exception:
                pass
            path = self.rename_clip(rec.path, tid) if rec.path else None   # named after its final jewel
            rec.last_clip = path.name if path else ''
            self.emit(dict(time=t, jewel=f'J{tid}', state='REC_SAVED', reason=reason, provisional=rec.provisional,
                           clip=rec.last_clip, duration_s=round(t-rec.start_t, 3)))
            self.stopped_note = (f'REC STOPPED J{tid} ({reason})', t)
            rec.pending_clip = path if rec.provisional else None
        rec.writer, rec.path, rec.arm_since, rec.provisional = None, None, None, False
        rec.rest_since = rec.gone_since = None
        rec.lock = True    # block a new clip until this jewel is seen resting again (no flicker restarts)

    def stop_all(self, reason, t=None):
        t = self.last_t if t is None else t
        for tid, rec in self.rec.items():
            if rec.writer is not None:
                self.stop(tid, reason, t)
        for orphan in self.orphans:            # never given to a jewel: kept as UNASSIGNED_...
            self.save_orphan_unassigned(orphan, t, f'{reason} - no jewel was placed away from its home')
        self.orphans = []

    def status(self, tr, t):
        """(trip, recording) text for the table."""
        rec = self.rec.get(tr.id) or RecState()
        trip = ('set down (tracker lost it; its gold box)' if tr.id in self.loose_owners else
                'in box' if self.inside_box(tr) and rec.picked else 'picked up' if rec.picked else
                'resting' if self.resting(tr, t) else '-')
        if rec.home is not None:
            trip += {'HOME': ' | at home', 'COVERED': ' | home covered (candidate)',
                     'LEFT': ' | left home' if rec.left_by == 'empty' else ' | left home? (moved)'}.get(rec.home_state, '')
        if rec.writer is not None:
            s = max(0, int(t-rec.start_t))
            recording = f'REC {s//60}:{s % 60:02d}' + (' (provisional)' if rec.provisional else '')
        elif rec.arm_since is not None:
            recording = f'arming {t-rec.arm_since:.1f}/{REC_START_S:.1f} s'
        elif rec.lock:
            recording = 'done - rest on bench to re-arm'
        else:
            recording = '-'
        return trip, recording

    def draw(self, vis, t, scale=1., line=1):
        """Target box, jewel homes, hand skeleton during trips, and the REC badge (bottom-right)."""
        for tid, here in self.loose_owners.items():   # tracker says MISSING, but its gold lies here (set down)
            lx1, ly1, lx2, ly2 = (int(v) for v in here.det_box)
            cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), (60, 220, 60), line)
            cv2.putText(vis, f'J{tid} set down', (lx1, ly2+round(14*scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        .42*scale, (60, 220, 60), line)
        for tid, rec in self.rec.items():   # homes: green at home, yellow covered, red left (thin = provisional)
            if rec.home is None:
                continue
            hx1, hy1, hx2, hy2 = rec.home.box
            color = HOME_COLORS.get(rec.home_state, (200, 200, 200))
            thick = line if rec.home_state != 'LEFT' or rec.left_by == 'empty' else max(1, line//2)
            cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), color, thick)
            cv2.putText(vis, f'J{tid} home', (hx1, hy2+round(14*scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        .42*scale, color, line)
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, 2*line)
            cv2.putText(vis, 'REC BOX', (x1, max(round(14*scale), y1-round(6*scale))),
                        cv2.FONT_HERSHEY_SIMPLEX, .5*scale, BOX_COLOR, line)
        if self.hand_vis and 0. <= t-self.hand_vis_t <= HAND_HOLD_S:
            from standalone.hand_keypoints import draw_hand
            for points, mid, inside, gap, hand_scale, counts, label in self.hand_vis:
                wrist = tuple(np.rint(points[0]).astype(int))
                if not counts:               # the other hand: thin grey skeleton, labelled, never triggers
                    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                                 (0, 9), (9, 12), (0, 13), (13, 16), (0, 17), (17, 20)):
                        cv2.line(vis, tuple(np.rint(points[a]).astype(int)), tuple(np.rint(points[b]).astype(int)),
                                 (120, 120, 120), line)
                    cv2.putText(vis, f'{label} (ignored)', wrist, cv2.FONT_HERSHEY_SIMPLEX, .6*scale,
                                (150, 150, 150), line)
                    continue
                limit = self.grip_limit(hand_scale, self.vis_size)
                gripping = inside and gap <= limit
                draw_hand(vis, SimpleNamespace(points=points), line)
                cv2.putText(vis, label, wrist, cv2.FONT_HERSHEY_SIMPLEX, .8*scale, (0, 240, 240), 2*line)
                cv2.putText(vis, f'gap {gap:.0f}px / trigger <= {limit:.0f}px',
                            (round(mid[0]+10*scale), round(mid[1]-10*scale)), cv2.FONT_HERSHEY_SIMPLEX,
                            .5*scale, REC_BADGE_COLOR if gripping else (200, 200, 200), line)
                cv2.line(vis, tuple(np.rint(points[4]).astype(int)), tuple(np.rint(points[8]).astype(int)),
                         REC_BADGE_COLOR if gripping else (200, 200, 200), line)
                cv2.circle(vis, tuple(np.rint(mid).astype(int)), round(6*scale),
                           REC_BADGE_COLOR if gripping else (200, 200, 200), 2*line)
        recording = [(tid, rec) for tid, rec in sorted(self.rec.items()) if rec.writer is not None]
        unassigned = [o for o in self.orphans if o.writer is not None]
        if recording or unassigned:
            parts = []
            for tid, rec in recording:
                s = max(0, int(t-rec.start_t))
                parts.append(f'J{tid} {s//60}:{s % 60:02d}')
            for o in unassigned:
                s = max(0, int(t-o.start_t))
                parts.append(f'J? {s//60}:{s % 60:02d}')
            text, filled = 'REC '+'  '.join(parts), True
        elif self.stopped_note is not None and 0. <= t-self.stopped_note[1] <= REC_STOPPED_SHOW_S:
            text, filled = self.stopped_note[0], False
        else:
            return
        font_scale, thick = .9*scale, max(2, 2*line)
        pad, dot = round(10*scale), round(9*scale)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        h, w = vis.shape[:2]
        x2, y2 = w-round(12*scale), h-round(8*scale)
        x1, y1 = x2-(tw+2*pad+(2*dot+pad if filled else 0)), y2-(th+base+2*pad)
        if filled:   # recording: solid red badge, white text, blinking white dot
            cv2.rectangle(vis, (x1, y1), (x2, y2), REC_BADGE_COLOR, -1)
            if int(t*2) % 2 == 0:
                cv2.circle(vis, (x1+pad+dot, (y1+y2)//2), dot, (255, 255, 255), -1)
            cv2.putText(vis, text, (x1+2*pad+2*dot, y2-pad-base), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thick)
        else:        # just stopped: red outline and red text
            cv2.rectangle(vis, (x1, y1), (x2, y2), REC_BADGE_COLOR, thick)
            cv2.putText(vis, text, (x1+pad, y2-pad-base), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, REC_BADGE_COLOR, thick)


class Models:
    """Gold INT8 + the original jewel_tracker_5 Worker tracking; optional hand landmarks for the pinch fallback."""
    def __init__(self, original, log=None):
        from standalone.tracking import GoldPredictor
        self.gold = GoldPredictor()
        self.original = original
        self.keypoints = None
        self.log = log or (lambda message: None)
        self.identity_error = ''
        self.reset_tracking()

    def enable_keypoints(self):
        if self.keypoints is None:
            model = ROOT/'model onnx'/'hand_landmarker.task'
            if not model.is_file():
                model = ROOT/'hand_landmarker.task'
            self.keypoints = make_hand_model(model)
        return self.keypoints

    def close_keypoints(self):
        if self.keypoints is not None:
            self.keypoints.close()
            self.keypoints = None

    def close_tracking(self):
        tracker = getattr(self, 'tracker', None)
        if tracker is None:
            return
        # Invalidate any background gallery publication for this old assignment.
        with tracker.lock:
            tracker.running = False
            tracker.flow_revision += 1
            tracker.identity_training_token = None
        thread = tracker.id_training_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.)

    def reset_tracking(self):
        old = getattr(self, 'tracker', None)
        self.close_tracking()
        original = self.original

        class PickupTracker(original.Worker):
            # Preserve assignment, learn_ids, update_tracks, matching, recovery and evaluate from the
            # original Worker. This test has no touchstone or material-testing workflow.
            def stone_present(self, frame):
                return True
            def update_rubbing(self, t):
                pass
            def update_acid(self, t):
                pass
            def check_done(self, *args):
                pass
            def save_capture(self, frame, jewels):
                pass

        self.tracker = PickupTracker(0, [], continuous=True)
        if old is not None:
            self.tracker.identity_training_lock = old.identity_training_lock
        self.identity_error = ''
        if not self.tracker.identity_slic or self.tracker.id_model is None:
            raise RuntimeError('Original SLIC identity tracking is unavailable. Run with the project .venv-slic Python.')

        def event(message, level):
            if 'gallery unavailable' in message:
                self.identity_error = message
            if message.startswith('Initial IDs/crops ready after'):
                message = 'Initial jewel IDs/crops ready; original tracker assignment locked.'
            self.log('Tracker: '+message)
        self.tracker.event.connect(event, original.QtCore.Qt.DirectConnection)

    def identity_status(self):
        if self.identity_error:
            return 'identity unavailable (see log)'
        if self.tracker.mode != 'LOCKED':
            return 'original assignment preparing'
        return ('SLIC identity ready' if self.tracker.id_model.ready else
                'SLIC identity gallery preparing')

    def process(self, frame, roi, t):
        """One original gold pass, followed by the original lock/update path. Returns the raw detections."""
        tracker = self.tracker
        tracker.frame_wh = (frame.shape[1], frame.shape[0])
        tracker.roi = np.asarray(roi, np.int32)
        tracker.recent.append(frame)
        raw, rejected = tracker.detect(self.gold, frame, tracker.roi, captured_at=t)
        # Mirrors Worker's run loop, without stone/rubbing/acid processing.
        with tracker.lock:
            tracker.rejected = rejected
            if tracker.mode == 'ARMING':
                tracker.try_lock(raw, t, frame)
            elif tracker.mode == 'LOCKED':
                tracker.update_tracks(raw, t, frame)
        return raw


def dependencies():
    if not (TRACKER/'jewel_tracker_5.py').is_file():
        raise RuntimeError('Set JEWEL_TRACKER_DIR to the directory containing jewel_tracker_5.py.')
    import jewel_tracker_5 as original  # preserves torch-before-Qt loading order
    import torch
    import ultralytics.utils.torch_utils as torch_utils
    torch.set_num_threads(2)
    torch_utils.NUM_THREADS = 2
    cv2.setNumThreads(2)
    from PyQt5 import QtCore, QtGui, QtWidgets
    for name in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        if '/cv2/' in os.environ.get(name, '').replace('\\', '/'):
            os.environ.pop(name, None)
    os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath))
    return original, QtCore, QtGui, QtWidgets


def window_class(original, QtCore, QtGui, QtWidgets):
    class Worker(QtCore.QThread):
        result = QtCore.pyqtSignal(object)
        log = QtCore.pyqtSignal(str)
        failure = QtCore.pyqtSignal(str)
        notice = QtCore.pyqtSignal(str)

        def __init__(self, source, output, target_fps, hand_fallback=True, cli_box=None):
            super().__init__()
            self.source, self.output, self.target_fps = source, output, target_fps
            self.hand_fallback, self.cli_box = hand_fallback, cli_box
            self.stop_event = threading.Event()
            self.reset_event = threading.Event()
            self.box_lock = threading.Lock()
            self.box_request = None      # ('set', box) from the GUI thread, applied on the next frame
            self.grabber = None

        def stop(self):
            self.stop_event.set()
            if self.grabber:
                self.grabber.stop()

        def request_box(self, box):
            with self.box_lock:
                self.box_request = ('set', box)

        def run(self):
            from standalone.settings import tracker_camera, tracker_roi
            from standalone.camera import make_grabber
            cap = models = recorder = None
            try:
                self.output.mkdir(parents=True, exist_ok=False)
                self.log.emit('Loading gold INT8 and the original tracker on CPU…')
                self.notice.emit('Loading existing CPU models; camera will open next…')
                models = Models(original, self.log.emit)
                hands, hand_status = None, 'hand pinch: off (unchecked)'
                if self.hand_fallback:
                    try:
                        hands = models.enable_keypoints()
                        hand_status = 'hand pinch: on'
                        self.log.emit('Hand model ready (MediaPipe, CPU): thumb/index pinch fallback for recording')
                    except Exception as exc:  # e.g. mediapipe not installed: recording still works without it
                        hand_status = f'HAND PINCH OFF - {type(exc).__name__}: {exc}'
                        self.log.emit(f'ERROR: hand-pinch recording trigger unavailable ({type(exc).__name__}: {exc}). '
                                      f'Python: {sys.executable}. Run with {ROOT/".venv-slic"} (has mediapipe).')
                video = isinstance(self.source, str) and Path(self.source).is_file()
                camera = tracker_camera(0 if video else self.source)
                reference_size = tuple(map(int, camera.get('resolution', '2560x1440').split('x')))
                packet_settings = load_json(ROOT/'packet_ui_settings.json')
                saved_roi = packet_settings.get('roi', []) if packet_settings.get('roi_override') else tracker_roi()
                zones = load_json(original.ZONES_FILE)
                if len(saved_roi) < 3:
                    raise ValueError('Save a valid bench ROI in the main application/Jewel Tracker first. No full-frame fallback is used.')
                if video:
                    cap = cv2.VideoCapture(self.source)
                    if not cap.isOpened():
                        raise ValueError('Cannot open video.')
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not 0 < fps <= 240:
                        raise ValueError('Video has no usable FPS timestamps.')
                    stride = max(1, math.ceil(fps/self.target_fps))
                else:
                    self.grabber = make_grabber(self.source, camera)
                    self.grabber.start()
                session = 0
                folder = self.output

                def stamp_of(value):
                    return (f'video {value:.3f}s' if video else
                            datetime.fromtimestamp(value, TIMEZONE).strftime('%H:%M:%S.%f')[:-3]+' IST')

                def event(data):
                    with (folder/'events.jsonl').open('a') as handle:
                        handle.write(json.dumps(data, default=json_default)+'\n')
                    clip = f" | {data['clip']}" if data.get('clip') else ''
                    self.log.emit(f"{stamp_of(data['time'])} | {data.get('jewel', '')} {data['state']} | {data.get('reason', '')}{clip}")

                recorder = BoxRecorder(folder, self.target_fps, event, hands)
                box_source = None
                index, previous_t, shape = -1, None, None
                last_wall, started_wall = 0., time.perf_counter()
                connection_message = None
                last_connection_notice = -math.inf
                while not self.stop_event.is_set():
                    if self.reset_event.is_set():
                        self.reset_event.clear()
                        recorder.stop_all('reset')
                        session += 1
                        folder = self.output/f'reset_{session:03d}'
                        folder.mkdir()
                        recorder = BoxRecorder(folder, self.target_fps, event, hands, recorder.box)
                        models.reset_tracking()
                        if video:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            index, previous_t, started_wall = -1, None, time.perf_counter()
                        self.log.emit('RESET: previous clips/events preserved; clear hands and lay out jewels again.')
                    if video:
                        ok = cap.grab()
                        index += 1
                        if not ok:
                            self.log.emit('Video ended. Clips retained; use Start for another source.')
                            break
                        if index % stride:
                            continue
                        ok, frame = cap.retrieve()
                        t = index/fps
                        if not ok:
                            raise OSError('Video frame decode failed.')
                        delay = started_wall+t-time.perf_counter()
                        if delay > 0 and self.stop_event.wait(delay):
                            break
                    else:
                        delay = 1/self.target_fps-(time.perf_counter()-last_wall)
                        if delay > 0 and self.stop_event.wait(delay):
                            break
                        frame, t = self.grabber.latest(previous_t)
                        if frame is None:
                            now = time.monotonic()
                            if now-last_connection_notice >= 1.:
                                applying = time.time() < getattr(self.grabber, 'af_busy_until', 0.)
                                message = (self.grabber.problem or
                                           ('Applying saved camera controls; waiting for the first frame…' if applying else
                                            f'Waiting for camera {self.source}. Check the device number and whether another application is using it.'))
                                self.notice.emit(message)
                                if message != connection_message:
                                    self.log.emit(message)
                                    connection_message = message
                                last_connection_notice = now
                            self.stop_event.wait(.03)
                            continue
                        if time.time()-t > .8:
                            self.log.emit('Stale camera frame rejected.')
                            continue
                    sample_gap = t-previous_t if previous_t is not None else None
                    previous_t, last_wall = t, time.perf_counter()
                    if shape is not None and shape != frame.shape:
                        raise ValueError('Resolution changed. Stop and restart; positions cannot be reused.')
                    shape = frame.shape
                    if box_source is None:
                        box, box_source = resolve_box(shape, reference_size, zones, self.cli_box)
                        recorder.set_box(box)
                        self.log.emit(f'Recording box from {box_source}: {recorder.box}' if recorder.box else
                                      'No recording box yet: click "Draw box" and drag over the work area.')
                    with self.box_lock:
                        request, self.box_request = self.box_request, None
                    if request is not None:
                        recorder.set_box(request[1])
                        self.log.emit(f'Recording box set: {recorder.box}' if recorder.box else 'Recording box cleared.')
                    scale = np.array([frame.shape[1]/reference_size[0], frame.shape[0]/reference_size[1]])
                    roi = np.rint(np.asarray(saved_roi)*scale).astype(np.int32)
                    if not (folder/'session.json').exists():
                        (folder/'session.json').write_text(json.dumps(dict(
                            source_kind='video' if video else 'camera',
                            camera_index=self.source if isinstance(self.source, int) else None,
                            saved_resolution=list(reference_size), capture_size=[frame.shape[1], frame.shape[0]],
                            saved_roi=saved_roi, recording_box=recorder.box, recording_box_source=box_source,
                            timestamp_kind='video_seconds' if video else 'unix_seconds',
                            max_inference_fps=self.target_fps, clip_fps=self.target_fps,
                            trigger='picked up off the bench + pinched in the box (main) / gold in the box with no hand seen (debounced)',
                            hand_pinch_fallback=hands is not None, hand_status=hand_status, pinch_px_at_start=HAND_PINCH_PX, pinch_hand_at_start=PINCH_HAND, python=sys.executable,
                            rec=dict(start_s=REC_START_S, stop_settle_s=REC_STOP_SETTLE_S, stop_gone_s=REC_STOP_GONE_S,
                                     max_s=REC_MAX_S, preroll_frames=REC_PREROLL_FRAMES, max_width=REC_MAX_W),
                            tracker='original_Worker_try_lock_update_tracks_SLIC_identity'), indent=2, default=json_default))
                    started = time.perf_counter()
                    raw = models.process(frame, roi, t)
                    tracker = models.tracker
                    if tracker.mode == 'LOCKED':
                        recorder.update(tracker, raw, frame, t)
                        message = 'Recording armed: pick a jewel up and bring it inside the box.'
                    else:
                        message = (f'Original assignment: {tracker.arm_count} jewels, '
                                   f'{tracker.stable:.0%}; keep jewels still and hands clear.')
                    elapsed = time.perf_counter()-started
                    display = frame.copy()
                    overlay_scale = max(1., frame.shape[1]/960.)
                    line = max(1, round(overlay_scale))
                    cv2.polylines(display, [roi], True, (255, 160, 0), line)
                    for d in raw:            # thin grey: raw detections (what the box trigger sees)
                        cv2.rectangle(display, tuple(map(int, d.box[:2])), tuple(map(int, d.box[2:])), (150, 150, 150), line)
                    rows = []
                    for tr in sorted(tracker.tracks, key=lambda x: x.id) if tracker.mode == 'LOCKED' else []:
                        rec = recorder.rec.get(tr.id) or RecState()
                        color = (REC_BADGE_COLOR if rec.writer is not None else (0, 220, 255) if rec.picked else
                                 (60, 220, 60))
                        x1, y1, x2, y2 = map(int, tr.box)
                        if t-tr.det_seen < .3:
                            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2*line)
                        else:
                            original.dashed_rect(display, (x1, y1), (x2, y2), color, dash=max(5, round(5*overlay_scale)))
                        trip, recording = recorder.status(tr, t)
                        cv2.putText(display, f'J{tr.id} {tr.state} | {trip}' + (' | REC' if rec.writer is not None else ''),
                                    (x1, max(round(16*overlay_scale), y1-round(6*overlay_scale))),
                                    cv2.FONT_HERSHEY_SIMPLEX, .42*overlay_scale, color, line)
                        rows.append([f'J{tr.id}', tr.state, trip, recording,
                                     rec.path.name if rec.path else rec.last_clip])
                    recorder.draw(display, t, overlay_scale, line)
                    observed_fps = f'{1/sample_gap:.1f}' if sample_gap and sample_gap > 0 else '-'
                    pinch = f' ({pinch_text()})' if hands is not None else ''
                    if hands is not None and recorder.hand_pinch_t is not None and 0. <= t-recorder.hand_pinch_t <= HAND_HOLD_S:
                        pinch = ' (pinch in box)'
                    if recorder.hand_error:
                        pinch += f' - ERROR {recorder.hand_error}'
                    status = (f'{message} | box: {recorder.box or "not set"} | {hand_status}{pinch} | {observed_fps} observed FPS | '
                              f'{elapsed*1000:.0f} ms processing | capture {frame.shape[1]}×{frame.shape[0]} | '
                              f'{models.identity_status()} | t={t:.3f}')
                    self.result.emit(dict(frame=small_frame(display), rows=rows, status=status,
                                          source_size=(frame.shape[1], frame.shape[0])))
                self.log.emit('Stopped. Clips and events: '+str(self.output))
            except Exception as exc:
                try:
                    self.output.mkdir(parents=True, exist_ok=True)
                    (self.output/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
                except OSError:
                    pass
                traceback.print_exc()
                self.failure.emit(str(exc))
            finally:
                if recorder:
                    recorder.stop_all('camera stopped')
                if models:
                    models.close_tracking()
                    models.close_keypoints()
                if cap:
                    cap.release()
                if self.grabber:
                    self.grabber.stop()
                    self.grabber.join(timeout=4)

    class VideoLabel(QtWidgets.QLabel):
        """Shows the feed; while drawing, a left-drag emits the box as fractions (x1, y1, x2, y2) of the image."""
        box_drawn = QtCore.pyqtSignal(object)

        def __init__(self, text):
            super().__init__(text)
            self.drawing = False
            self.drag_start = self.drag_end = None

        def image_rect(self):
            pixmap = self.pixmap()
            if pixmap is None or pixmap.isNull():
                return None
            return ((self.width()-pixmap.width())/2, (self.height()-pixmap.height())/2,
                    pixmap.width(), pixmap.height())

        def fraction(self, pos):
            rect = self.image_rect()
            if rect is None:
                return None
            x, y, w, h = rect
            return min(max((pos.x()-x)/w, 0.), 1.), min(max((pos.y()-y)/h, 0.), 1.)

        def mousePressEvent(self, event):
            if self.drawing and event.button() == QtCore.Qt.LeftButton:
                self.drag_start = self.drag_end = self.fraction(event.pos())
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if self.drag_start is not None:
                self.drag_end = self.fraction(event.pos())
                self.update()
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            if self.drag_start is not None and self.drag_end is not None:
                (ax, ay), (bx, by) = self.drag_start, self.drag_end
                self.box_drawn.emit((min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)))
            self.drag_start = self.drag_end = None
            self.update()
            super().mouseReleaseEvent(event)

        def paintEvent(self, event):
            super().paintEvent(event)
            rect = self.image_rect()
            if self.drag_start is None or self.drag_end is None or rect is None:
                return
            x, y, w, h = rect
            (ax, ay), (bx, by) = self.drag_start, self.drag_end
            painter = QtGui.QPainter(self)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 165, 0), 2, QtCore.Qt.DashLine))
            painter.drawRect(QtCore.QRectF(x+min(ax, bx)*w, y+min(ay, by)*h, abs(bx-ax)*w, abs(by-ay)*h))
            painter.end()

    class Window(QtWidgets.QWidget):
        def __init__(self, args):
            super().__init__()
            from standalone.settings import tracker_source
            self.args, self.worker, self.last_image, self.source_size = args, None, None, None
            self.setWindowTitle('Pickup trigger test - box recording')
            self.resize(1250, 900)
            layout = QtWidgets.QVBoxLayout(self)
            bar = QtWidgets.QHBoxLayout(); layout.addLayout(bar)
            bar.addWidget(QtWidgets.QLabel('Camera:'))
            self.camera = QtWidgets.QSpinBox(); self.camera.setRange(0, 15); bar.addWidget(self.camera)
            saved = tracker_source()
            try:
                packet_source = json.loads((ROOT/'packet_ui_settings.json').read_text()).get('source')
                if packet_source is not None and str(packet_source).isdigit():
                    saved = str(packet_source)
            except (OSError, ValueError, TypeError):
                pass
            self.available_cameras = capture_devices()
            automatic_selection = None
            indices = [index for index, _ in self.available_cameras]
            if args.source is None and saved.isdigit() and self.available_cameras and int(saved) not in indices:
                automatic_selection = f'Saved camera {saved} is unavailable; selected camera {indices[0]} for this test only.'
                saved = str(indices[0])
            if saved.isdigit() and int(saved) <= 15:
                self.camera.setValue(int(saved))
            self.source = QtWidgets.QLineEdit(args.source if args.source is not None else saved)
            bar.addWidget(self.source, 1)
            for text, callback in [('Open camera', self.use_camera), ('Open video', self.open_video),
                                   ('Start', self.start), ('Stop', self.stop), ('Reset', self.reset)]:
                button = QtWidgets.QPushButton(text); button.clicked.connect(callback); bar.addWidget(button)
            box_bar = QtWidgets.QHBoxLayout(); layout.addLayout(box_bar)
            self.draw_button = QtWidgets.QPushButton('Draw box')
            self.draw_button.setCheckable(True)
            self.draw_button.setStyleSheet('QPushButton:checked {background: #ea580c; color: white;}')
            self.draw_button.toggled.connect(self.toggle_draw)
            box_bar.addWidget(self.draw_button)
            clear_button = QtWidgets.QPushButton('Clear box'); clear_button.clicked.connect(self.clear_box)
            box_bar.addWidget(clear_button)
            self.hand_fallback = QtWidgets.QCheckBox('Hand-pinch fallback (MediaPipe; applies on Start)')
            self.hand_fallback.setChecked(not args.no_hand_fallback)
            box_bar.addWidget(self.hand_fallback)
            box_bar.addWidget(QtWidgets.QLabel('Pinch trigger: thumb-index gap ≤'))
            self.pinch_px = QtWidgets.QSpinBox()
            self.pinch_px.setRange(0, 2000); self.pinch_px.setSingleStep(10); self.pinch_px.setSuffix(' px')
            self.pinch_px.setSpecialValueText('auto')   # 0 -> automatic (hand / jewel size)
            self.pinch_px.setToolTip('Live: applies immediately, also while recording. Read the "gap ...px" '
                                     'shown on the feed while pinching in the box. 0 = auto.')
            saved_px = load_json(SETTINGS_FILE).get('pinch_px', HAND_PINCH_PX)
            set_pinch_px(saved_px if isinstance(saved_px, (int, float)) and saved_px >= 0 else HAND_PINCH_PX)
            self.pinch_px.setValue(HAND_PINCH_PX or 0)
            self.pinch_px.valueChanged.connect(self.on_pinch_px)
            box_bar.addWidget(self.pinch_px)
            box_bar.addWidget(QtWidgets.QLabel('hand:'))
            self.pinch_hand = QtWidgets.QComboBox(); self.pinch_hand.addItems(PINCH_HANDS)
            self.pinch_hand.setToolTip('Only this hand\'s pinch triggers; the other hand is ignored. Live. '
                                       'The feed labels each hand R / L - if your right hand shows L, pick Left.')
            set_pinch_hand(load_json(SETTINGS_FILE).get('pinch_hand', PINCH_HAND))
            self.pinch_hand.setCurrentText(PINCH_HAND)
            self.pinch_hand.currentTextChanged.connect(self.on_pinch_hand)
            box_bar.addWidget(self.pinch_hand)
            box_bar.addStretch(1)
            self.video = VideoLabel('Start with all jewels on the bed and hands clear.')
            self.video.setMinimumSize(640, 360); self.video.setAlignment(QtCore.Qt.AlignCenter)
            self.video.setStyleSheet('background:#15202b;color:#ddd;'); layout.addWidget(self.video, 1)
            self.video.box_drawn.connect(self.on_box)
            self.status = QtWidgets.QLabel('Recording starts when a picked-up jewel is brought inside the box.')
            self.status.setWordWrap(True); layout.addWidget(self.status)
            help_text = QtWidgets.QLabel(
                f'A clip starts when, for {REC_START_S:g} s, a jewel picked up off the bench is pinched in the orange box '
                '(thumb-index gap at most the "Pinch trigger" value) - or, when no hand is seen at the box, its gold '
                f'detection is in the box. Clips open with the last {REC_PREROLL_FRAMES} frames. '
                f'It stops when the jewel lies still on the bench for {REC_STOP_SETTLE_S:g} s, is missing/out of the ROI for '
                f'{REC_STOP_GONE_S:g} s, or after {REC_MAX_S:g} s. One clip per trip: the jewel must rest on the bench before another. '
                'Several jewels in the box record in parallel, one clip each. '
                'Keep the box inside the bench ROI. Clips are raw frames, saved in the session folder.')
            help_text.setWordWrap(True); layout.addWidget(help_text)
            self.table = QtWidgets.QTableWidget(0, 5)
            self.table.setHorizontalHeaderLabels(['ID', 'Tracker state', 'Trip', 'Recording', 'Clip'])
            self.table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
            for column, width in enumerate((70, 120, 110, 220)):
                self.table.setColumnWidth(column, width)
            self.table.setMaximumHeight(180); layout.addWidget(self.table)
            self.logs = QtWidgets.QPlainTextEdit(); self.logs.setReadOnly(True)
            self.logs.setMaximumBlockCount(350); self.logs.setMaximumHeight(130); layout.addWidget(self.logs)
            if self.available_cameras:
                self.logs.appendPlainText('Available capture cameras: '+', '.join(f'{index}: {name}' for index, name in self.available_cameras))
            if automatic_selection:
                self.logs.appendPlainText(automatic_selection)
                self.status.setText(automatic_selection+' Click Open camera or Start.')
            if args.autostart:
                QtCore.QTimer.singleShot(0, self.start)

        def toggle_draw(self, checked):
            self.video.drawing = checked
            self.video.setCursor(QtCore.Qt.CrossCursor if checked else QtCore.Qt.ArrowCursor)
            if checked:
                self.status.setText('Drag a box over the work area on the video.')

        def on_box(self, fractions):
            self.draw_button.setChecked(False)
            if self.source_size is None:
                self.logs.appendPlainText('Start a source first: the box is drawn on the live picture.')
                return
            w, h = self.source_size
            box = [round(fractions[0]*w), round(fractions[1]*h), round(fractions[2]*w), round(fractions[3]*h)]
            if box[2]-box[0] < 4 or box[3]-box[1] < 4:
                self.logs.appendPlainText('Box too small; drag a larger box.')
                return
            try:
                BOX_FILE.write_text(json.dumps(dict(box=box, size=[w, h]), indent=2))
            except OSError as exc:
                self.logs.appendPlainText(f'Could not save {BOX_FILE.name}: {exc}')
            if self.worker and self.worker.isRunning():
                self.worker.request_box(box)
            self.logs.appendPlainText(f'Recording box saved: {box} (capture {w}×{h}).')

        def clear_box(self):
            try:
                BOX_FILE.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                self.logs.appendPlainText(f'Could not remove {BOX_FILE.name}: {exc}')
            if self.worker and self.worker.isRunning():
                self.worker.request_box(None)
            self.logs.appendPlainText('Recording box cleared (running clips finish by their own stop rules).')

        def save_setting(self, key, value):
            settings = load_json(SETTINGS_FILE)
            settings[key] = value
            try:
                SETTINGS_FILE.write_text(json.dumps(settings, indent=2))
            except OSError as exc:
                self.logs.appendPlainText(f'Could not save {SETTINGS_FILE.name}: {exc}')
            self.logs.appendPlainText(f'Pinch trigger: {pinch_text()}')

        def on_pinch_px(self, value):
            set_pinch_px(value)   # module setting: the running Worker uses it from the next frame
            self.save_setting('pinch_px', int(value))

        def on_pinch_hand(self, value):
            set_pinch_hand(value)
            self.save_setting('pinch_hand', PINCH_HAND)

        def use_camera(self):
            self.source.setText(str(self.camera.value()))
            self.start()

        def open_video(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Video', str(ROOT), 'Videos (*.mp4 *.avi *.mkv *.mov)')
            if path:
                self.source.setText(path)

        def start(self):
            if self.worker and self.worker.isRunning():
                self.logs.appendPlainText('Stop the current source before changing it. Reset keeps the same source.')
                return
            value = self.source.text().strip()
            if not value:
                return
            source = int(value) if value.isdigit() else value
            if isinstance(source, int) and sys.platform.startswith('linux'):
                devices = capture_devices()
                if source not in [index for index, _ in devices]:
                    options = ', '.join(str(index) for index, _ in devices) or 'none (check USB connection and access permissions)'
                    message = f'Camera {source} is not an available video-capture device. Available camera indices: {options}.'
                    self.logs.appendPlainText('ERROR: '+message)
                    self.status.setText(message)
                    return
            folder = self.args.output/datetime.now(TIMEZONE).strftime('%Y%m%d_%H%M%S_%f')
            previous = self.worker
            self.worker = Worker(source, folder, self.args.fps, self.hand_fallback.isChecked(), self.args.box)
            if previous:
                previous.deleteLater()
            self.worker.result.connect(self.show_result)
            self.worker.log.connect(self.logs.appendPlainText)
            self.worker.notice.connect(self.status.setText)
            self.worker.failure.connect(self.show_error)
            self.table.setRowCount(0)
            self.worker.start()

        def show_error(self, message):
            self.logs.appendPlainText('ERROR: '+message)
            self.status.setText('ERROR: '+message)

        def stop(self):
            if self.worker:
                self.worker.stop()

        def reset(self):
            if self.worker and self.worker.isRunning():
                self.worker.reset_event.set()
            else:
                self.logs.appendPlainText('Start a source first. Reset keeps clips and assigns new IDs.')

        def show_result(self, result):
            self.source_size = result['source_size']
            rgb = cv2.cvtColor(result['frame'], cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            self.last_image = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
            self.paint_frame()
            if not self.draw_button.isChecked():
                self.status.setText(result['status'])
            self.table.setRowCount(len(result['rows']))
            for row, values in enumerate(result['rows']):
                recording = values[3].startswith('REC')
                for col, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    if recording and col in (0, 3):
                        item.setBackground(QtGui.QColor('#fee2e2'))
                        item.setForeground(QtGui.QColor('#991b1b'))
                        font = item.font(); font.setBold(True); item.setFont(font)
                    self.table.setItem(row, col, item)

        def paint_frame(self):
            if self.last_image is not None:
                self.video.setPixmap(QtGui.QPixmap.fromImage(self.last_image).scaled(
                    self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        def resizeEvent(self, event):
            self.paint_frame()
            super().resizeEvent(event)

        def closeEvent(self, event):
            self.stop()
            if self.worker and self.worker.isRunning():
                self.logs.appendPlainText('Stopping capture/inference. Close again after it stops.')
                event.ignore()
                QtCore.QTimer.singleShot(400, self.close)
                return
            super().closeEvent(event)

    return Window


def self_test():
    """No weights/camera: the recording trigger state machine on synthetic tracks."""
    import tempfile
    frame = np.full((240, 400, 3), 200, np.uint8)
    box = (250, 60, 350, 180)
    bench, in_box = (100., 120.), (300., 120.)

    def track(tid, pos):
        return SimpleNamespace(id=tid, pos=np.array(pos), vel=np.zeros(2), state='OK', det_seen=0., det_box=(0, 0, 1, 1), box=(0, 0, 1, 1))

    def det(c):
        return SimpleNamespace(c=np.array(c), box=(int(c[0])-10, int(c[1])-10, int(c[0])+10, int(c[1])+10))

    def tracker(*tracks):
        return SimpleNamespace(tracks=list(tracks), roi_dist=lambda p: 1e9, id_model=None, identify=None)

    def on_bench(tr, t, pos=bench):
        tr.state, tr.det_seen, tr.pos = 'OK', t, np.array(pos)

    def times(start, stop):
        return [round(i*.1, 3) for i in range(round(start*10), round(stop*10))]

    def states(events, jewel='J1'):   # recording events only (HOME_MOVED is informational)
        return [e['state'] for e in events if e.get('jewel') == jewel and e['state'] != 'HOME_MOVED']

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        assert small_frame(np.zeros((1080, 1920, 3), np.uint8)).shape[:2] == (540, 960)

        # 1. Lifted (hidden in the hand), raw box appears inside the box: start after REC_START_S
        #    with the pre-roll; set back on the bench and still for REC_STOP_SETTLE_S: stop.
        events = []
        rec = BoxRecorder(folder/'one', 10., events.append, box=box)
        j1 = track(1, bench); trk = tracker(j1)
        for t in times(0, .5):
            on_bench(j1, t); rec.update(trk, [], frame, t)
        assert not rec.rec[1].picked and rec.rec[1].rested and not rec.preroll
        j1.state = 'OCCLUDED'
        for t in times(.5, 1.1):
            rec.update(trk, [], frame, t)
        assert rec.rec[1].picked and not states(events), 'A mere lift must not start a clip'
        for t in times(1.1, 1.5):
            rec.update(trk, [det(in_box)], frame, t)
        assert rec.rec[1].writer is None and rec.rec[1].arm_since == 1.1
        for t in times(1.5, 2.6):
            rec.update(trk, [det(in_box)], frame, t)
        assert states(events) == ['REC_STARTED'] and events[0]['preroll_frames'] == REC_PREROLL_FRAMES
        assert 1.6 <= events[0]['time'] <= 1.7 and events[0]['reason'] == 'brought into the box'
        for t in times(2.6, 4.1):
            on_bench(j1, t); rec.update(trk, [], frame, t)
        assert states(events) == ['REC_STARTED', 'REC_SAVED'] and events[1]['reason'] == 'placed back'
        for t in times(4.1, 5.):
            on_bench(j1, t); rec.update(trk, [], frame, t)
        assert states(events) == ['REC_STARTED', 'REC_SAVED'] and not rec.rec[1].lock
        clip = folder/'one'/events[1]['clip']
        cap = cv2.VideoCapture(str(clip)); count = 0
        while cap.read()[0]:
            count += 1
        cap.release()
        assert count >= REC_PREROLL_FRAMES+15, f'clip has {count} frames'

        # 2. A flicker inside the box shorter than REC_START_S starts nothing.
        events = []
        rec = BoxRecorder(folder/'two', 10., events.append, box=box)
        j1 = track(1, bench); trk = tracker(j1)
        for t in times(0, .5):
            on_bench(j1, t); rec.update(trk, [], frame, t)
        j1.state = 'OCCLUDED'
        for t in times(.5, 3.):
            rec.update(trk, [det(in_box)] if 1. <= t < 1.3 else [], frame, t)
        assert not events, 'A detection flicker in the box must not start a clip'

        # 3. While recording, MISSING / LEFT_ROI for REC_STOP_GONE_S stops the clip ("left").
        for t in times(3., 3.7):
            rec.update(trk, [det(in_box)], frame, t)
        assert states(events) == ['REC_STARTED']
        j1.state = 'LEFT_ROI'
        for t in times(3.7, 5.4):
            rec.update(trk, [], frame, t)
        assert states(events) == ['REC_STARTED', 'REC_SAVED'] and events[1]['reason'] == 'left'
        # One clip per trip: back in the box without resting on the bench first -> no new clip.
        j1.state = 'OCCLUDED'
        for t in times(5.4, 6.5):
            rec.update(trk, [det(in_box)], frame, t)
        assert states(events) == ['REC_STARTED', 'REC_SAVED'] and rec.rec[1].lock

        # 4. Carried into the box in plain view (tracked all the way) also starts a clip;
        #    a jewel lying in the box from the start never does, nor is its box credited to J1.
        events = []
        rec = BoxRecorder(folder/'four', 10., events.append, box=box)
        j1, j2 = track(1, bench), track(2, (280., 100.))
        trk = tracker(j1, j2)
        for t in times(0, .5):
            on_bench(j1, t); on_bench(j2, t, (280., 100.))
            j2.det_box = det((280., 100.)).box
            rec.update(trk, [det((280., 100.))], frame, t)
        j1.state = 'OCCLUDED'
        for t in times(.5, 1.5):   # J1 hidden in hand; J2's own box inside the target box
            on_bench(j2, t, (280., 100.))
            rec.update(trk, [det((280., 100.))], frame, t)
        assert not events, "A jewel lying in the box must not trigger another jewel's clip"
        for t in times(1.5, 2.5):  # J1 now visible and tracked inside the box
            on_bench(j1, t, (320., 150.)); j1.det_box = det((320., 150.)).box
            on_bench(j2, t, (280., 100.))
            rec.update(trk, [det((280., 100.)), det((320., 150.))], frame, t)
        assert states(events, 'J1') == ['REC_STARTED'] and not states(events, 'J2')

        # 5. Hidden in the fingers: thumb/index pinch inside the box stands in for the jewel.
        class Pinch:
            def detect(self, crop):
                points = np.zeros((21, 2), np.float32)
                points[4], points[8] = (296., 120.), (304., 120.)   # crop == full frame here
                return [SimpleNamespace(points=points, scale=100.)]
        events = []
        rec = BoxRecorder(folder/'five', 10., events.append, hands=Pinch(), box=box)
        j1 = track(1, bench); trk = tracker(j1)
        for t in times(0, .5):
            on_bench(j1, t); rec.update(trk, [], frame, t)
        j1.state = 'OCCLUDED'
        for t in times(.5, 1.3):
            rec.update(trk, [], frame, t)
        assert states(events) == ['REC_STARTED'] and events[0]['reason'] == 'pinched in the box'
        j1.state = 'MISSING'       # out of sight but pinched at the box: keep recording
        for t in times(1.3, 3.5):
            rec.update(trk, [], frame, t)
        assert states(events) == ['REC_STARTED']
        rec.stop_all('test end')
        assert states(events) == ['REC_STARTED', 'REC_SAVED']
        vis = frame.copy(); rec.draw(vis, 3.6)

        # 5a. Bangle gripped from above: thumb under it, fingers over it - tips ~ the bangle's width apart.
        #     That wide grip starts a clip for a bangle-sized jewel, but is no pinch for a small stud.
        class Grip:
            def detect(self, crop):
                points = np.zeros((21, 2), np.float32)
                points[4], points[8] = (300., 75.), (300., 165.)    # 90 px apart, midpoint in the box
                return [SimpleNamespace(points=points, scale=100.)]  # tight-pinch limit: 35 px
        global HAND_PINCH_PX
        fixed_px, HAND_PINCH_PX = HAND_PINCH_PX, None   # automatic (size-scaled) limit for this case
        for width, expect in ((70., True), (20., False)):
            events = []
            rec = BoxRecorder(folder/f'grip{int(width)}', 10., events.append, hands=Grip(), box=box)
            j1 = track(1, bench); j1.det_box = (100-width/2, 120-width/2, 100+width/2, 120+width/2)
            trk = tracker(j1)
            for t in times(0, .5):
                on_bench(j1, t); rec.update(trk, [], frame, t)
            assert rec.rec[1].size == width
            j1.state = 'OCCLUDED'
            for t in times(.5, 1.5):
                rec.update(trk, [], frame, t)
            assert bool(states(events)) == expect, f'{width:.0f} px jewel, 90 px grip: expected start={expect}'
            if expect:
                assert events[0]['reason'] == 'pinched in the box'
                rec.stop_all('test end')
        HAND_PINCH_PX = fixed_px

        # 5c. Pinch is the main trigger: jewel carried over the box (gold detected in it) with the hand
        #     open -> no clip; the same hand pinching within HAND_PINCH_PX -> clip "pinched in the box".
        class Hover:
            gap = 300.
            def detect(self, crop):
                points = np.full((21, 2), (300., 120.), np.float32)   # hand over the box
                points[4], points[8] = (300., 120.-self.gap/2), (300., 120.+self.gap/2)
                return [SimpleNamespace(points=points, scale=100.)]
        limit = 150. if HAND_PINCH_PX is None else float(HAND_PINCH_PX)
        hover = Hover(); hover.gap = limit*2.
        events = []
        rec = BoxRecorder(folder/'hover', 10., events.append, hands=hover, box=box)
        j1 = track(1, bench); trk = tracker(j1)
        for t in times(0, .5):
            on_bench(j1, t); rec.update(trk, [], frame, t)
        for t in times(.5, 2.5):   # visible, tracked inside the box, its gold box in the box, hand open
            on_bench(j1, t, in_box); j1.det_box = det(in_box).box
            rec.update(trk, [det(in_box)], frame, t)
        assert rec.rec[1].picked and not events, 'Carrying the jewel over the box without a pinch must not record'
        hover.gap = limit*.5
        for t in times(2.5, 3.5):
            on_bench(j1, t, in_box); j1.det_box = det(in_box).box
            rec.update(trk, [det(in_box)], frame, t)
        assert states(events) == ['REC_STARTED'] and events[0]['reason'] == 'pinched in the box'
        rec.stop_all('test end')

        # 5d. Gold second: hand model on but no hand seen -> gold in the box still starts the clip.
        class NoHand:
            def detect(self, crop):
                return []
        events = []
        rec = BoxRecorder(folder/'nohand', 10., events.append, hands=NoHand(), box=box)
        j1 = track(1, bench); trk = tracker(j1)
        for t in times(0, .5):
            on_bench(j1, t); rec.update(trk, [], frame, t)
        j1.state = 'OCCLUDED'
        for t in times(.5, 1.5):
            rec.update(trk, [det(in_box)], frame, t)
        assert states(events) == ['REC_STARTED'] and events[0]['reason'] == 'gold in the box - no hand seen'
        rec.stop_all('test end')

        # 5b. Box drawn over jewels already lying there: none of them starts a clip.
        events = []
        rec = BoxRecorder(folder/'redraw', 10., events.append)
        j1, j2 = track(1, in_box), track(2, (280., 100.))
        trk = tracker(j1, j2)
        for t in times(0, .5):     # resting on the bench before the box exists
            on_bench(j1, t, in_box); on_bench(j2, t, (280., 100.)); rec.update(trk, [], frame, t)
        assert rec.rec[1].rested and rec.rec[2].rested
        rec.set_box(box)
        for t in times(.5, 2.):
            on_bench(j1, t, in_box); on_bench(j2, t, (280., 100.)); rec.update(trk, [], frame, t)
        assert not events, 'Drawing the box over resting jewels must not start their clips'

        # 5e. Placed back with a JUMPY detection box (+-15 px every frame, a lost detection now and then):
        #     still counts as lying still -> the clip stops about 1 s after it is set down.
        events = []
        rec = BoxRecorder(folder/'jitter', 10., events.append, box=box)
        j1 = track(1, bench); trk = tracker(j1)
        for t in times(0, .5):
            on_bench(j1, t); rec.update(trk, [], frame, t)
        j1.state = 'OCCLUDED'
        for t in times(.5, 1.6):
            rec.update(trk, [det(in_box)], frame, t)
        assert states(events) == ['REC_STARTED']
        rng = np.random.default_rng(0)
        for i, t in enumerate(times(3., 6.)):
            if i % 5 == 3:                  # detection lost for a frame
                j1.state = 'OCCLUDED'
            else:
                on_bench(j1, t, np.array(bench)+rng.uniform(-15, 15, 2))
                j1.vel = rng.uniform(-300, 300, 2)   # jumpy box: huge momentary speed
            rec.update(trk, [], frame, t)
            if len(states(events)) > 1:
                break
        assert states(events) == ['REC_STARTED', 'REC_SAVED'] and events[1]['reason'] == 'placed back'
        assert events[1]['time'] <= 3.+REST_WINDOW_S+REC_STOP_SETTLE_S+.4, f"stopped at {events[1]['time']}"

        # 5f. PARALLEL: J1 is recording (pinched in the box); J2 hidden in another hand does not take the pinch,
        #     but once J2 is surely in the box it gets its own clip at the same time.
        events = []
        rec = BoxRecorder(folder/'parallel', 10., events.append, hands=Pinch(), box=box)
        j2_bench = (150., 200.)
        j1, j2 = track(1, bench), track(2, j2_bench); trk = tracker(j1, j2)
        for t in times(0, .5):
            on_bench(j1, t); on_bench(j2, t, j2_bench); rec.update(trk, [], frame, t)
        j1.state = 'OCCLUDED'
        for t in times(.5, 1.3):
            on_bench(j2, t, j2_bench); rec.update(trk, [], frame, t)
        assert states(events, 'J1') == ['REC_STARTED'] and not states(events, 'J2')
        j1.state, j2.state = 'MISSING', 'OCCLUDED'
        for t in times(1.3, 2.5):          # J2 picked up, hidden: must not steal J1's pinch
            rec.update(trk, [], frame, t)
        assert not states(events, 'J2'), 'A hidden second jewel must not start on the first jewel\'s pinch'
        for t in times(2.5, 3.5):          # J2 carried into the box in view, tracked there
            on_bench(j2, t, (320., 150.)); j2.det_box = det((320., 150.)).box
            rec.update(trk, [det((320., 150.))], frame, t)
        assert states(events, 'J2') == ['REC_STARTED'] and states(events, 'J1') == ['REC_STARTED']
        assert rec.rec[1].writer is not None and rec.rec[2].writer is not None, 'both clips record in parallel'
        rec.stop_all('test end')

        # 5g. TWO HANDS: only PINCH_HAND counts. The left hand pinching in the box starts nothing and does not
        #     block gold; the right hand pinching starts the clip.
        class TwoHands:
            right_gap = 300.
            def detect(self, crop):
                left = np.full((21, 2), (290., 110.), np.float32)
                left[4], left[8] = (288., 110.), (292., 110.)               # left: tight pinch in the box
                right = np.full((21, 2), (310., 130.), np.float32)
                right[4], right[8] = (310., 130.-self.right_gap/2), (310., 130.+self.right_gap/2)
                return [SimpleNamespace(points=left, scale=100., handedness='Left'),
                        SimpleNamespace(points=right, scale=100., handedness='Right')]
        hand_before = PINCH_HAND
        set_pinch_hand('Right')
        limit = 150. if HAND_PINCH_PX is None else float(HAND_PINCH_PX)
        two = TwoHands(); two.right_gap = limit*2.
        events = []
        rec = BoxRecorder(folder/'twohands', 10., events.append, hands=two, box=box)
        j1 = track(1, bench); trk = tracker(j1)
        for t in times(0, .5):
            on_bench(j1, t); rec.update(trk, [], frame, t)
        j1.state = 'OCCLUDED'
        for t in times(.5, 2.):
            rec.update(trk, [], frame, t)
        assert not events, 'The left hand\'s pinch must not start a clip'
        two.right_gap = limit*.5
        for t in times(2., 3.):
            rec.update(trk, [], frame, t)
        assert states(events) == ['REC_STARTED'] and events[0]['reason'] == 'pinched in the box'
        vis = frame.copy(); rec.draw(vis, 2.9)
        rec.stop_all('test end')
        # Only the left hand at the box (right hand away): gold in the box still starts (left does not block it).
        class LeftOnly:
            def detect(self, crop):
                left = np.full((21, 2), (290., 110.), np.float32)
                return [SimpleNamespace(points=left, scale=100., handedness='Left')]
        events = []
        rec = BoxRecorder(folder/'leftonly', 10., events.append, hands=LeftOnly(), box=box)
        j1 = track(1, bench); trk = tracker(j1)
        for t in times(0, .5):
            on_bench(j1, t); rec.update(trk, [], frame, t)
        j1.state = 'OCCLUDED'
        for t in times(.5, 1.5):
            rec.update(trk, [det(in_box)], frame, t)
        assert states(events) == ['REC_STARTED'] and events[0]['reason'] == 'gold in the box - no hand seen'
        rec.stop_all('test end')
        set_pinch_hand(hand_before)

        # 7. HOME POSITION (tracks with segmented masks, a white bench, gold patches, a hand covering homes).
        GOLD, SKIN = (0, 170, 220), (60, 90, 140)

        def scene(gold=(), cover=()):
            f = np.full((240, 400, 3), 240, np.uint8)
            for x, y in gold:
                cv2.rectangle(f, (int(x)-12, int(y)-12), (int(x)+11, int(y)+11), GOLD, -1)
            for x, y in cover:
                cv2.rectangle(f, (int(x)-20, int(y)-20), (int(x)+20, int(y)+20), SKIN, -1)
            return f

        def mdet(c, r=12):
            x, y = int(c[0]), int(c[1])
            return SimpleNamespace(c=np.array(c, float), box=(x-r, y-r, x+r, y+r),
                                   mask=np.ones((2*r, 2*r), np.uint8), origin=(x-r, y-r))

        def see(tr, d, t):   # the tracker matched detection d to track tr this frame
            tr.state, tr.det_seen, tr.pos, tr.det_box = 'OK', t, np.array(d.c, float), d.box
            tr.det_mask, tr.det_origin = d.mask, d.origin

        class HandAtBox:     # the right hand pinching in the target box when `on`
            on = False
            def detect(self, crop):
                if not self.on:
                    return []
                points = np.full((21, 2), (300., 120.), np.float32)
                points[4], points[8] = (296., 120.), (304., 120.)
                return [SimpleNamespace(points=points, scale=100.)]
        A, B = (100., 120.), (160., 120.)

        def two_at_home(rec, j1, j2, upto=.5):
            for t in times(0, upto):
                da, db = mdet(A), mdet(B)
                see(j1, da, t); see(j2, db, t)
                rec.update(trk, [da, db], scene(gold=[A, B]), t)
            assert rec.rec[1].home_state == rec.rec[2].home_state == 'HOME'

        # 7a. J1 is lifted while J2 lies under the same hand; the tracker then puts J1's id on J2's gold.
        #     J2's home still shows gold -> J2 never counts as picked; J1's home shows empty bench -> J1 left.
        #     The pinch in the box records J1 - not J2 - and J1's id lying on J2's home is no "placed back".
        hand = HandAtBox(); events = []
        rec = BoxRecorder(folder/'home_swap', 10., events.append, hands=hand, box=box)
        j1, j2 = track(1, A), track(2, B); trk = tracker(j1, j2)
        two_at_home(rec, j1, j2)
        j1.state = j2.state = 'OCCLUDED'
        for t in times(.5, 1.):        # the hand over both homes
            rec.update(trk, [], scene(cover=[A, B]), t)
        assert rec.rec[1].home_state == rec.rec[2].home_state == 'COVERED'
        hand.on = True
        for t in times(1., 3.):        # hand at the box; J2 visible at home again, but matched to J1's id
            db = mdet(B); see(j1, db, t); j2.state = 'MISSING'
            rec.update(trk, [db], scene(gold=[B]), t)
        assert rec.rec[1].confirmed_left() and rec.rec[2].home_state == 'HOME'
        assert states(events, 'J1') == ['REC_STARTED'] and not states(events, 'J2'), events
        assert events[0]['reason'] == 'pinched in the box' and events[0]['provisional'] is False
        vis = scene(gold=[B]); rec.draw(vis, 2.9)
        assert rec.status(j1, 2.9)[0].endswith('left home') and rec.status(j2, 2.9)[0].endswith('at home')
        rec.stop_all('test end')

        # 7b. False trigger: the hand covers J1 at home and a pinch is seen in the box -> provisional clip;
        #     the hand lifts and J1 is still lying at home -> the clip is NOT deleted: it goes on unassigned and,
        #     with no jewel ever placed away from its home, is saved as UNASSIGNED_...
        hand = HandAtBox(); events = []
        rec = BoxRecorder(folder/'home_false', 10., events.append, hands=hand, box=box)
        j1 = track(1, A); trk = tracker(j1)
        for t in times(0, .5):
            da = mdet(A); see(j1, da, t); rec.update(trk, [da], scene(gold=[A]), t)
        j1.state, hand.on = 'OCCLUDED', True
        for t in times(.5, 1.5):
            rec.update(trk, [], scene(cover=[A]), t)
        assert states(events) == ['REC_STARTED'] and events[0]['provisional'] is True
        clip = folder/'home_false'/events[0]['clip']
        hand.on = False
        for t in times(1.5, 6.):       # J1 lying at home, no hand: everything at rest -> the J? clip ends
            da = mdet(A); see(j1, da, t); rec.update(trk, [da], scene(gold=[A]), t)
            if events[-1]['state'] == 'REC_SAVED_UNASSIGNED':
                break
        assert [e['state'] for e in events] == ['REC_STARTED', 'REC_UNASSIGNED', 'REC_SAVED_UNASSIGNED'], events
        assert rec.rec[1].writer is None and rec.rec[1].home_state == 'HOME' and not rec.orphans
        last = events[-1]
        assert last['clip'].startswith('UNASSIGNED_') and last['time'] <= 4.5, last
        assert (folder/'home_false'/last['clip']).exists() and not clip.exists()

        # 7i. The user's case: J1 is lifted with the forearm over its home the whole time (never seen empty),
        #     its gold is seen in the box (clip, provisional), then it is put back on the VERY spot it came
        #     from. Nobody else is away -> it went and came back: saved as J1's clip promptly, never J?.
        hand = HandAtBox(); events = []
        rec = BoxRecorder(folder/'went_and_back', 10., events.append, hands=hand, box=box)
        j1, j2 = track(1, A), track(2, B); trk = tracker(j1, j2)
        two_at_home(rec, j1, j2)
        G = (300., 120.)
        j1.state = 'OCCLUDED'
        for t in times(.5, 3.):        # forearm over J1's home; J1's gold in the box (hand model misses it)
            db, dg = mdet(B), mdet(G); see(j2, db, t)
            rec.update(trk, [db, dg], scene(gold=[B, G], cover=[A]), t)
        assert states(events) == ['REC_STARTED'] and events[0]['provisional'] is True, events
        for t in times(3., 6.):        # put back exactly at A
            da, db = mdet(A), mdet(B); see(j1, da, t); see(j2, db, t)
            rec.update(trk, [da, db], scene(gold=[A, B]), t)
            if len(states(events)) > 1:
                break
        assert states(events) == ['REC_STARTED', 'REC_SAVED'], events
        assert events[-1]['jewel'] == 'J1' and events[-1]['clip'].startswith('J1_'), events
        assert events[-1]['time'] <= 3.+REST_WINDOW_S+.5, events[-1]['time']
        assert (folder/'went_and_back'/events[-1]['clip']).exists() and not rec.orphans

        # 7j. The user's case: J1 (recording) is set down at P; before its clip stops, the hand reaching for J2
        #     and into the box passes its WRIST over P (J1 hidden, its id lost) while J2 is pinched in the box.
        #     Uncovered, J1's gold lies at P: J1's clip stops "placed back" and stays J1's (never handed to J2 or
        #     made J?), while J2's own clip goes on; J2 set down -> J2's clip stops.
        class Reach:                   # pinching in the box; the rest of the hand (palm / wrist) over `over`
            pinch, over = False, None
            def detect(self, crop):
                if not self.pinch:
                    return []
                points = np.full((21, 2), self.over if self.over is not None else (300., 120.), np.float32)
                points[4], points[8], points[12] = (296., 120.), (304., 120.), (300., 126.)
                return [SimpleNamespace(points=points, scale=100.)]
        hand = Reach(); events = []
        rec = BoxRecorder(folder/'wrist_over', 10., events.append, hands=hand, box=box)
        j1, j2 = track(1, A), track(2, B); trk = tracker(j1, j2)
        two_at_home(rec, j1, j2)
        P, Q = (60., 200.), (160., 200.)
        j1.state, hand.pinch = 'OCCLUDED', True
        for t in times(.5, 2.):        # J1 pinched in the box
            db = mdet(B); see(j2, db, t); rec.update(trk, [db], scene(gold=[B]), t)
        assert states(events) == ['REC_STARTED'], events
        hand.pinch = False
        for t in times(2., 2.5):       # J1 set down at P (rests only briefly)
            dp, db = mdet(P), mdet(B); see(j1, dp, t); see(j2, db, t)
            rec.update(trk, [dp, db], scene(gold=[P, B]), t)
        assert states(events) == ['REC_STARTED'] and rec.rec[1].home_t > rec.rec[1].start_t, events
        j1.state = j2.state = 'OCCLUDED'
        hand.pinch, hand.over = True, P
        for t in times(2.5, 3.):       # the hand over P and B (takes J2)
            rec.update(trk, [], scene(cover=[P, B]), t)
        for t in times(3., 4.5):       # J2 in the box, pinched; J2's home empty; the wrist still over P
            rec.update(trk, [], scene(cover=[P]), t)
        assert states(events, 'J2') == ['REC_STARTED'], events
        hand.over = None
        j1.state = 'MISSING'
        stopped = None
        for t in times(4.5, 6.):       # the wrist off P: J1's gold lies there (the tracker keeps it MISSING)
            rec.update(trk, [mdet(P)], scene(gold=[P]), t)
            if stopped is None and 'REC_SAVED' in states(events):
                stopped = t
        assert states(events) == ['REC_STARTED', 'REC_SAVED'] and stopped <= 4.5+REST_WINDOW_S+.3, (stopped, events)
        assert events[-1 if events[-1]['jewel'] == 'J1' else -2]['clip'].startswith('J1_')
        assert not any(e['state'] in ('REC_REASSIGNED', 'REC_UNASSIGNED') for e in events), events
        assert rec.rec[2].writer is not None, "J2's clip goes on"
        hand.pinch = False
        for t in times(6., 8.):        # J2 set down at Q
            dq = mdet(Q); see(j2, dq, t); rec.update(trk, [mdet(P), dq], scene(gold=[P, Q]), t)
            if 'REC_SAVED' in states(events, 'J2'):
                break
        assert states(events, 'J2') == ['REC_STARTED', 'REC_SAVED'], events
        assert [e['jewel'] for e in events if e['state'] == 'REC_SAVED'] == ['J1', 'J2']

        # 7d. The user's case: the clip starts for J2 (wrong id), J2 turns out still at home, and J1's home is
        #     never seen empty (a sleeve over it) -> clip unassigned. J1 is then PLACED at a new spot: it moved
        #     from its home, so the clip is saved as J1's.
        #     Also set down only ~30 px (COG) from where it lay before: still "moved from its home pose".
        for case, new_spot in (('far', (60., 200.)), ('near', (A[0]+30., A[1]))):
            hand = HandAtBox(); events = []
            rec = BoxRecorder(folder/f'home_placed_{case}', 10., events.append, hands=hand, box=box)
            j1, j2 = track(1, A), track(2, B); trk = tracker(j1, j2)
            two_at_home(rec, j1, j2)
            j1.state = 'OCCLUDED'
            for t in times(.5, .7):
                db = mdet(B); see(j2, db, t); rec.update(trk, [db], scene(gold=[B], cover=[A]), t)
            j2.state, hand.on = 'OCCLUDED', True
            for t in times(.7, 1.5):
                rec.update(trk, [], scene(cover=[A, B]), t)
            assert states(events, 'J2') == ['REC_STARTED'] and events[0]['provisional'] is True, events
            hand.on = False
            for t in times(1.5, 3.):   # J2 lying at home; J1's home still covered (sleeve), J1 in the hand
                db = mdet(B); see(j2, db, t); rec.update(trk, [db], scene(gold=[B], cover=[A]), t)
            assert [e['state'] for e in events] == ['REC_STARTED', 'REC_UNASSIGNED'], events
            done = None
            for t in times(3., 6.):    # J1 set down at a new pose
                db, dc = mdet(B), mdet(new_spot); see(j2, db, t); see(j1, dc, t)
                rec.update(trk, [db, dc], scene(gold=[B, new_spot], cover=[A]), t)
                if done is None and any(e['state'] == 'REC_SAVED' for e in events):
                    done = t
            saved = [e for e in events if e['state'] == 'REC_SAVED']
            assert saved and saved[0]['jewel'] == 'J1' and saved[0]['clip'].startswith('J1_'), (case, events)
            assert done <= 3.+REST_WINDOW_S+.2, f'{case}: J? finalised only at {done}'
            assert (folder/f'home_placed_{case}'/saved[0]['clip']).exists() and not rec.orphans
            assert rec.rec[1].home_state == 'HOME' and float(np.hypot(*(rec.rec[1].home.pos-np.array(new_spot)))) < 1
            assert len(saved) == 1, 'one placement, one clip'

        # 7e. PARALLEL PINCH: J1 is recording (pinched in the box). J2 still at home never takes the pinch;
        #     once J2 is lifted (its home shows empty bench) the pinch starts J2's own clip at the same time.
        hand = HandAtBox(); events = []
        rec = BoxRecorder(folder/'home_parallel', 10., events.append, hands=hand, box=box)
        j1, j2 = track(1, A), track(2, B); trk = tracker(j1, j2)
        two_at_home(rec, j1, j2)
        j1.state, hand.on = 'OCCLUDED', True
        for t in times(.5, 2.):        # J1 in the hand, pinched in the box; J2 lying at home
            db = mdet(B); see(j2, db, t); rec.update(trk, [db], scene(gold=[B]), t)
        assert states(events, 'J1') == ['REC_STARTED'] and not states(events, 'J2'), events
        j2.state = 'OCCLUDED'
        for t in times(2., 3.2):       # J2 lifted too: its home empty; the pinch goes on in the box
            rec.update(trk, [], scene(), t)
        assert states(events, 'J2') == ['REC_STARTED'], events
        assert rec.rec[1].writer is not None and rec.rec[2].writer is not None, 'both record in parallel'
        rec.stop_all('test end')

        # 7g. HELD STILL IN THE HAND is not "set down": J1 hovers in the hand right next to the box (visible,
        #     barely moving) -> its trip and home are kept; pinched in the box -> clip; held still outside the
        #     box while recording -> keeps recording; released (no hand on it) -> placed back.
        class HandAt:
            where, pinch = None, False
            def detect(self, crop):
                if self.pinch:
                    points = np.full((21, 2), (300., 120.), np.float32)
                    points[4], points[8] = (296., 120.), (304., 120.)
                    return [SimpleNamespace(points=points, scale=100.)]
                if self.where is None:
                    return []
                points = np.full((21, 2), self.where, np.float32)
                points[4], points[8] = (self.where[0]-30., self.where[1]-25.), (self.where[0]+30., self.where[1]+25.)
                return [SimpleNamespace(points=points, scale=100.)]
        hand = HandAt(); events = []
        rec = BoxRecorder(folder/'held', 10., events.append, hands=hand, box=box)
        j1 = track(1, A); trk = tracker(j1)
        for t in times(0, .5):
            da = mdet(A); see(j1, da, t); rec.update(trk, [da], scene(gold=[A]), t)
        hover = (box[0]-25., 120.)
        j1.state, hand.where = 'OCCLUDED', A
        for t in times(.5, 1.):        # the hand takes it
            rec.update(trk, [], scene(cover=[A]), t)
        hand.where = hover
        for t in times(1., 2.5):       # held still in the hand next to the box
            dh = mdet((hover[0]+rng.uniform(-3, 3), hover[1])); see(j1, dh, t)
            rec.update(trk, [dh], scene(gold=[hover]), t)
        assert rec.rec[1].picked and float(np.hypot(*(rec.rec[1].home.pos-np.array(A)))) < 1, \
            'held still in the hand must not count as set down'
        assert not events, events
        j1.state, hand.where, hand.pinch = 'OCCLUDED', None, True
        for t in times(2.5, 3.5):      # into the box, pinched (hidden in the fingers)
            rec.update(trk, [], scene(), t)
        assert states(events) == ['REC_STARTED'] and events[0]['reason'] == 'pinched in the box', events
        hand.pinch, hand.where = False, hover
        for t in times(3.5, 5.5):      # out of the box, held still in the hand: still being worked
            dh = mdet(hover); see(j1, dh, t); rec.update(trk, [dh], scene(gold=[hover]), t)
        assert states(events) == ['REC_STARTED'], 'held still in the hand must not stop the clip'
        hand.where = None
        for t in times(5.5, 8.):       # let go: lying there now
            dh = mdet(hover); see(j1, dh, t); rec.update(trk, [dh], scene(gold=[hover]), t)
            if len(states(events)) > 1:
                break
        stop = [e for e in events if e['state'] == 'REC_SAVED']
        assert states(events) == ['REC_STARTED', 'REC_SAVED'] and stop[0]['reason'] == 'placed back', events
        assert stop[0]['time'] <= 5.5+HAND_HOLD_S+REST_WINDOW_S+REC_STOP_SETTLE_S+.4, stop[0]['time']

        # 7h. SET DOWN BUT THE TRACKER KEEPS IT MISSING: its raw gold box lies there unclaimed (grey on the
        #     feed) -> still "placed back" from that box; meanwhile J2's track is lost while J2 lies at home -
        #     J2's unclaimed gold at its own home must not stop J1's clip.
        hand = HandAtBox(); events = []
        rec = BoxRecorder(folder/'missing_placed', 10., events.append, hands=hand, box=box)
        j1, j2 = track(1, A), track(2, B); trk = tracker(j1, j2)
        two_at_home(rec, j1, j2)
        j1.state, hand.on = 'OCCLUDED', True
        for t in times(.5, 1.6):       # J1 in the hand, pinched in the box -> recording
            db = mdet(B); see(j2, db, t); rec.update(trk, [db], scene(gold=[B]), t)
        assert states(events) == ['REC_STARTED'], events
        j1.state, j2.state = 'MISSING', 'MISSING'   # the tracker loses both
        for t in times(1.6, 3.5):      # J1 still worked in the box; J2's gold lies at its home, unclaimed
            rec.update(trk, [mdet(B)], scene(gold=[B]), t)
        assert states(events) == ['REC_STARTED'], "J2's unclaimed gold at its home is not J1 set down"
        hand.on = False
        P = (60., 200.)
        for t in times(3.5, 6.):       # J1 set down at P; the tracker keeps J1 MISSING (no re-attach)
            dp = mdet((P[0]+rng.uniform(-6, 6), P[1])); rec.update(trk, [mdet(B), dp], scene(gold=[B, P]), t)
            if 'REC_SAVED' in states(events):
                break
        stop = [e for e in events if e['state'] == 'REC_SAVED']
        assert stop and stop[0]['reason'] == 'placed back', events
        assert stop[0]['time'] <= 3.5+REST_WINDOW_S+REC_STOP_SETTLE_S+.5, stop[0]['time']
        assert float(np.hypot(*(rec.rec[1].home.pos-np.array(P)))) < 10 and rec.rec[2].home_state == 'HOME'

        # 7f. PLACED with a FLICKERING box, right next to the target box: the centre jumps 70 px every 4th
        #     frame, the detection is lost every 5th, the box size flickers -> still "placed back", promptly.
        events = []
        rec = BoxRecorder(folder/'flicker', 10., events.append, box=box)
        spot = (box[0]-25., 120.)      # 25 px left of the target box
        j1 = track(1, spot); trk = tracker(j1)
        for t in times(0, .5):
            on_bench(j1, t, spot); rec.update(trk, [], frame, t)
        j1.state = 'OCCLUDED'
        for t in times(.5, 1.6):
            rec.update(trk, [det(in_box)], frame, t)
        assert states(events) == ['REC_STARTED']
        rng = np.random.default_rng(1)
        for i, t in enumerate(times(3., 7.)):
            if i % 5 == 4:
                j1.state = 'OCCLUDED'      # detection lost
            else:
                jump = (70., 0.) if i % 4 == 1 else (0., 0.)
                on_bench(j1, t, np.array(spot)+rng.uniform(-12, 12, 2)+jump)
                w = 20+(i % 3)*8
                j1.det_box = (spot[0]-w, 110, spot[0]+w, 130)
            rec.update(trk, [], frame, t)
            if len(states(events)) > 1:
                break
        assert states(events) == ['REC_STARTED', 'REC_SAVED'] and events[1]['reason'] == 'placed back', events
        assert events[1]['time'] <= 3.+REST_WINDOW_S+REC_STOP_SETTLE_S+.8, f"stopped at {events[1]['time']}"

        # 7c. Wrong id on a provisional clip: both homes covered, the clip starts for J2 (seen last); the hand
        #     lifts: J2 still at home, J1's home empty -> the clip is handed to J1 and saved as J1_...
        hand = HandAtBox(); events = []
        rec = BoxRecorder(folder/'home_reassign', 10., events.append, hands=hand, box=box)
        j1, j2 = track(1, A), track(2, B); trk = tracker(j1, j2)
        two_at_home(rec, j1, j2)
        j1.state = 'OCCLUDED'
        for t in times(.5, .7):        # J1 covered first, J2 still seen
            db = mdet(B); see(j2, db, t); rec.update(trk, [db], scene(gold=[B], cover=[A]), t)
        j2.state, hand.on = 'OCCLUDED', True
        for t in times(.7, 1.5):
            rec.update(trk, [], scene(cover=[A, B]), t)
        assert states(events, 'J2') == ['REC_STARTED'] and events[0]['provisional'] is True, events
        for t in times(1.5, 3.):       # hand off the bench: J2 lying at home, J1's home empty
            db = mdet(B); see(j2, db, t); rec.update(trk, [db], scene(gold=[B]), t)
        assert [e['state'] for e in events] == ['REC_STARTED', 'REC_REASSIGNED'], events
        assert rec.rec[1].writer is not None and rec.rec[2].writer is None
        rec.stop_all('test end')
        saved = [e for e in events if e['state'] == 'REC_SAVED']
        assert saved[0]['jewel'] == 'J1' and saved[0]['clip'].startswith('J1_')
        assert (folder/'home_reassign'/saved[0]['clip']).exists()

        # 6. No box: nothing ever records.
        events = []
        rec = BoxRecorder(folder/'six', 10., events.append)
        j1 = track(1, in_box); trk = tracker(j1); j1.state = 'OCCLUDED'
        for t in times(0, 2.):
            rec.update(trk, [det(in_box)], frame, t)
        assert not events
    print('PASS: box recording trigger - lift alone ignored, debounced start with pre-roll, placed-back / left stops, '
          'flicker rejection, one clip per trip, visible carry, resting-in-box jewel not credited, pinch is the main '
          'trigger (open hand over the box ignored), gold only when no hand is seen, jumpy box still stops '
          'promptly, parallel clips, only the chosen hand (right) pinches, home position: jewel under the hand '
          'not recorded, wrong id handed to the jewel that left, still-at-home clip kept and given to the jewel '
          'placed away from its home (else saved UNASSIGNED_).')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', help='Camera index or video path; defaults to saved source')
    parser.add_argument('--fps', type=float, default=5., help='Maximum inference sampling rate and clip FPS (default 5)')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs'/'pickup_trigger_box')
    parser.add_argument('--box', type=lambda s: tuple(float(v) for v in s.split(',')),
                        help='Recording box x1,y1,x2,y2 in capture pixels (overrides rec_box.json)')
    parser.add_argument('--no-hand-fallback', action='store_true', help='Do not load the MediaPipe pinch fallback')
    parser.add_argument('--autostart', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--smoke-ui', action='store_true')
    parser.add_argument('--check-models', action='store_true')
    parser.add_argument('--list-cameras', action='store_true', help='List Linux capture devices without opening a stream')
    args = parser.parse_args()
    if args.box is not None and len(args.box) != 4:
        parser.error('--box needs four numbers: x1,y1,x2,y2')
    if not 1 <= args.fps <= 10:
        parser.error('--fps must be between 1 and 10')
    if args.self_test:
        self_test()
        return 0
    if args.list_cameras:
        devices = capture_devices()
        for index, name in devices:
            print(f'Camera {index}: {name} (/dev/video{index})')
        if not devices:
            print('No readable Linux video-capture devices found.')
        return 0
    original, QtCore, QtGui, QtWidgets = dependencies()
    if args.check_models:
        models = Models(original)
        frame = np.full((540, 960, 3), 220, np.uint8)
        roi = np.array([[0, 0], [959, 0], [959, 539], [0, 539]], np.int32)
        models.process(frame, roi, 0.)
        assert models.tracker.identity_slic and models.tracker.id_model is not None
        models.close_tracking()
        print('PASS: original Worker + SLIC identity and gold INT8 execute on CPU.')
        if not args.no_hand_fallback:
            hands = models.enable_keypoints().detect(frame)
            assert all(hand.points.shape == (21, 2) for hand in hands)
            models.close_keypoints()
            print('PASS: hand landmark model (pinch fallback) executes on CPU.')
        return 0
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    Window = window_class(original, QtCore, QtGui, QtWidgets)
    if args.smoke_ui:
        args.autostart = False
    window = Window(args); window.show()
    if args.smoke_ui:
        app.processEvents()
        window.draw_button.click()
        assert window.video.drawing
        window.draw_button.click()
        assert not window.video.drawing
        before = window.pinch_px.value()
        window.pinch_px.setValue(before+10)
        assert HAND_PINCH_PX == before+10, 'Pinch spin box must change the live trigger distance'
        window.pinch_px.setValue(0)
        assert HAND_PINCH_PX is None
        window.pinch_px.setValue(before)
        hand_before = window.pinch_hand.currentText()
        window.pinch_hand.setCurrentText('Any')
        assert PINCH_HAND == 'Any', 'Hand selector must change the live pinch hand'
        window.pinch_hand.setCurrentText(hand_before)
        window.close()
        print('PASS: box recording UI starts without capture; Draw box toggles.')
        return 0
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(main())
