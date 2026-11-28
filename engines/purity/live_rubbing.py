"""Original engines/purity/live_rubbing.py definitions.

Loaded into tracker globals by the explicit temporary compatibility binder.
The source bodies are unchanged; see docs/IMPORT_PATH_MIGRATION.md.
"""

def spot_patch(frame, box, pad=6):
    """Small grey patch of a box (with a little bench around it), for "is it still lying there?"."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = frame.shape[:2]
    crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
    if crop.size == 0:
        return None
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


class StoneWatcher(threading.Thread):
    """Predict stones with the INT8 OpenVINO model on each latest camera frame.

    A missed, expired or failed prediction clears the automatic box. No dark
    shape search, brightness/shape validation or retained occlusion box is used.
    """

    def __init__(self, grabber, model, roi_fn, min_interval=0.):
        super().__init__(daemon=True)
        self.grabber, self.model, self.roi_fn = grabber, model, roi_fn
        self.min_interval = min_interval
        self.bbox = None
        self.found_t = 0.0
        self.running = True

    @property
    def present(self):
        return self.bbox is not None and time.time() - self.found_t < STONE_PRESENT_S

    def run(self):
        last_ts, last_started = None, -1e9
        while self.running:
            if time.monotonic()-last_started < self.min_interval:
                time.sleep(.005)
                continue
            frame, ts = self.grabber.latest(copy_frame=False)
            if frame is None or time.time() - ts > STONE_RESULT_MAX_AGE_S:
                self.bbox, self.found_t = None, 0.0
                time.sleep(0.005)
                continue
            if ts == last_ts:
                time.sleep(0.005)  # wait only when the camera has no new frame
                continue
            last_ts, last_started = ts, time.monotonic()
            try:
                self.update(frame, captured_at=ts)
            except Exception as exc:
                self.bbox, self.found_t = None, 0.0
                write_log(f"STONE ERROR  OpenVINO prediction failed: {exc}")
            # Immediately take the newest frame; never queue frames or repeat one.

    def update(self, frame, captured_at=None):
        observed_at = time.time() if captured_at is None else captured_at
        roi = self.roi_fn()
        roi = roi.copy() if roi is not None else None
        box = self.model_box(frame, roi)
        current_roi = self.roi_fn()
        same_roi = ((roi is None and current_roi is None)
                    or (roi is not None and current_roi is not None
                        and np.array_equal(roi, current_roi)))
        if time.time() - observed_at > STONE_RESULT_MAX_AGE_S or not same_roi:
            box = None
        self.bbox = box
        self.found_t = observed_at if box is not None else 0.0

    def model_box(self, frame, roi=None):
        """Highest-confidence stone detection whose centre lies in the ROI."""
        if self.model is None:
            return None
        r = self.model.predict(frame, imgsz=640, conf=STONE_CONF, device=DEVICE, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        h, w = frame.shape[:2]
        best = None
        for b, confidence in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if confidence < STONE_CONF or not np.isfinite(b).all():
                continue
            x1, y1, x2, y2 = np.clip(b, (0, 0, 0, 0), (w, h, w, h)).astype(int)
            if x2 <= x1 or y2 <= y1:
                continue
            centre = (float(x1 + x2) / 2, float(y1 + y2) / 2)
            if roi is not None and cv2.pointPolygonTest(roi, centre, False) < 0:
                continue
            if best is None or confidence > best[0]:
                best = (float(confidence), (int(x1), int(y1), int(x2), int(y2)))
        return best[1] if best else None


def stone_observation_matches(before, after):
    """Accept normal detector jitter; a moved stone only invalidates this sample."""
    if before is None or after is None:
        return False
    a, b = np.asarray(before, float), np.asarray(after, float)
    diagonal = 0.5 * (np.linalg.norm(a[2:] - a[:2]) + np.linalg.norm(b[2:] - b[:2]))
    displacement = np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2)
    return (box_iou(before, after) >= RUB_STONE_MIN_IOU
            and displacement <= RUB_STONE_MAX_CENTER_RATIO * max(diagonal, 1.0))


def rubbing_evidence(polys, confs, frame_shape, stone):
    """Clip full-frame model masks to the current stone; never invoke a model."""
    h, w = frame_shape[:2]
    sx1, sy1, sx2, sy2 = np.clip(stone, (0, 0, 0, 0), (w, h, w, h)).astype(int)
    best = None
    for poly, confidence in zip(polys, confs):
        if poly is None or len(poly) < 3 or confidence < RUB_GOLD_CONF:
            continue
        px, py, pw, ph = cv2.boundingRect(poly.astype(np.int32))
        x1, y1, x2, y2 = max(0, px), max(0, py), min(w, px + pw), min(h, py + ph)
        ix1, iy1, ix2, iy2 = max(x1, sx1), max(y1, sy1), min(x2, sx2), min(y2, sy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        # Local mask allocation avoids a full-camera-sized buffer per detection.
        mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
        cv2.fillPoly(mask, [(poly - (x1, y1)).astype(np.int32)], 1)
        raw = int(mask.sum())
        if raw < RUB_MIN_MASK_PX:
            continue
        on_stone = np.zeros_like(mask)
        on_stone[iy1-y1:iy2-y1, ix1-x1:ix2-x1] = mask[iy1-y1:iy2-y1, ix1-x1:ix2-x1]
        on_stone = largest_component(on_stone)
        n = int(on_stone.sum())
        if n < RUB_MIN_MASK_PX or n / raw < RUB_MIN_INSIDE_RATIO:
            continue
        score = float(confidence) * (n / raw) * n
        if best is None or score > best[0]:
            best = (score, on_stone, (x1, y1))
    if best is None:
        return None
    _, mask, (ox, oy) = best
    m = cv2.moments(mask, binaryImage=True)
    ys, xs = np.nonzero(mask)
    return dict(c=np.array([ox + m['m10'] / m['m00'], oy + m['m01'] / m['m00']]),
                box=(ox + xs.min(), oy + ys.min(), ox + xs.max() + 1, oy + ys.max() + 1),
                mask=mask, origin=(ox, oy), stone=tuple(stone))


def projected_rub_geometry(tr, stone):
    """Place saved rubbing masks on the stone's current translation and scale."""
    marks = getattr(tr, 'rub_marks', None)
    if not marks or stone is None:
        return list(tr.rub_boxes), list(tr.rub_masks)
    current = np.asarray(stone, float)
    boxes, masks = [], []
    for mask, origin, recorded_stone in marks:
        previous = np.asarray(recorded_stone, float)
        scale = np.maximum(current[2:] - current[:2], 1.) / np.maximum(previous[2:] - previous[:2], 1.)
        origin = np.rint(current[:2] + (np.asarray(origin) - previous[:2]) * scale).astype(int)
        size = np.maximum(1, np.rint(np.asarray(mask.shape[::-1]) * scale).astype(int))
        if tuple(size) != mask.shape[::-1]:
            mask = cv2.resize(mask, tuple(size), interpolation=cv2.INTER_NEAREST)
        x, y, w, h = cv2.boundingRect(mask)
        if w and h:
            ox, oy = map(int, origin)
            boxes.append((ox+x, oy+y, ox+x+w, oy+y+h))
            masks.append((mask, (ox, oy)))
    return boxes, masks


class RubbingObserver:
    """Consume the same full-frame observation as tracking, on the worker thread.

    Motion and box jitter do not reset history. Invalid/missed observations
    cannot contribute credit. Stone disappearance and explicit Reset clear it.
    """
    def __init__(self, stone_box, generation_fn=None):
        self.stone_box = stone_box
        self.generation_fn = generation_fn or (lambda: 0)
        self.active = False
        self.reset()

    def snapshot(self):
        return self.ev, self.rubbing, self.ts

    def reset(self):
        self.moves = deque(maxlen=10)
        self.prev = None
        self.motion_stone = self.reference_size = None
        self.ev, self.rubbing, self.ts = None, False, 0.0

    def suspend(self, ts):
        # Preserve progress, but do not bridge an unknown movement with a jump.
        self.prev = None
        self.ev, self.rubbing, self.ts = None, False, ts

    def observe(self, polys, confs, frame_shape, ts, captured_stone, revision, source_frame=None):
        if revision != self.generation_fn():
            return  # explicit Reset already owns the new generation
        stone = self.stone_box()
        if stone is None:
            self.reset()
            return
        if not 0 <= time.time() - ts <= LIVE_RESULT_MAX_AGE_S:
            self.suspend(ts)
            return
        if not stone_observation_matches(captured_stone, stone):
            self.suspend(ts)
            return
        ev = rubbing_evidence(polys, confs, frame_shape, stone)
        self.judge(ev, ts)
        if self.ev is not None:
            self.ev = dict(self.ev, timestamp=ts, frame=source_frame)

    def judge(self, ev, ts):
        self.ev, self.rubbing, self.ts = ev, False, ts
        if ev is None:
            self.prev = None
            return  # no gold mask: preserve history, never display an old mask
        stone = np.asarray(ev['stone'], float)
        if self.motion_stone is not None:
            previous = self.motion_stone
            displacement = np.linalg.norm((stone[:2] + stone[2:] - previous[:2] - previous[2:]) / 2)
            if box_iou(stone, previous) >= .95 and displacement <= RUB_STONE_JITTER_PX:
                stone = previous
        self.motion_stone = stone
        size = np.maximum(stone[2:] - stone[:2], 1.)
        if self.reference_size is None:
            self.reference_size = size.copy()
        # Translation and scale of the stone do not count as jewel movement.
        point = (ev['c'] - stone[:2]) / size * self.reference_size
        previous, self.prev = self.prev, (point, ts)
        if not self.active or previous is None or not 0 < ts - previous[1] < RUB_GAP_S:
            return
        move = float(np.linalg.norm(point - previous[0]))
        self.moves.append((ts, move))
        recent = [distance for when, distance in self.moves if 0 <= ts - when < RUB_GAP_S][-RUB_MOVE_WINDOW:]
        self.rubbing = (len(recent) == RUB_MOVE_WINDOW and max(recent) >= 1.2 * RUB_MOVE_PX
                        and sum(recent) / RUB_MOVE_WINDOW >= .6 * RUB_MOVE_PX)

