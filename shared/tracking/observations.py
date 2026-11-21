"""Original shared/tracking/observations.py definitions.

Loaded into tracker globals by the explicit temporary compatibility binder.
The source bodies are unchanged; see docs/IMPORT_PATH_MIGRATION.md.
"""

class Det:
    __slots__ = ("box", "c", "area", "conf", "gold", "mask", "origin", "app", "stable_since", "idres", "sig")

    def __init__(self, box, c, area, conf, gold=1.0, mask=None, origin=(0, 0)):
        self.box, self.c, self.area, self.conf, self.gold = box, c, area, conf, gold
        self.mask, self.origin = mask, origin  # mask is local to its box; origin = its top-left in the frame
        self.app = None  # appearance fingerprint, filled in when needed
        self.stable_since = None  # standing still since (set by the worker)
        self.idres = None         # (jewel id or None, probability) from the ID model, once it is stable
        self.sig = None           # version 6 SLIC scores for contested detections


def touching(a, b, gap):
    """True if the masks of a and b come within gap pixels of each other."""
    # only pixels inside both boxes grown by gap can be that close, so work on that region alone
    x1 = max(a.origin[0], b.origin[0]) - gap
    y1 = max(a.origin[1], b.origin[1]) - gap
    x2 = min(a.origin[0] + a.mask.shape[1], b.origin[0] + b.mask.shape[1]) + gap
    y2 = min(a.origin[1] + a.mask.shape[0], b.origin[1] + b.mask.shape[0]) + gap
    if x2 <= x1 or y2 <= y1:
        return False

    def paste(d):
        canvas = np.zeros((y2 - y1, x2 - x1), np.uint8)
        ox, oy = d.origin
        sx1, sy1 = max(x1, ox), max(y1, oy)
        sx2, sy2 = min(x2, ox + d.mask.shape[1]), min(y2, oy + d.mask.shape[0])
        if sx2 > sx1 and sy2 > sy1:
            canvas[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = d.mask[sy1 - oy:sy2 - oy, sx1 - ox:sx2 - ox]
        return canvas

    ca, cb = paste(a), paste(b)
    if not ca.any() or not cb.any():
        return False
    dist_to_a = cv2.distanceTransform(cv2.bitwise_not(ca), cv2.DIST_L2, 5)
    return bool(dist_to_a[cb > 0].min() <= gap)


def merge_dets(group):
    """One detection covering all of group (pieces of the same jewel)."""
    x1 = min(d.origin[0] for d in group)
    y1 = min(d.origin[1] for d in group)
    x2 = max(d.origin[0] + d.mask.shape[1] for d in group)
    y2 = max(d.origin[1] + d.mask.shape[0] for d in group)
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    for d in group:
        ox, oy = d.origin[0] - x1, d.origin[1] - y1
        mask[oy:oy + d.mask.shape[0], ox:ox + d.mask.shape[1]] |= d.mask
    m = cv2.moments(mask, binaryImage=True)
    area = max(m["m00"], 1.0)
    c = np.array([x1 + m["m10"] / area, y1 + m["m01"] / area])
    box = np.array([min(d.box[0] for d in group), min(d.box[1] for d in group),
                    max(d.box[2] for d in group), max(d.box[3] for d in group)])
    gold = sum(d.gold * d.area for d in group) / max(sum(d.area for d in group), 1.0)
    return Det(box, c, float(area), max(d.conf for d in group), gold, mask, (x1, y1))


def merge_pieces(dets):
    """Join detections whose masks touch (or overlap / sit inside each other) into one.
    A spread-out chain often comes back as a big detection plus a separate piece; the model's
    own IoU-based NMS keeps both because they barely overlap."""
    if len(dets) < 2 or any(d.mask is None for d in dets):
        return dets
    group_of = list(range(len(dets)))

    def root(i):
        while group_of[i] != i:
            i = group_of[i]
        return i

    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if root(i) != root(j) and touching(dets[i], dets[j], MERGE_GAP_PX):
                group_of[root(j)] = root(i)
    groups = {}
    for i, d in enumerate(dets):
        groups.setdefault(root(i), []).append(d)
    return [g[0] if len(g) == 1 else merge_dets(g) for g in groups.values()]


def gold_fraction(hsv, mask):
    """Share of gold-coloured pixels among the non-background pixels under mask (None if too few)."""
    h, s, v = (c.astype(np.int16) for c in hsv[mask > 0].T)
    background = ((s < BG_WHITE_S_MAX) & (v > BG_WHITE_V_MIN)) | (v < BG_DARK_V_MAX)
    n = int((~background).sum())
    if n < GOLD_MIN_PX:
        return None
    skin = (h <= SKIN_H_MAX) & (s >= SKIN_S_MIN) & ~background
    if float(skin.sum()) / n > SKIN_MAX_FRAC:
        return 0.0  # the operator's fingers, not gold
    gold = (h >= GOLD_H[0]) & (h <= GOLD_H[1]) & (s >= GOLD_S_MIN) & (v >= GOLD_V_MIN) & ~background
    return float(gold.sum()) / n


def otsu_jewel_mask(img, mask):
    """Jewel pixels by Otsu on saturation (gold is saturated; the white bench and the dark stone are
    not), kept to the detection's mask."""
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    _, otsu = cv2.threshold(cv2.GaussianBlur(sat, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    otsu[mask == 0] = 0
    return otsu


def count_link_holes(img, mask):
    """Holes inside the jewel's Otsu-on-saturation mask: a chain's links leave many however it is
    piled; bangles, rings and pendants almost none."""
    otsu = otsu_jewel_mask(img, mask)
    contours, hierarchy = cv2.findContours(otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return sum(1 for c, h in zip(contours, hierarchy[0]) if h[3] >= 0 and cv2.contourArea(c) >= CHAIN_HOLE_MIN_PX)


def appearance(frame, det, siglip=None, rotations=False):
    """Fingerprint of one detection: chain or solid, jewel pixels, LAB colour histogram and optionally
    SigLIP2 embedding(s). None when no jewel pixels are visible."""
    ox, oy = (int(v) for v in det.origin)
    if det.mask is not None:
        h, w = det.mask.shape
        img, mask = frame[oy:oy + h, ox:ox + w], det.mask.copy()
    else:
        x1, y1, x2, y2 = (int(v) for v in det.box)
        img = frame[max(y1, 0):y2, max(x1, 0):x2]
        mask = np.full(img.shape[:2], 255, np.uint8)
    if img.size == 0 or img.shape[:2] != mask.shape:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bench = (hsv[:, :, 1] < BG_WHITE_S_MAX) & (hsv[:, :, 2] > BG_WHITE_V_MIN)
    jewel = ((mask > 0) & ~bench & (img.max(axis=2) > 0)).astype(np.uint8) * 255
    pixels = int(np.count_nonzero(jewel))
    if pixels < GOLD_MIN_PX:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    colour = cv2.calcHist([lab], [0, 1, 2], jewel, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    colour /= max(float(colour.sum()), 1.0)
    app = dict(pixels=pixels, colour=colour, chain=count_link_holes(img, mask) >= CHAIN_MIN_HOLES, emb=None)
    if siglip is not None:
        app["emb"] = siglip_embed(siglip, img, siglip_mask(img, mask, jewel), rotations)
    return app


def siglip_mask(img, mask, fallback):
    """What SigLIP2 sees: the Otsu jewel mask (on real pieces it separated jewels as well as the plain
    crop and slightly better than the seg mask), or the seg mask if Otsu finds nothing."""
    otsu = otsu_jewel_mask(img, mask)
    return otsu if np.count_nonzero(otsu) >= GOLD_MIN_PX else fallback


def siglip_embed(siglip, img, jewel, rotations):
    ys, xs = np.nonzero(jewel)
    m = int(0.1 * max(np.ptp(xs), np.ptp(ys)))
    y1, y2, x1, x2 = max(ys.min() - m, 0), ys.max() + m + 1, max(xs.min() - m, 0), xs.max() + m + 1
    crop = img[y1:y2, x1:x2].copy()
    crop[jewel[y1:y2, x1:x2] == 0] = 127  # grey background, as in the backend
    return siglip.embed_rotations(crop) if rotations else siglip.embed(crop)[None]


def similarity(a, b):
    """0..1 fingerprint similarity (geometric mean of area ratio, colour and SigLIP2 when both have it);
    0 for a chain vs a solid piece."""
    if a["chain"] != b["chain"]:
        return 0.0
    parts = [min(a["pixels"], b["pixels"]) / max(a["pixels"], b["pixels"]),
             float(np.sum(np.sqrt(a["colour"] * b["colour"])))]
    if a["emb"] is not None and b["emb"] is not None:
        parts.append(float(np.clip(np.max(a["emb"] @ b["emb"].T), 0.0, 1.0)))
    return float(np.prod(parts) ** (1 / len(parts)))


def colour_similarity(a, b):
    return float(np.sum(np.sqrt(a["colour"] * b["colour"])))


class Track:
    def __init__(self, tid, det, t):
        self.id = tid
        self.pos = det.c.astype(float)
        self.vel = np.zeros(2)
        self.box = det.box
        self.area = det.area
        self.last_seen = t
        self.state = "OK"
        self.out_frames = 0
        self.rub_time = 0.0       # seconds of rubbing credited to this jewel
        self.rub_last = 0.0       # last time rubbing was credited to it
        self.rubbed = False
        self.settle_since = None  # rubbed and lying still off the stone since
        self.done = False         # rubbed (and acid tested) and placed back
        self.rub_boxes = deque(maxlen=30)  # where its rub streak was drawn on the stone
        self.rub_masks = deque(maxlen=30)  # observed gold (mask, origin) during rubbing
        self.rub_marks = deque(maxlen=30)  # original masks and stone poses, for moving-stone projection
        self.rub_img = None       # fixed, unannotated photo at rubbing confirmation
        self.new_rub_target = None
        self.acid_roi_mask = self.acid_roi_stone = None
        self.acid_done = False    # acid test seen on its rub streak
        self.acid_box = None
        self.acid_result = None
        self.acid_crop = None
        self.app = det.app        # appearance fingerprint taken at lock (never updated)
        self.det_seen = t         # last time the detector itself found it
        self.det_box = det.box    # its box at that moment
        self.bridged = False      # (no CSRT any more in v4: always False)
        self.kind, self.kind_conf = "", 0.0  # jewel type from the classifier, at lock
        self.kind_note = ""
        self.move = 0.0           # movement score (smoothed centre jitter); the rubbed jewel moves most
        self.det_mask, self.det_origin = det.mask, det.origin
        self.spot, self.removed_since = "jewel", None  # out of sight: "jewel" / "removed" / "hand"

    def predicted(self, t):
        # extrapolate only briefly; after that hold the last known position
        return self.pos + self.vel * min(t - self.last_seen, 0.3)

    def correct(self, det, t):
        dt = t - self.last_seen
        if dt < 0.5:
            v = (det.c - self.pos) / max(dt, 1e-3)
            self.vel = 0.7 * self.vel + 0.3 * v
            self.pos = 0.4 * self.pos + 0.6 * det.c
        else:  # re-acquired after a gap: jump to it, forget velocity
            self.vel[:] = 0
            self.pos = det.c.astype(float)
        self.box = det.box
        self.area = 0.9 * self.area + 0.1 * det.area
        self.last_seen = t
        self.det_seen, self.det_box = t, det.box
        self.det_mask, self.det_origin = det.mask, det.origin  # its own pixels at that moment


def box_iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = float(ix) * float(iy)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def dashed_rect(img, p1, p2, col, dash=14):
    """Dashed rectangle: a jewel that is not where its box is drawn (out of sight)."""
    (x1, y1), (x2, y2) = p1, p2
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1])))
        for s0 in range(0, length, 2 * dash):
            s1 = min(length, s0 + dash)
            pa = (int(a[0] + (b[0] - a[0]) * s0 / length), int(a[1] + (b[1] - a[1]) * s0 / length))
            pb = (int(a[0] + (b[0] - a[0]) * s1 / length), int(a[1] + (b[1] - a[1]) * s1 / length))
            cv2.line(img, pa, pb, col, 2)


def crop_box(frame, box, pad=ID_CROP_PAD):
    """The detection box plus a little bench around it, shrunk to at most ID_CROP_MAX px."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = frame.shape[:2]
    px, py = int((x2 - x1) * pad) + 4, int((y2 - y1) * pad) + 4
    c = frame[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
    if c.size and max(c.shape[:2]) > ID_CROP_MAX:
        f = ID_CROP_MAX / max(c.shape[:2])
        c = cv2.resize(c, (max(1, int(c.shape[1] * f)), max(1, int(c.shape[0] * f))), interpolation=cv2.INTER_AREA)
    return c


def augment(crop, rng):
    """A capture crop as it may look later: turned any way, mirrored, a bit bigger/smaller, lit differently."""
    h, w = crop.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(0, 360)), float(rng.uniform(0.85, 1.15)))
    m[:, 2] += rng.uniform(-0.06, 0.06, 2) * (w, h)
    out = cv2.warpAffine(crop, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-18, 18), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (5, 5), 0)
    return out

