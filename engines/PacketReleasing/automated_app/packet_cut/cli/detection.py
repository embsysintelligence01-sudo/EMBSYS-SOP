"""Detection module."""

from ._imports import *

from .config import *

def blur_score(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

def class_names_dict(model):

    names = model.names

    if isinstance(names, dict):
        return names

    return {
        i: name
        for i, name in enumerate(names)
    }

def detection_present(result, selected_ids=None):

    if result.boxes is None:
        return False

    if len(result.boxes) == 0:
        return False

    if selected_ids is None:
        return True

    if result.boxes.cls is None:
        return False

    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    return any(
        int(c) in selected_ids
        for c in classes
    )

def evaluate_hand_detection(
    hand_result,
    hand_ids,
    frame_shape,
    packet_mask=None,
    topstrip_mask=None,
    reference_mask=None
):
    """
    Evaluates raw hand predictions and filters out false positives caused by red seals.
    Prints diagnostic logs and returns (is_hand_present, valid_hand_mask, diagnostics_list).
    """
    h, w = frame_shape
    diagnostics = []

    if hand_result.boxes is None or len(hand_result.boxes) == 0:
        return False, np.zeros((h, w), dtype=np.uint8), diagnostics

    # Base reference packet mask: use current packet_mask, fallback to stored reference_mask
    ref_pkt = None
    if packet_mask is not None and np.any(packet_mask):
        ref_pkt = packet_mask
    elif reference_mask is not None and np.any(reference_mask):
        ref_pkt = reference_mask

    boxes = hand_result.boxes.xyxy.cpu().numpy()
    confidences = hand_result.boxes.conf.cpu().numpy()
    classes = hand_result.boxes.cls.cpu().numpy().astype(int)

    has_masks = (
        hand_result.masks is not None
        and hand_result.masks.data is not None
        and len(hand_result.masks.data) == len(boxes)
    )

    valid_mask = np.zeros((h, w), dtype=np.uint8)
    any_valid_hand = False

    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, classes)):
        if cls_id not in hand_ids:
            continue

        x1, y1, x2, y2 = box.astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        box_area = max(0, (x2 - x1) * (y2 - y1))

        # Extract individual mask
        inst_mask = np.zeros((h, w), dtype=np.uint8)
        if has_masks:
            m_raw = hand_result.masks.data[i].detach().cpu().numpy()
            m_resized = cv2.resize(
                m_raw.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )
            inst_mask[m_resized >= 0.5] = 255
        else:
            inst_mask[y1:y2, x1:x2] = 255

        mask_area = int(np.count_nonzero(inst_mask))

        # Analyze position relative to packet / seal
        in_packet_ratio = 0.0
        in_topstrip_ratio = 0.0
        pos_rel = "OUTSIDE_PACKET"

        if mask_area > 0:
            if ref_pkt is not None and np.any(ref_pkt):
                overlap_pkt = np.count_nonzero((inst_mask > 0) & (ref_pkt > 0))
                in_packet_ratio = overlap_pkt / mask_area
            if topstrip_mask is not None and np.any(topstrip_mask):
                overlap_ts = np.count_nonzero((inst_mask > 0) & (topstrip_mask > 0))
                in_topstrip_ratio = overlap_ts / mask_area

        if in_packet_ratio > 0.85:
            pos_rel = "INSIDE_PACKET"
        elif in_packet_ratio > 0.20:
            pos_rel = "CROSSING_BORDER"
        else:
            pos_rel = "OUTSIDE_PACKET"

        # Check touching image boundaries (a real human hand enters from outside the camera frame)
        touches_border = (x1 <= 5 or y1 <= 5 or x2 >= w - 5 or y2 >= h - 5)

        # False-positive seal suppression condition:
        # 1. High overlap with TopStrip mask (> 50%)
        # OR
        # 2. Entirely enclosed inside packet (> 85%), not touching camera boundary, and small/moderate area
        is_seal_fp = False
        reason = "REAL_HAND"

        if in_topstrip_ratio > 0.50:
            is_seal_fp = True
            reason = f"SEAL_FP (topstrip_overlap={in_topstrip_ratio:.1%})"
        elif in_packet_ratio > 0.85 and not touches_border:
            # Fully inside packet, detached from image borders -> characteristic of printed seal or sticker
            is_seal_fp = True
            reason = f"SEAL_FP (enclosed_in_packet={in_packet_ratio:.1%}, detached_from_edge)"

        diag = {
            "index": i + 1,
            "conf": conf,
            "bbox": (x1, y1, x2, y2),
            "area_px": mask_area,
            "pos_rel": pos_rel,
            "in_pkt_pct": in_packet_ratio * 100,
            "is_seal_fp": is_seal_fp,
            "reason": reason
        }
        diagnostics.append(diag)

        print(
            f"[HAND DIAGNOSTIC] Pred {i+1}: conf={conf:.3f}, bbox={(x1, y1, x2, y2)}, "
            f"area={mask_area}px, pos={pos_rel} (pkt_overlap={in_packet_ratio:.1%}), "
            f"touches_border={touches_border} -> {reason}"
        )

        if not is_seal_fp:
            any_valid_hand = True
            valid_mask[inst_mask > 0] = 255

    return any_valid_hand, valid_mask, diagnostics

def find_number_model():

    matches = sorted(
        glob.glob(PACKET_NUMBER_MODEL_PATTERN)
    )

    matches = [
        Path(p) for p in matches
        if Path(p).is_file()
    ]

    if not matches:
        raise FileNotFoundError(
            "Packet-number detector not found.\n"
            f"Expected: {PACKET_NUMBER_MODEL_PATTERN}"
        )

    if len(matches) > 1:

        print("Multiple packet number models found:")

        for item in matches:
            print(item)

        print("Using:", matches[0])

    return matches[0]

def get_union_mask(
    result,
    frame_shape,
    selected_ids=None
):

    h, w = frame_shape

    union = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    if result.masks is None:
        return union

    if result.masks.data is None:
        return union

    mask_data = (
        result.masks.data
        .detach()
        .cpu()
        .numpy()
    )

    classes = None

    if result.boxes is not None and result.boxes.cls is not None:

        classes = (
            result.boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

    for i, mask in enumerate(mask_data):

        if selected_ids is not None:

            if classes is None:
                continue

            if i >= len(classes):
                continue

            if classes[i] not in selected_ids:
                continue

        has_polys = hasattr(result.masks, "xy") and result.masks.xy is not None
        if has_polys and i < len(result.masks.xy) and len(result.masks.xy[i]) >= 3:
            poly = result.masks.xy[i].astype(np.int32)
            cv2.fillPoly(union, [poly], 255)
        else:
            m_unpadded = unpad_letterbox_mask(mask, (h, w))
            union[m_unpadded >= 0.5] = 255

    # Remove isolated noise.
    # Do not use morphological closing because it can bridge a real cut.
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    return cv2.morphologyEx(
        union,
        cv2.MORPH_OPEN,
        kernel
    )

def matching_class_ids(model, keywords):

    found = []

    for class_id, name in class_names_dict(model).items():

        normalized = re.sub(
            r"[^a-z0-9]",
            "",
            str(name).lower()
        )

        if any(k in normalized for k in keywords):
            found.append(int(class_id))

    return found or None

def predict(
    model,
    frame,
    confidence,
    imgsz=IMGSZ,
    device="cpu"
):

    return model.predict(
        source=frame,
        imgsz=imgsz,
        conf=confidence,
        device=device,
        verbose=False
    )[0]

def require_file(path, description):

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path.resolve()}"
        )

    return path
