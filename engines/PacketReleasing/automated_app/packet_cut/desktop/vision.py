"""Vision module."""

from ._imports import *

from .config import *

def blur_score(frame):
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def class_names_dict(model):
    names = model.names
    if isinstance(names, dict):
        return names
    return {i: name for i, name in enumerate(names)}

def detect_packet_number(frame, number_model, device):
    """Detects packet number ROI box and crops number region."""
    h, w = frame.shape[:2]
    result = number_model.predict(
        source=frame,
        imgsz=NUMBER_IMGSZ,
        conf=NUMBER_CONF,
        device=device,
        verbose=False
    )[0]

    boxes = None
    if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        boxes = result.obb.xyxy.cpu().numpy()
    elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()

    if boxes is None or len(boxes) == 0:
        return None, None, result

    box = boxes[0].astype(int)
    x1 = max(0, min(int(box[0]), w - 1))
    y1 = max(0, min(int(box[1]), h - 1))
    x2 = max(0, min(int(box[2]), w))
    y2 = max(0, min(int(box[3]), h))

    if x2 <= x1 or y2 <= y1:
        return None, None, result

    pad_x = int((x2 - x1) * 0.08)
    pad_y = int((y2 - y1) * 0.15)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    crop = frame[cy1:cy2, cx1:cx2].copy()
    return crop, (x1, y1, x2, y2), result

def detection_present(result, selected_ids=None):
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return False
    if selected_ids is None:
        return True
    if result.boxes.cls is None:
        return False
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    return any(int(c) in selected_ids for c in classes)

def draw_mask(frame, mask, color, alpha=0.25):
    if mask is None or not np.any(mask):
        return frame
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def get_union_mask(result, frame_shape, selected_ids=None):
    h, w = frame_shape[:2]
    union = np.zeros((h, w), dtype=np.uint8)
    if result is None or result.masks is None or result.masks.data is None:
        return union

    mask_data = result.masks.data.detach().cpu().numpy()
    boxes = result.boxes if result.boxes is not None else None
    classes = boxes.cls.detach().cpu().numpy().astype(int) if (boxes is not None and boxes.cls is not None) else None
    has_polys = hasattr(result.masks, "xy") and result.masks.xy is not None

    for i in range(len(mask_data)):
        if selected_ids is not None and classes is not None:
            if i < len(classes) and classes[i] not in selected_ids:
                continue

        if has_polys and i < len(result.masks.xy) and len(result.masks.xy[i]) >= 3:
            poly = result.masks.xy[i].astype(np.int32)
            cv2.fillPoly(union, [poly], 255)
        else:
            m_unpadded = unpad_letterbox_mask(mask_data[i], (h, w))
            union[m_unpadded >= 0.5] = 255

    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(union, cv2.MORPH_OPEN, kernel)

def matching_class_ids(model, keywords):
    found = []
    for class_id, name in class_names_dict(model).items():
        normalized = re.sub(r"[^a-z0-9]", "", str(name).lower())
        if any(k in normalized for k in keywords):
            found.append(int(class_id))
    return found or None

def predict(model, frame, confidence, imgsz=IMGSZ, device="cpu"):
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
        raise FileNotFoundError(f"{description} not found: {path.resolve()}")
    return path
