"""Ocr module."""

from ._imports import *

from .config import *

from .dashboard import create_dashboard, show

_OCR_ENGINE = None

def detect_packet_number(frame, number_model, device):
    """
    Infers packet number using the dedicated YOLO model.
    Note: The trained ONNX model uses task='obb' (Oriented Bounding Boxes).
    We check both result.obb and result.boxes to support either format.
    """
    print("\n[PACKET OCR] Inference started")
    print(f"[PACKET OCR] Input frame dimensions: {frame.shape}")

    result = number_model.predict(
        source=frame,
        imgsz=NUMBER_IMGSZ,
        conf=NUMBER_CONF,
        device=device,
        verbose=False
    )[0]

    # Support both OBB (Oriented Bounding Boxes) and standard Boxes
    boxes = None
    confidences = None
    classes = None

    if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        boxes = result.obb.xyxy.cpu().numpy()
        confidences = result.obb.conf.cpu().numpy()
        classes = result.obb.cls.cpu().numpy().astype(int)
    elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

    print(f"[PACKET OCR] Inference completed")
    print(f"[PACKET OCR] Number of detections: {len(boxes) if boxes is not None else 0}")

    if boxes is None or len(boxes) == 0:
        return None, None, result

    for i, (box, confidence, class_id) in enumerate(zip(boxes, confidences, classes)):
        print(
            f"[PACKET OCR] Detection {i + 1}: "
            f"class={class_id}, "
            f"confidence={confidence:.4f}, "
            f"box={box.tolist()}"
        )

    best = int(np.argmax(confidences))
    x1, y1, x2, y2 = boxes[best].astype(int)
    h, w = frame.shape[:2]

    # Clamp coordinates to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        print("[PACKET OCR] Invalid bounding box coordinates:", x1, y1, x2, y2)
        return None, None, result

    crop = frame[y1:y2, x1:x2].copy()

    print(f"[PACKET OCR] Detection confidence: {confidences[best]:.4f}")
    print(f"[PACKET OCR] Bounding box coordinates: {(x1, y1, x2, y2)}")
    print(f"[PACKET OCR] Crop dimensions: {crop.shape}")

    return crop, (x1, y1, x2, y2), result

def front_side_and_number(
    cap,
    number_model,
    device
):
    print("\n[PACKET OCR] Stage entered")
    print(f"[PACKET OCR] Model loaded: {number_model is not None}")
    if hasattr(number_model, "ckpt_path"):
        print(f"[PACKET OCR] Model path: {number_model.ckpt_path}")

    # Track live detection state
    frame_count = 0
    cached_bbox = None
    cached_conf = 0.0
    cached_crop = None
    status = "YOLO MODEL LOADED"

    # Step 1: Live placement with live packet-number detection every 3 frames
    deadline = time.time() + FRONT_NUMBER_TIMEOUT

    while time.time() < deadline:
        ok, live_frame = cap.read()
        if not ok:
            continue

        frame_count += 1

        # Run detection every 3 frames to keep CPU responsive
        if frame_count % 3 == 0:
            status = "DETECTING NUMBER REGION"
            crop_res, bbox_res, yolo_res = detect_packet_number(
                live_frame,
                number_model,
                device
            )
            if bbox_res is not None:
                cached_crop = crop_res
                cached_bbox = bbox_res
                # Extract best conf
                if hasattr(yolo_res, "obb") and yolo_res.obb is not None and len(yolo_res.obb) > 0:
                    cached_conf = float(yolo_res.obb.conf.cpu().numpy()[int(np.argmax(yolo_res.obb.conf.cpu().numpy()))])
                elif hasattr(yolo_res, "boxes") and yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
                    cached_conf = float(yolo_res.boxes.conf.cpu().numpy()[int(np.argmax(yolo_res.boxes.conf.cpu().numpy()))])
                status = "NUMBER REGION DETECTED"
            else:
                cached_crop = None
                cached_bbox = None
                cached_conf = 0.0
                status = "DETECTING NUMBER REGION"

        # Draw live bounding box overlay
        display_frame = live_frame.copy()
        if cached_bbox is not None:
            bx1, by1, bx2, by2 = cached_bbox
            cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"packet-ocr: {cached_conf:.2f}",
                (bx1, max(20, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        extra_info = [
            ("YOLO Model", "LOADED"),
            ("Confidence", f"{cached_conf:.2f}" if cached_conf > 0 else "N/A"),
            ("Bounding Box", "FOUND" if cached_bbox is not None else "SEARCHING")
        ]

        dashboard = create_dashboard(
            display_frame,
            "FRONT SIDE",
            status,
            [
                "Align packet FRONT side",
                "Press F when number is boxed",
                "Q to quit"
            ],
            crop=cached_crop,
            extra_info=extra_info
        )
        key = show(dashboard, [])

        if key in (ord("q"), 27):
            return None

        if key == ord("f"):
            break

    else:
        print("[PACKET OCR] Timed out waiting for front-side placement.")
        return None

    # Step 2: Fresh capture & detection for stabilization
    for remaining in range(2, 0, -1):
        ok, live_frame = cap.read()
        if not ok:
            continue
        dashboard = create_dashboard(
            live_frame,
            "FRONT SIDE",
            "STABILIZING",
            [
                f"Keep packet still... {remaining}",
                "Preparing high-resolution capture"
            ],
            crop=cached_crop,
            extra_info=[("YOLO Model", "LOADED"), ("Status", "STABILIZING")]
        )
        show(dashboard, [])
        time.sleep(0.4)

    # Capture a fresh frame
    ok, frame = cap.read()
    if not ok:
        print("[PACKET OCR] Failed to capture fresh frame.")
        return None

    crop, bbox, result = detect_packet_number(
        frame,
        number_model,
        device
    )

    if crop is None:
        print("[PACKET OCR] Number region not detected on fresh capture.")
        while True:
            ok, live_frame = cap.read()
            if not ok:
                continue

            dashboard = create_dashboard(
                live_frame,
                "FRONT SIDE",
                "DETECTING NUMBER REGION",
                [
                    "Packet number region not found",
                    "Adjust packet position",
                    "Press F to retry detection",
                    "Q to quit"
                ],
                extra_info=[("YOLO Model", "LOADED"), ("Bounding Box", "NOT FOUND")]
            )
            key = show(dashboard, [])

            if key in (ord("q"), 27):
                return None

            if key == ord("f"):
                return front_side_and_number(
                    cap,
                    number_model,
                    device
                )

    # Step 3: PaddleOCR execution
    x1, y1, x2, y2 = bbox
    preview = frame.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)

    status = "RUNNING PADDLEOCR"
    # Show status before running OCR inference so user sees the transition
    interim_dashboard = create_dashboard(
        preview,
        "PACKET NUMBER OCR",
        status,
        ["Running PaddleOCR recognition..."],
        crop=crop,
        extra_info=[
            ("YOLO Model", "LOADED"),
            ("Bounding Box", f"{x2 - x1}x{y2 - y1} px"),
            ("OCR State", "PROCESSING")
        ]
    )
    show(interim_dashboard, [])

    packet_number, ocr_score, candidates = run_ocr(crop)

    if not packet_number:
        status = "OCR TEXT UNREADABLE"
        packet_number = "UNREADABLE"
        ocr_conf_str = "0.00"
    else:
        status = "OCR CONFIRMATION REQUIRED"
        ocr_conf_str = f"{ocr_score:.2f}"

    print(f"[PACKET OCR] Final packet number: {packet_number} (confidence: {ocr_conf_str})")

    # Step 4: Operator confirmation loop
    while True:
        extra_info = [
            ("YOLO Model", "LOADED"),
            ("Bounding Box", f"{x2 - x1}x{y2 - y1} px"),
            ("OCR Text", packet_number),
            ("OCR Conf", ocr_conf_str),
            ("Confirmed", "NO")
        ]

        dashboard = create_dashboard(
            preview,
            "PACKET NUMBER OCR",
            status,
            [
                f"Detected: {packet_number}",
                "Y = Accept & proceed",
                "R = Retry detection",
                "Q = Quit"
            ],
            crop=crop,
            extra_info=extra_info
        )
        key = show(dashboard, [])

        if key in (ord("y"), ord("Y")):
            if packet_number == "UNREADABLE":
                print("[PACKET OCR] Cannot accept unreadable text. Please retry with 'R'.")
                continue
            status = "PACKET NUMBER CONFIRMED"
            confirm_dashboard = create_dashboard(
                preview,
                "PACKET NUMBER OCR",
                status,
                [f"Confirmed: {packet_number}", "Proceeding to next step..."],
                crop=crop,
                extra_info=[
                    ("YOLO Model", "LOADED"),
                    ("OCR Text", packet_number),
                    ("Confirmed", "YES")
                ]
            )
            show(confirm_dashboard, [])
            cv2.waitKey(600)
            print(f"[PACKET OCR] PACKET NUMBER CONFIRMED: {packet_number}")
            return packet_number

        if key in (ord("r"), ord("R")):
            return front_side_and_number(
                cap,
                number_model,
                device
            )

        if key in (ord("q"), 27):
            return None

def get_ocr_engine():
    """
    Initializes and returns a single reusable PaddleOCR instance.
    """
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        print("[PACKET OCR] Initializing PaddleOCR engine...")
        _OCR_ENGINE = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
        print("[PACKET OCR] PaddleOCR engine ready.")
    return _OCR_ENGINE

def preprocess_crop_for_ocr(crop, scale_factor=3.0):
    """
    Preprocesses the number crop using grayscale, CLAHE, 2x/3x enlargement and sharpening.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    # 3. Enlargement / Scaling
    ch, cw = enhanced.shape[:2]
    resized = cv2.resize(
        enhanced,
        (int(cw * scale_factor), int(ch * scale_factor)),
        interpolation=cv2.INTER_CUBIC
    )
    # 4. Sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(resized, -1, kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

def run_ocr(crop):
    """
    Runs PaddleOCR 3.7.0 on the cropped number region.
    Tries preprocessed and original crops, extracts text and confidences,
    and prioritizes 6-digit numeric candidates.
    """
    print("[PACKET OCR] PaddleOCR started")
    ocr_engine = get_ocr_engine()

    def parse_paddle_results(predict_results):
        candidates = []
        if not predict_results:
            return candidates

        for res in predict_results:
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])
            for text, score in zip(rec_texts, rec_scores):
                clean_text = str(text).strip()
                if clean_text:
                    candidates.append((clean_text, float(score)))
        return candidates

    # Attempt 1: Preprocessed crop (grayscale + CLAHE + 3x resize + sharpening)
    prep_crop = preprocess_crop_for_ocr(crop, scale_factor=3.0)
    pred_res = ocr_engine.predict(prep_crop)
    candidates = parse_paddle_results(pred_res)
    print(f"[PACKET OCR] Preprocessed 3x candidates: {candidates}")

    # Attempt 2: If empty, try simple 3x cubic upscale
    if not candidates:
        simple_3x = cv2.resize(
            crop,
            (crop.shape[1] * 3, crop.shape[0] * 3),
            interpolation=cv2.INTER_CUBIC
        )
        pred_res = ocr_engine.predict(simple_3x)
        candidates = parse_paddle_results(pred_res)
        print(f"[PACKET OCR] Simple 3x candidates: {candidates}")

    # Attempt 3: If still empty, try raw crop
    if not candidates:
        pred_res = ocr_engine.predict(crop)
        candidates = parse_paddle_results(pred_res)
        print(f"[PACKET OCR] Raw crop candidates: {candidates}")

    if not candidates:
        print("[PACKET OCR] PaddleOCR produced no detections.")
        return "", 0.0, []

    # Selection logic: Prioritize 6-digit numeric candidate
    six_digit_match = None
    digit_match = None

    for text, score in candidates:
        # Check pure digits
        digits = re.sub(r"\D", "", text)
        if len(digits) == 6 and six_digit_match is None:
            six_digit_match = (digits, score)
        elif len(digits) >= 4 and digit_match is None:
            digit_match = (digits, score)

    if six_digit_match is not None:
        best_text, best_score = six_digit_match
        print(f"[PACKET OCR] Selected exact 6-digit packet number: {best_text} (conf: {best_score:.4f})")
    elif digit_match is not None:
        best_text, best_score = digit_match
        print(f"[PACKET OCR] Selected numeric candidate: {best_text} (conf: {best_score:.4f})")
    else:
        # Highest confidence text candidate
        best_candidate = max(candidates, key=lambda x: x[1])
        best_text, best_score = best_candidate
        print(f"[PACKET OCR] Selected top confidence candidate: {best_text} (conf: {best_score:.4f})")

    return best_text, best_score, candidates
