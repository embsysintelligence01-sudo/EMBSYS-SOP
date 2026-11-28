"""Ocr module."""

from ._imports import *

from .config import *

_OCR_ENGINE = None
_NUMBER_RECOGNIZER = None

def get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        options = {}
        if os.environ.get("PACKET_OCR_DET_MODEL"):
            options["text_detection_model_name"] = os.environ["PACKET_OCR_DET_MODEL"]
        if os.environ.get("PACKET_OCR_REC_MODEL"):
            options["text_recognition_model_name"] = os.environ["PACKET_OCR_REC_MODEL"]
        if os.environ.get("WORKFLOW_DEVICE") == "cpu":
            options["device"] = "cpu"
            options["cpu_threads"] = int(os.environ.get("OMP_NUM_THREADS", "4"))
        _OCR_ENGINE = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **options,
        )
    return _OCR_ENGINE

def preprocess_crop_for_ocr(crop, scale_factor=3.0):
    if crop is None or crop.size == 0:
        return None
    height, width = crop.shape[:2]
    # Upscale small number crops without sending a huge crop to PaddleOCR.
    scale_factor = min(scale_factor, 960 / max(width, 1), 256 / max(height, 1))
    scaled = cv2.resize(
        crop, (0, 0),
        fx=scale_factor, fy=scale_factor,
        interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(contrast, -1, kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

def run_ocr(crop, max_passes=3):
    if crop is None or crop.size == 0:
        return None, 0.0, []
    ocr_engine = get_ocr_engine()
    if ocr_engine is None:
        return None, 0.0, []

    def parse_paddle_results(predict_results):
        candidates = []
        if not predict_results:
            return candidates
        res = predict_results[0] if isinstance(predict_results, list) else predict_results
        if hasattr(res, "keys"):
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            for t, s in zip(texts, scores):
                txt = re.sub(r"[^0-9A-Za-z]", "", str(t).strip())
                if txt:
                    candidates.append((txt, float(s)))
        elif isinstance(res, list):
            for item in res:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    if isinstance(item[1], (list, tuple)) and len(item[1]) == 2:
                        txt = re.sub(r"[^0-9A-Za-z]", "", str(item[1][0]).strip())
                        score = float(item[1][1])
                        if txt:
                            candidates.append((txt, score))
        return candidates

    prep_crop = preprocess_crop_for_ocr(crop, scale_factor=3.0)
    pred_res = ocr_engine.ocr(prep_crop)
    candidates = parse_paddle_results(pred_res)

    if not candidates and max_passes >= 2:
        simple_3x = cv2.resize(crop, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        pred_res = ocr_engine.ocr(simple_3x)
        candidates = parse_paddle_results(pred_res)

    if not candidates and max_passes >= 3:
        pred_res = ocr_engine.ocr(crop)
        candidates = parse_paddle_results(pred_res)

    if not candidates:
        return None, 0.0, []

    six_digits = [c for c in candidates if len(c[0]) == 6 and c[0].isdigit()]
    if six_digits:
        best = max(six_digits, key=lambda c: c[1])
        return best[0], best[1], candidates

    digits_only = [c for c in candidates if c[0].isdigit()]
    if digits_only:
        best = max(digits_only, key=lambda c: c[1])
        return best[0], best[1], candidates

    best = max(candidates, key=lambda c: c[1])
    return best[0], best[1], candidates


def get_number_recognizer():
    """Load only the recognizer: the packet-number model already found the text."""
    global _NUMBER_RECOGNIZER
    if _NUMBER_RECOGNIZER is None:
        from paddleocr import TextRecognition
        options = {}
        if os.environ.get("WORKFLOW_DEVICE") == "cpu":
            options.update(device="cpu", cpu_threads=int(os.environ.get("OMP_NUM_THREADS", "4")))
        _NUMBER_RECOGNIZER = TextRecognition(
            model_name=os.environ.get("PACKET_OCR_REC_MODEL", "en_PP-OCRv5_mobile_rec"),
            **options,
        )
    return _NUMBER_RECOGNIZER


def run_number_ocr(crop, allow_detection_fallback=False):
    """Recognize one cropped number line; periodically recover difficult layouts."""
    if crop is None or crop.size == 0:
        return None, 0.0, []
    prepared = preprocess_crop_for_ocr(crop)
    results = get_number_recognizer().predict(prepared)
    candidates = []
    for result in results:
        text = re.sub(r"[^0-9A-Za-z]", "", str(result.get("rec_text", "")).strip()).upper()
        confidence = float(result.get("rec_score", 0.0))
        if text:
            candidates.append((text, confidence))
    best = max(candidates, key=lambda item: item[1]) if candidates else (None, 0.0)
    if best[0] and len(best[0]) >= 4 and best[1] >= 0.65:
        return best[0], best[1], candidates
    # At most one full text-detection pass, only on a scheduled failed attempt.
    if allow_detection_fallback:
        return run_ocr(crop, max_passes=1)
    return best[0], best[1], candidates
