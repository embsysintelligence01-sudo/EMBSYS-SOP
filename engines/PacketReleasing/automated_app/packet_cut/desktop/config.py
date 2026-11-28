"""Configuration for this application."""
from __future__ import annotations

from typing import Tuple
from ._imports import *

def _find_model_dir() -> Path:
    env_dir = os.environ.get("PACKET_CUT_MODELS_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    for p in (
        Path(__file__).resolve().parents[2] / "models",
        Path(__file__).resolve().parents[3] / "models",
        Path(r"D:\Project\Embsys\Packet-cut-new\models"),
    ):
        if p.exists():
            return p
    return Path(r"D:\Project\Embsys\Packet-cut-new\models")


MODEL_DIR = _find_model_dir()

PACKET_MODEL_PATH = MODEL_DIR / "packet-seg.onnx"

HAND_MODEL_PATH = MODEL_DIR / "hand-seg.onnx"

SCISSOR_MODEL_PATH = MODEL_DIR / "scissor-seg.pt"

TOPSTRIP_MODEL_PATH = MODEL_DIR / "strip-seg.onnx"

PACKET_NUMBER_MODEL_PATH = MODEL_DIR / "packet-no-obb.onnx"

# Use the configured model directory for every model unless explicitly overridden.
CUT_MODEL_PATH = Path(
    os.environ.get("PACKET_CUT_MODEL", "")
).expanduser() if os.environ.get("PACKET_CUT_MODEL") else (
    MODEL_DIR / "cut-seg.pt"
)
CUT_MODEL_CLASS_ID = int(os.environ.get("PACKET_CUT_CLASS_ID", "0"))
CUT_MODEL_MIN_CONF = float(os.environ.get("PACKET_CUT_MIN_CONF", "0.70"))
JEWEL_MODEL_PATH = MODEL_DIR / "jeweltrack.onnx"
JEWEL_CONF = float(os.environ.get("JEWEL_CONF", "0.25"))

IMGSZ = 320

PACKET_CONF = float(os.environ.get("PACKET_CONF", "0.12"))

HAND_CONF = float(os.environ.get("HAND_CONF", "0.25"))

SCISSOR_CONF = float(os.environ.get("SCISSOR_CONF", "0.22"))

TOPSTRIP_CONF = 0.25

NUMBER_IMGSZ = 640

NUMBER_CONF = 0.05

BLUR_VARIANCE_MIN = 25.0

OUTPUT_DIR = Path("packet_cut_outputs")

# Automatic stage recovery, retry, backoff, and timeout configuration
AUTO_INFER_MAX_RETRIES = int(os.environ.get("AUTO_INFER_MAX_RETRIES", "2"))
AUTO_INFER_BACKOFF_SEC = float(os.environ.get("AUTO_INFER_BACKOFF_SEC", "0.15"))
AUTO_EMPTY_TESTBED_FRAMES = int(os.environ.get("AUTO_EMPTY_TESTBED_FRAMES", "3"))
AUTO_EMPTY_TESTBED_TIMEOUT_SEC = float(os.environ.get("AUTO_EMPTY_TESTBED_TIMEOUT_SEC", "12.0"))
AUTO_FRONTSIDE_TIMEOUT_SEC = float(os.environ.get("AUTO_FRONTSIDE_TIMEOUT_SEC", "20.0"))
AUTO_OCR_MAX_ATTEMPTS = int(os.environ.get("AUTO_OCR_MAX_ATTEMPTS", "12"))
AUTO_OCR_TIMEOUT_SEC = float(os.environ.get("AUTO_OCR_TIMEOUT_SEC", "18.0"))
AUTO_BACKSIDE_CONFIRM_FRAMES = int(os.environ.get("AUTO_BACKSIDE_CONFIRM_FRAMES", "4"))
AUTO_BACKSIDE_TIMEOUT_SEC = float(os.environ.get("AUTO_BACKSIDE_TIMEOUT_SEC", "25.0"))
AUTO_CUT_VERIFY_MAX_RETRIES = int(os.environ.get("AUTO_CUT_VERIFY_MAX_RETRIES", "2"))
AUTO_MOUTH_INSPECT_TIMEOUT_SEC = float(os.environ.get("AUTO_MOUTH_INSPECT_TIMEOUT_SEC", "25.0"))
AUTO_JEWELLERY_COUNT_TIMEOUT_SEC = float(os.environ.get("AUTO_JEWELLERY_COUNT_TIMEOUT_SEC", "45.0"))
AUTO_JEWELLERY_STABLE_FRAMES = int(os.environ.get("AUTO_JEWELLERY_STABLE_FRAMES", "8"))
AUTO_JEWELLERY_STABLE_SEC = float(os.environ.get("AUTO_JEWELLERY_STABLE_SEC", "1.5"))

# Default Cutting / Inspection ROI (excludes bottom table rail y >= 405)
DEFAULT_CUTTING_ROI = (40, 40, 600, 400)


def get_cutting_roi(frame_shape: Tuple[int, int] = (480, 640)) -> Tuple[int, int, int, int]:
    """Returns valid (x1, y1, x2, y2) cutting ROI within frame bounds.

    Filters out background clutter (such as the metal bottom extrusion rail)
    by strictly bounding the workspace.
    """
    h, w = frame_shape[:2]
    env_roi = os.environ.get("PACKET_CUT_ROI") or os.environ.get("INSPECTION_TESTBED_ROI")
    if env_roi:
        try:
            parts = [int(p.strip()) for p in env_roi.split(",") if p.strip()]
            if len(parts) == 4:
                return (max(0, parts[0]), max(0, parts[1]), min(w, parts[2]), min(h, parts[3]))
        except Exception:
            pass
    # Safe default: x in [40, 600], y in [40, 400] on 640x480 frame
    return (40, 40, min(600, w - 20), min(400, int(h * 0.8333)))

