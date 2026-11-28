"""Configuration for this application."""
from ._imports import *

MODEL_DIR = Path(r"D:\Project\Embsys\Packet-cut-new\models")

PACKET_MODEL_PATH = MODEL_DIR / "packet-seg.onnx"

HAND_MODEL_PATH = MODEL_DIR / "hand-seg.onnx"

SCISSOR_MODEL_PATH = MODEL_DIR / "scissor-seg.pt"

TOPSTRIP_MODEL_PATH = MODEL_DIR / "strip-seg.onnx"

PACKET_NUMBER_MODEL_PATH = MODEL_DIR / "packet-no-obb.onnx"

PACKET_NUMBER_MODEL_PATTERN = str(
    MODEL_DIR / "packet-no-obb.onnx"
)

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

CAMERA_ID = 0

IMGSZ = 320

PACKET_CONF = float(os.environ.get("PACKET_CONF", "0.12"))

HAND_CONF = 0.30

SCISSOR_CONF = 0.30

TOPSTRIP_CONF = 0.25

NUMBER_IMGSZ = 640

NUMBER_CONF = 0.05

FRONT_NUMBER_TIMEOUT = 60

SEAL_INSPECTION_TIMEOUT = 90

CLEAR_WAIT_SECONDS = 5.0

CUT_VERIFY_SECONDS = 10.0

VERIFY_SAMPLE_INTERVAL = 0.75

MIN_PIECE_AREA_PX = 150

MIN_PIECE_AREA_RATIO = 0.08

MIN_TOTAL_AREA_RATIO = 0.55

MAX_TOTAL_AREA_RATIO = 1.45

MIN_BACKGROUND_GAP_PX = 4.0

MIN_NORM_GAP_RATIO = 0.015

MIN_GAP_COVERAGE_RATIO = 0.55

MIN_EXPOSED_GAP_PIXELS = 10

MAX_CENTROID_SHIFT_PX = 12.0

MAX_AREA_CHANGE_RATIO = 0.20

MIN_VALID_SAMPLE_RATIO = 0.90

MIN_VERIFY_SAMPLES = 8

BLUR_VARIANCE_MIN = 25.0

WINDOW = "Packet-Cut Verification"

OUTPUT_DIR = Path("packet_cut_outputs")
