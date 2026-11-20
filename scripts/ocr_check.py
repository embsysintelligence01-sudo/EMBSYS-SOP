"""Check existing isolated OCR only after all three required caches exist."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.config import PACKET
cache = Path.home()/'.paddlex/official_models'
for name in ('PP-OCRv5_mobile_det','en_PP-OCRv5_mobile_rec','PP-LCNet_x1_0_textline_ori'):
    directory=cache/name
    if not directory.is_dir() or not any(directory.glob('*.json')) or not any(directory.glob('*.pdiparams')):
        raise SystemExit('Required local PaddleOCR cache missing; offline validation unavailable')
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True')
sys.path.insert(0,str(PACKET))
from standalone.packet_ocr import check_ocr
check_ocr(PACKET)
