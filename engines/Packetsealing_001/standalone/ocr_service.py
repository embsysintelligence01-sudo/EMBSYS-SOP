"""Compatibility subprocess entry; implementation is engines.sealing.ocr_service."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from engines.sealing.ocr_service import main
if __name__=='__main__':main()
