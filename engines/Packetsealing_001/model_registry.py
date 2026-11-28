"""Compatibility model-registry import for directly executed legacy tools."""
from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from shared.inference.model_registry import *
