"""Read-only location of historical images used by regression tests only."""
import os
from pathlib import Path


def assets_root():
    return Path(os.environ.get('EMBSYS_REGRESSION_ASSETS', Path(__file__).resolve().parent))
