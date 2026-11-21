"""Temporary package alias for legacy tests/tools; canonical code: engines.sealing."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from shared.compat import alias
alias(__name__, 'engines.sealing')
