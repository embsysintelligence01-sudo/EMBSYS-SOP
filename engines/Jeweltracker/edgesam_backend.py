"""Temporary compatibility alias; implementation moved to shared.inference.edgesam."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from shared.compat import alias
_module=alias(__name__, 'shared.inference.edgesam')
if __name__ == '__main__':
    if hasattr(_module, 'main'):
        raise SystemExit(_module.main())
