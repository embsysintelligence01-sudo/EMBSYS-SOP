#!/usr/bin/env bash
# Remove regenerable code caches only; saved evidence/models remain untouched.
set -euo pipefail
APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - "$APP_ROOT" <<'PY'
from pathlib import Path
import shutil, sys
root=Path(sys.argv[1]).resolve()
protected={'captures','outputs','recordings','evidence','diagnostics','dataset','fixtures','runtime','.git'}
for pattern in ('__pycache__','.pytest_cache'):
    for path in root.rglob(pattern):
        relative=path.relative_to(root)
        if path.is_symlink() or any(p in protected or p.startswith('.venv') for p in relative.parts):
            continue
        if not path.is_dir():continue
        if pattern=='__pycache__' and any(p.is_file() and p.suffix not in {'.pyc','.pyo'} for p in path.rglob('*')):
            continue  # A cache-named folder containing unknown data is preserved.
        shutil.rmtree(path)
print('Removed regenerable source caches; runtime evidence, fixtures and models preserved.')
PY
