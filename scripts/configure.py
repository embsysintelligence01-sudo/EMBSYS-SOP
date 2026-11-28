"""Discover existing environments read-only and record only local interpreter paths."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
original = ROOT.parent / 'jewel_tracker/jewel_tracker'
packet = ROOT.parent / 'packet_sealing'
paths = {
    'purity_python': Path(os.environ.get('PURITY_PYTHON', original / '.venv-slic/bin/python')),
    'packet_python': Path(os.environ.get('PACKET_PYTHON', original / '.venv-slic/bin/python')),
    'ocr_python': Path(os.environ.get('PACKET_OCR_PYTHON', packet / '.venv-ocr/bin/python')),
}
# Engine 3 is optional at install time and remains development-only.
release = Path(os.environ.get('RELEASING_PYTHON', ROOT.parent / 'packet-cut-system/.venv-ubuntu/bin/python'))
if release.is_file() or os.environ.get('RELEASING_PYTHON'):
    paths['releasing_python'] = release
for key, path in paths.items():
    # Do not resolve symlinks: the environment path controls site-packages.
    if not path.is_file():
        raise SystemExit(f'{key} unavailable; set its environment override before install.sh')
config = ROOT / 'configs/local.toml'
if not config.exists():
    entries = [f'{key} = {json.dumps(os.path.relpath(path, ROOT))}' for key, path in paths.items()]
    reviewer = ROOT.parent / 'puritytesting/acid_temporal_lab'
    if reviewer.is_dir():
        entries.append('reviewer_dir = ' + json.dumps(os.path.relpath(reviewer, ROOT)))
    config.write_text('\n'.join(entries) + '\n')
    config.chmod(0o600)
else:
    print('Keeping existing configs/local.toml')
sys.path.insert(0, str(ROOT))
from app.config import settings, resolve_path
paths = {key: resolve_path(settings()[key]) for key in paths}
# Reproducible version inventory without VCS URLs or private index credentials.
code = '''import importlib.metadata as m, json
print(json.dumps(sorted({d.metadata['Name']+'=='+d.version for d in m.distributions() if d.metadata['Name']})))'''
for key, path in paths.items():
    rows = json.loads(subprocess.check_output([str(path), '-c', code], text=True))
    (ROOT / 'configs' / (key.replace('_python', '') + '-environment.txt')).write_text('\n'.join(rows) + '\n')
print('Configured existing environments; installed/upgraded zero packages.')
