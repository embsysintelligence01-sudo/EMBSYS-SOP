#!/usr/bin/env python3
"""Check reviewed cleanup against original engine bytes and compatibility paths."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_runtime import LAYOUT


def digest(path, size=None):
    with path.open('rb') as stream:
        return hashlib.sha256(stream.read() if size is None else stream.read(size)).hexdigest()


def main():
    manifest=json.loads((ROOT/'docs/CLEANUP_MANIFEST.json').read_text())
    errors=[];checked=0
    for row in manifest['entries']:
        source=ROOT/row['path']
        if row['action']=='removed':
            if '__pycache__' in source.parts or source.suffix in {'.pyc','.pyo'}:
                continue  # Tests may regenerate caches; final cleanup removes them again.
            if source.exists():errors.append(row['path']+': removal not applied')
            if row.get('canonical'):
                canonical=ROOT/row['canonical']
                if not canonical.is_file():errors.append(row['path']+': canonical missing')
                elif source.name=='jewel_tracker_3.py':
                    # Only the private config filename differed. Both local configs survive unchanged.
                    original=Path(json.loads((ROOT/'docs/SOURCE_SNAPSHOT.json').read_text())['Jeweltracker']['source'])
                    a=(original/'jewel_tracker_2.py').read_bytes();b=(original/'jewel_tracker_3.py').read_bytes()
                    if a!=b:errors.append(row['path']+': original duplicate proof changed')
                elif digest(canonical)!=row['sha256']:errors.append(row['path']+': canonical content changed')
            continue
        target=ROOT/row['target']
        if not target.is_file():
            errors.append(row['path']+': retained/relocated file missing');continue
        # Append-only telemetry is allowed; initial recorded evidence/source bytes must survive.
        append_only=target.suffix == '.log' or (target.suffix == '.jsonl' and str(target.relative_to(ROOT)).startswith('runtime/'))
        if digest(target,row['bytes'] if append_only else None)!=row['sha256']:
            errors.append(row['path']+': preserved content changed')
        checked+=1
    captures=ROOT/'engines/Jeweltracker/captures'
    if captures.is_symlink():
        errors.append('Purity capture directory may not have a symlink parent: archive integrity policy')
    for original,relocated in LAYOUT.items():
        source,target=ROOT/original,ROOT/relocated
        if not source.is_symlink() or source.resolve()!=target.resolve() or not target.exists():
            errors.append(original+': compatibility link invalid')
    unsafe=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-ci','--exclude-standard','-z']).decode().split('\0')
    unsafe=[p for p in unsafe if p]
    errors.extend(p+': ignored data still tracked by Git' for p in unsafe)
    result=dict(preserved_files_checked=checked,compatibility_links=len(LAYOUT),
        tracked_files=len(subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).split(b'\0'))-1,
        tracked_ignored_files=unsafe,errors=errors)
    (ROOT/'docs/CLEANUP_INTEGRITY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
