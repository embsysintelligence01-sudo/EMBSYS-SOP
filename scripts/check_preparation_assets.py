#!/usr/bin/env python3
"""Read-only integrity checks for the preparation snapshot and shared registry."""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.inference.model_registry import validate

snapshot = json.loads((ROOT/'configs/packet_releasing_snapshot.json').read_text())
errors = validate()
for prefix, entries in [('engines/PacketReleasing', snapshot['files']), ('', snapshot['existing_engine_source_baseline'])]:
    for relative, digest in entries.items():
        path = ROOT/prefix/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append({'file':str(path.relative_to(ROOT)), 'error':'snapshot checksum mismatch'})
print(json.dumps({'copied_files':len(snapshot['files']), 'existing_source_files':len(snapshot['existing_engine_source_baseline']), 'errors':errors},indent=2))
raise SystemExit(bool(errors))
