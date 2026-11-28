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
from services.source_integrity import verify_sources
errors.extend(verify_sources())
print(json.dumps({'copied_files':len(snapshot['files']), 'existing_source_files':len(snapshot['existing_engine_source_baseline']), 'errors':errors},indent=2))
raise SystemExit(bool(errors))
