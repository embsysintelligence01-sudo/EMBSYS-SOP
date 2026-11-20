"""Resolve original migration assets after reviewed cleanup; never used by SOP inference."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def asset_location(relative):
    relative = 'engines/' + relative.removeprefix('engines/')
    manifest = ROOT/'docs/CLEANUP_MANIFEST.json'
    if not manifest.exists():
        return ROOT/relative
    entries = json.loads(manifest.read_text())['entries']
    entry = next((row for row in entries if row['path'] == relative), None)
    if entry is None:
        return ROOT/relative
    if entry['action'] == 'removed':
        canonical = entry.get('canonical')
        return ROOT/canonical if canonical else None
    return ROOT/entry['target']
