#!/usr/bin/env python3
"""Record the explicitly scoped source changes; fails on unreviewed detector edits."""

from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]
old = json.loads((ROOT / "configs/packet_releasing_snapshot.json").read_text())
changes = {}
allowed = {
    'engines/PacketReleasing/automated_app/packet_cut/desktop/automation.py':'Default-on visualization config only; detection/verification configuration unchanged.',
    'engines/PacketReleasing/automated_app/packet_cut/desktop/release_observer.py':'Skip final live-preview overlay assembly in production; independent checks and stage evidence generation unchanged.',
    'engines/PacketReleasing/automated_app/tests/test_release_performance.py':'Verify headless preview suppression retains hand/scissor detection and cutting photos.',

    "engines/PacketReleasing/automated_app/packet_cut/desktop/ocr.py": "Optional explicit offline OCR model directories; recognition, preprocessing and confidence rules unchanged.",
    "engines/purity/tracker.py": "Optional preview rendering, development autoconnect flag; default worker processing unchanged.",
    "engines/sealing/workflow.py": "Attempt-scoped output root, actual capture-release barrier, optional live preview; evidence drawing remains active.",
    "engines/PacketReleasing/automated_app/packet_cut/desktop/worker.py": "Actual acquisition-release signal before finalization, optional preview conversion, bounded diagnostic snapshot.",
    "engines/Jeweltracker/tests/test_classification_crop.py": "Test current cropped detector coordinate restoration instead of outdated whole-frame input assumption.",
    "engines/Jeweltracker/tests/test_departure_trigger_removed.py": "Keep rubbing verification gated while independently testing the existing jewel-placement baseline.",
    "engines/Jeweltracker/tests/test_overlap_trigger.py": "Assert existing independent placement baseline without accepting false contact/rubbing evidence.",
    "engines/Jeweltracker/tests/test_stone_model.py": "Provide the filename suffix required by installed Ultralytics for the unchanged OpenVINO reference.",
}
for prefix, entries in [
    ("engines/PacketReleasing/", old["files"]),
    ("", old["existing_engine_source_baseline"]),
]:
    for relative, before in entries.items():
        name = prefix + relative
        after = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if after != before:
            if name not in allowed:
                raise RuntimeError("Unreviewed change: " + name)
            changes[name] = dict(before=before, after=after, reason=allowed[name])
value = dict(
    backup_branch="backup/pre-production-20261128",
    changes=changes,
    preserved_environment="Windows experiment environment relocated intact to runtime/preserved-windows-environment; original path is a compatibility link",
    models="No original binary replaced; model hashes retained",
    deleted_source_files=[],
)
(ROOT / "configs/production_migration.json").write_text(
    json.dumps(value, indent=2) + "\n"
)
print("Recorded", len(changes), "reviewed source migrations")
