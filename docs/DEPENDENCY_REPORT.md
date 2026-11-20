# Dependency and cleanup report

Generated before production copying. Full static import inventory: DEPENDENCIES.json. Dynamic imports and external consumers cannot be proven by AST alone.

## Omitted duplicates

- `Jeweltracker/jewel_tracker_5.py.before_linux_antiflicker.bak`: historical backup/copy filename; no source, test, shell or packaging reference to its filename.
- `Jeweltracker/jewel_tracker copy.py`: historical backup/copy filename; no source, test, shell or packaging reference to its filename.
- `Jeweltracker/jewel_tracker_similarity copy.py`: historical backup/copy filename; no source, test, shell or packaging reference to its filename.
- `Jeweltracker/README-linux.md.before_live_scheduling.bak`: historical backup/copy filename; no source, test, shell or packaging reference to its filename.
- `Jeweltracker/camera.json.before_linux_antiflicker.bak`: historical backup/copy filename; no source, test, shell or packaging reference to its filename.
- `Jeweltracker/jewel_tracker copy 2.py`: historical backup/copy filename; no source, test, shell or packaging reference to its filename.
- `Jeweltracker/jewel_tracker_5.py.before_live_scheduling.bak`: historical backup/copy filename; no source, test, shell or packaging reference to its filename.
- `Jeweltracker/jewel_tracker_5.py.before_temporal_acid.bak`: historical backup/copy filename; no source, test, shell or packaging reference to its filename.
- `Jeweltracker/acid_test/integration.py.before_observation_timing.bak`: historical backup/copy filename; no source, test, shell or packaging reference to its filename.
- `Jeweltracker/acid_test/sequence.py.before_observation_timing.bak`: historical backup/copy filename; no source, test, shell or packaging reference to its filename.

## Retained legacy and optional features

- All numbered tracker modules (including versions 2/3/6), original tracker and similarity modules: regression tests, packaging references or uncertain external usage.
- tracker_update_20261027: retained for uncertain historical tooling dependencies; not selected by launcher.
- pickup_trigger_test.py and pickup_trigger_motion_class.py: imported by regression tests; retained.
- service.py, routes.py, triger_motion_rohan and packet_mouth_lab: optional/legacy workflows; retained, including Hailo imports. Hailo is not a main launcher dependency and is not installed.
- All camera, evidence, recording, recovery, PDF, OCR, training/packaging modules and functional tests retained.
- Original outer backup directories are untouched and are not part of the live repository being copied.

## Copy exclusions

Directory names excluded recursively: .git, .mypy_cache, .pytest_cache, .vscode, __pycache__, build, captures, dataset, diagnostics, dist, evidence, outputs, recordings, similarity_crops, .venv*. Generated logs/video/zip/bytecode excluded; original datasets and evidence remain in originals. Model binaries retained locally and ignored by Git. No code deduplication based on similarity alone.

## Import connections

- packet_app.py -> standalone.app -> jewel_tracker_5 and standalone modules. JEWEL_TRACKER_DIR is injected into every managed engine.
- jewel_tracker_5 imports acid_test.integration, acidphintegration, continuous_testing, rub_preparation, int8_models and related evidence/camera modules.
- Subprocess working directories remain engine-local so existing relative assets and configuration remain compatible.

## Subsequent verified cleanup

The earlier conservative retention list above is the initial migration record. The actual subsequent decisions and relocation targets are recorded in [CLEANUP_REPORT.md](CLEANUP_REPORT.md), with static/dynamic references in CLEANUP_REFERENCES.json and the applied per-file map in CLEANUP_MANIFEST.json. Some historical files initially retained in engines now live under development/; runtime evidence has compatibility links to runtime/. Current launch/test/import paths and algorithms remain unchanged.
