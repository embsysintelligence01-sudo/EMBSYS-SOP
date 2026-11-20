# Directory tree

```text
EMBSYS-SOP/
├── launcher.py
├── run.sh
├── pyproject.toml
├── README.md
├── .gitignore
├── .git/                         # only Git repository in the new tree
├── app/
│   ├── __init__.py
│   ├── main_window.py
│   ├── process_manager.py
│   └── config.py
├── engines/
│   ├── Jeweltracker/
│   │   ├── jewel_tracker_5.py
│   │   ├── acid_test/, acidphintegration/, new_rub_detection/
│   │   ├── tests/, regression_assets.py, surveillance.py
│   │   ├── camera/evidence/recovery/reporting modules
│   │   ├── model folders and classification assets (local, ignored)
│   │   └── camera/ROI/private default JSON (local, ignored)
│   └── Packetsealing_001/
│       ├── packet_app.py
│       ├── standalone/
│       ├── tests/, packet_mouth_lab/, retained legacy tools
│       ├── model onnx/             (local, ignored)
│       └── camera/ROI settings     (local, ignored)
├── configs/
│   ├── app.toml
│   ├── local.toml                 (local, ignored)
│   ├── *.example.json
│   └── *-environment.txt          (installed version inventories)
├── development/                   # preserved source/tools/candidate models/fixtures
│   ├── README.md, run.sh, run_legacy.py
│   ├── legacy/Jeweltracker/        # versions 2/6/similarity and historical snapshot/tests
│   ├── legacy/Packetsealing_001/   # pickup snapshot
│   ├── models/                    # local candidate models, ignored binaries
│   ├── fixtures/                  # local sample images, ignored
│   ├── benchmarks/
│   └── editor/
├── runtime/                       # packet output and telemetry, ignored
│   ├── Jeweltracker/
│   └── Packetsealing_001/
├── scripts/
│   ├── install.sh
│   ├── configure.py
│   ├── validate.py
│   ├── engine_check.py
│   ├── ocr_check.py
│   ├── audit.py
│   ├── cleanup.sh
│   ├── prepare_runtime.py
│   ├── cleanup_layout.py
│   ├── verify_cleanup.py
│   └── check_storage_paths.py
├── tests/
│   ├── test_launcher.py
│   └── test_integration.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CHANGELOG.md
│   ├── CLEANUP_REPORT.md
│   ├── CLEANUP_MANIFEST.json
│   ├── CLEANUP_REFERENCES.json
│   ├── CLEANUP_INTEGRITY.json
│   ├── DEPENDENCY_REPORT.md
│   ├── DEPENDENCIES.json
│   ├── SOURCE_SNAPSHOT.json
│   ├── COPY_MANIFEST.json
│   ├── SOURCE_CHANGES.json
│   ├── SECURITY.md
│   ├── AUDIT.json
│   ├── VALIDATION.md
│   ├── VALIDATION.json
│   ├── VALIDATION_NOTES.md
│   ├── SURVEILLANCE_HANDOFF.md
│   ├── SURVEILLANCE_LOGIC_AUDIT.json
│   └── TREE.md
└── outputs/validation/            (private local logs, ignored)
```

Purity captures remain physical under engines/Jeweltracker/captures/ to satisfy archive integrity. Packet output and telemetry link into runtime/; local sample images link into development/fixtures/. Existing historical evidence is not migrated into the new production tree. The tree groups the preserved engine modules for readability; `COPY_MANIFEST.json` lists individual copied files.
