# Post-cleanup validation report

No new regression failure identities were introduced. Existing production and historical baseline failures remain explicitly FAILED. After restoring physical purity capture storage, actual containment and packet recording/export checks plus all 21 focused handoff tests pass.

See [cleanup report](CLEANUP_REPORT.md), [validation notes](VALIDATION_NOTES.md), and [before/after failure comparison](CLEANUP_REGRESSION_COMPARISON.json). Raw logs remain private and ignored under outputs/validation/.

| Check | Result | Detail |
|---|---|---|
| launcher-tests | PASSED | exit=0; 9.5s; private log: outputs/validation/launcher-tests.log; Ran 10 tests in 9.164s; OK |
| launcher-ui | PASSED | exit=0; 0.1s; private log: outputs/validation/launcher-ui.log |
| purity-imports | PASSED | exit=0; 2.9s; private log: outputs/validation/purity-imports.log |
| slic-imports | PASSED | exit=0; 1.9s; private log: outputs/validation/slic-imports.log |
| purity-ui | PASSED | exit=0; 1.9s; private log: outputs/validation/purity-ui.log |
| packet-ui | PASSED | exit=0; 1.9s; private log: outputs/validation/packet-ui.log |
| purity-models | PASSED | exit=0; 6.0s; private log: outputs/validation/purity-models.log |
| packet-models | PASSED | exit=0; 4.7s; private log: outputs/validation/packet-models.log |
| offline-ocr | PASSED | exit=0; 7.6s; private log: outputs/validation/offline-ocr.log |
| purity-regression | FAILED | exit=1; 145.8s; private log: outputs/validation/purity-regression.log; Ran 414 tests in 143.388s; FAILED (failures=1) |
| slic-regression | FAILED | exit=1; 140.4s; private log: outputs/validation/slic-regression.log; Ran 414 tests in 138.018s; FAILED (failures=1) |
| packet-regression | PASSED | exit=0; 10.7s; private log: outputs/validation/packet-regression.log; Ran 352 tests in 8.530s; OK |
| rub-regression | PASSED | exit=0; 2.4s; private log: outputs/validation/rub-regression.log; Ran 43 tests in 2.025s; OK |
| mouth-lab-regression | FAILED | exit=1; 3.3s; private log: outputs/validation/mouth-lab-regression.log; Ran 12 tests in 1.653s; FAILED (failures=1) |
| historical-regression | FAILED | exit=1; 23.9s; private log: outputs/validation/historical-regression.log; Ran 50 tests in 22.000s; FAILED (failures=2, errors=1) |
| pickup-self-test | PASSED | exit=0; 0.4s; private log: outputs/validation/pickup-self-test.log |
| pickup-motion-self-test | PASSED | exit=0; 0.4s; private log: outputs/validation/pickup-motion-self-test.log |
| archived-pickup-self-test | PASSED | exit=0; 0.4s; private log: outputs/validation/archived-pickup-self-test.log |
| model-copy-integrity | PASSED | 47 original model/asset roles checked at retained, relocated or identical canonical paths; missing/changed:  |
| required-model-paths | PASSED | 17 required assets; 3 temporal models; missing:  |
| cleanup-integrity | PASSED | exit=0; 4.7s; private log: outputs/validation/cleanup-integrity.log |
| storage-paths | PASSED | exit=0; 2.3s; private log: outputs/validation/storage-paths.log |
| source-security-audit | PASSED | exit=0; 10.4s; private log: outputs/validation/source-security-audit.log |
| handoff-regression | PASSED | exit=0; 3.6s; private log: outputs/validation/handoff-regression.log; Ran 17 tests in 0.368s; OK |
| surveillance-export-regression | PASSED | exit=0; 2.6s; private log: outputs/validation/surveillance-export-regression.log; Ran 4 tests in 0.595s; OK |
| live-camera-and-operator-SOP | UNTESTED | Requires bench camera and operator: live ROI, full purity/sealing sequence, evidence export and physical camera release. No camera connection attempted. |
| optional-reviewer-and-legacy-hardware | UNTESTED | External recording reviewer, Qt6 standalone acid UI and legacy Hailo/pickup hardware not exercised. |
| cleanup-baseline-comparison | PASSED | Production failure identities match the untouched-source baseline; historical snapshot failures/errors match the before-cleanup run. No new regression failure. |
| cleanup-storage-restoration | PASSED | Purity captures restored to physical original location after existing guard rejected a symlink parent. Actual containment and controlled packet recording/export checks pass; saved evidence retained. |
| prepared-storage-install-and-lock | PASSED | Install is idempotent, installs/upgrades zero packages; preparation and production launching share exclusive ownership. Development registry and shell syntax checked. |
