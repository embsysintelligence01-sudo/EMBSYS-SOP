# Production integration test results

Camera-free automated validation. Physical USB disconnect/reconnect, serial hardware, monitor hot unplug and power-cycle testing are pending.

| Check | Status | Time (s) | Detail |
|---|---|---:|---|
| standalone-sealing-replay | PASSED | 19.92 |  · outputs/production-validation/standalone-sealing-replay.log |
| reset-replay | PASSED | 23.76 |  · outputs/production-validation/reset-replay.log |
| crash-replay | PASSED | 21.67 |  · outputs/production-validation/crash-replay.log |
| debug-close-replay | PASSED | 14.46 |  · outputs/production-validation/debug-close-replay.log |
| purity-regression | PASSED | 165.539 | Ran 519 tests in 165.539s; OK · outputs/production-purity-regression.log |
| model-budget | PASSED | 2.562 | Ran 1 test in 2.562s; OK · outputs/model-budget-tests.log |
| model-cache-replay | PASSED | 19.07 |  · outputs/production-validation/model-cache-replay.log |
| offline-sealing-ocr | PASSED | 5.58 |  · outputs/production-validation/offline-sealing-ocr.log |
| local-regression-fixtures | PASSED | 14.66 | Ran 51 tests in 12.262s; OK · outputs/production-validation/local-regression-fixtures.log |
| sealing-regression | PASSED | 11.09 | Ran 356 tests in 8.904s; OK · outputs/production-validation/sealing-regression.log |
| linked-replay | PASSED | 26.95 |  · outputs/production-validation/linked-replay.log |
| publication-jobs | PASSED | 0.47 | Ran 2 tests in 0.349s; OK · outputs/production-validation/publication-jobs.log |
| http-service-recovery | PASSED | 12.29 |  · outputs/production-validation/http-service-recovery.log |
| virtual-panel | PASSED | 0.21 | Ran 2 tests in 0.071s; OK · outputs/production-validation/virtual-panel.log |
| release-regression | PASSED | 25.45 | Ran 94 tests in 20.863s; OK (skipped=1) · outputs/production-validation/release-regression.log |
| release-imports | PASSED | 4.07 |  · outputs/production-validation/release-imports.log |
| releasing-replay | PASSED | 12.6 |  · outputs/production-validation/releasing-replay.log |
| controller-api | PASSED | 4.98 | Ran 22 tests in 4.475s; OK · outputs/production-validation/controller-api.log |
| linked-debug-close-replay | PASSED | 29.01 |  · outputs/production-validation/linked-debug-close-replay.log |
| source-model-integrity | PASSED | 10.35 |  · outputs/production-validation/source-model-integrity.log |
| headless-handoff | PASSED | 2.12 | Ran 4 tests in 0.083s; OK · outputs/production-validation/headless-handoff.log |
| purity-diagnostics | PASSED | 2.23 | Ran 3 tests in 0.108s; OK · outputs/production-validation/purity-diagnostics.log |
| desktop-integration | PASSED | 13.7 | Ran 48 tests in 11.579s; OK (skipped=2) · outputs/production-validation/desktop-integration.log |
