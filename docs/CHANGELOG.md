# Change record

## Recording trigger correction — 2026-11-21

Restored provisional recording for covered or unconfirmed pickup candidates, the standalone's 0.45-second recent-pinch tolerance, and its minimum 720-pixel hand context crop. The start hold remains 0.5 seconds. The extended stone area stays fixed while a pinch is being checked and during the active cycle. Initial calibration continues while background home observations run.

Right, Left and Any selection now applies consistently to pinch and gold fallback. The selected hand blocks fallback for one second after being seen near the area. Ignored hands stay visible. Hand-model failures are reported explicitly and permit the enabled gold fallback to continue. The preview distinguishes PINCH HOLD and GOLD HOLD, and trigger manifests include the source, candidate identities, selected hands, provisional status and hand-model errors. Uncertain recordings remain UNASSIGNED.

Added direct comparisons against the unchanged standalone recorder using identical synthetic frames for covered homes, brief landmark misses, no hand and occupied homes. Added regressions for provisional identity, fallback selection, calibration and area freezing. Existing rubbing, acid, classification and handoff checks continue independently.

## 0.1.0 — 2026-11-20

Created a new sibling `EMBSYS-SOP` from local working files, without downloading either GitHub repository. Preserved source branches, commits, working-tree status and launcher names in `SOURCE_SNAPSHOT.json`. `COPY_MANIFEST.json` records hashes before integration edits. Original directories and their Git history are the untouched backups. Initialized a separate root repository on `main`; no remote, commit or push.

Added the PyQt5 launcher, location-based configuration, managed subprocesses with exclusive camera ownership, setup/environment inventory, validation, cleanup, examples and documentation. The installer reuses existing Python environments without package changes. Runtime evidence is stored in the new engines, with historical evidence remaining in the originals.

Removed only the ten unreferenced backup/copy files listed in `DEPENDENCY_REPORT.md`, from the new production copy. Historical numbered trackers, update folders, pickup experiments, Hailo modules, training/packaging sources and optional UIs were retained where usage is uncertain or tests/features reference them. Full static imports are in `DEPENDENCIES.json`.

Integration changes:

- `Packetsealing_001/packet_app.py`: default tracker path resolves to sibling `Jeweltracker`.
- `Jeweltracker/packet_stage.py`: default packet path resolves to sibling `Packetsealing_001`; launcher supplies `JEWEL_PACKET_DIR` too.
- Current/historical tracker camera defaults: moved connection data into ignored local JSON; sanitized source defaults and credential-bearing documentation examples. Private overlays retain the original camera behavior.
- `jewel_tracker_5.py`: optional recordings reviewer accepts `EMBSYS_REVIEWER_DIR`, preserving its existing UI and arguments.
- Copied engine shell launchers open the combined launcher; original shell launchers remain untouched.
- Three image-based Jeweltracker regression test files use `regression_assets.assets_root()` to read their original historical images through `EMBSYS_REGRESSION_ASSETS`. This avoids copying generated evidence and keeps assertions/model behavior unchanged. Validation reads these originals without writing to them. Regression subprocesses omit the runtime reviewer override so temporary-directory reviewer tests remain isolated.

No SOP thresholds, triggers, detection/inference algorithms, recording/export rules or operator workflow steps were changed. The new launcher opens the existing windows.

Known remaining concerns:

- The normal purity environment lacks some OpenCV features present in SLIC (including HOGDescriptor and ximgproc). Preserve the environment; use SLIC for packet sealing and the optional in-window handoff. Environment-related failures are reported rather than fixed by changing model packages.
- Two source regression assertions also fail against an isolated copy of the untouched original code: classification crop frame identity and optional mouth-lab packet filter boundary behavior. They are documented without changing production logic or weakening assertions.
- Live camera/ROI operation, full operator sequences, real evidence completion and physical camera release require bench validation. Camera-free UI/switching checks cannot establish these results.
- The original external Python environments, cached PaddleOCR models, local FFmpeg executable and optional recordings reviewer remain required local dependencies. A fresh Git clone needs separately provisioned models/settings/environments.
- Legacy Hailo, older standalone applications and training utilities retain their extra dependencies; they are not part of the two production launcher paths.
- Previously embedded camera passwords need rotation; originals and any published history still retain them. See `SECURITY.md`.

## 0.2.0 — 2026-11-20: non-blocking surveillance handoff

Removed evidence-confirmation, active-cycle, per-jewel-result, gallery-readiness and fresh-detection gates from the existing PacketStage transition. Retained reset/lifecycle and actual camera/thread release barriers. Pending recorded-video recovery and PDF processing may continue while packet surveillance runs. Auxiliary model users drain before cache teardown.

Added surveillance-only evidence labels and independent purity/packet outcome fields. Preserved historical IDs, copied mutable track state, retained uncertainty, disabled handoff batch reset, and supported raw recorded surveillance with zero assigned IDs or an unavailable purity gold model. No detector, classifier, rubbing, acid, pickup or sealing thresholds were changed. No physical approval, packet-release or hardware-control features were added.

Moved packet output outside the purity retention directory. Added regression coverage for absent rubbing/masks, missed acid, invalid predictions, pending recovery, gallery-not-ready state, unknown/stale identities, camera release, auxiliary job lifetime, independent export statuses and recording without assigned IDs. Updated the old handoff gate tests to reflect surveillance requirements. Both launcher paths now select the existing SLIC environment without installing/upgrading packages.

## Verified unwanted-file cleanup — 2026-11-20

Performed the reference-traced cleanup described in CLEANUP_REPORT.md: removed the unreferenced duplicate tracker version 3, the unused historical patch, five exact duplicate packet-snapshot assets and bytecode caches. Preserved camera configurations and canonical models. Relocated old tracker entry points, the complete historical tracker update with tests, a pickup snapshot, unreferenced candidate models, benchmark/editor artifacts and local sample images into development/. Preserved saved session evidence. Moved telemetry/packet output into ignored runtime/ with compatible links; restored physical purity capture storage after confirming the unchanged archive containment guard rejects symlink parents. No archived session content was discarded.

Added guarded runtime setup, an explicit development runner, cleanup integrity checks and relocation-aware model/source validation. Expanded validation to run all available historical tests and deterministic pickup self-tests as well as the current production regression suites. Git tracked-file review found zero tracked files; strengthened ignore rules now include joblib/serialized models and runtime/fixture paths. No production SOP code, thresholds, timing, handoff eligibility or identity semantics changed during cleanup. No commit or push.
