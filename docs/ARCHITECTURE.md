# Architecture

`run.sh` resolves the configured purity interpreter and starts `launcher.py`. The launcher imports only its PyQt5 UI and configuration/process management modules; model libraries are imported in engine processes.

```text
run.sh -> launcher.py -> app/main_window.py
                           |
                     ProcessManager
                     /             
        purity Python                SLIC Python
        jewel_tracker_5.py           packet_app.py
        engines/Jeweltracker/        engines/Packetsealing_001/
                     \              /
                       local models
                       camera/ROI JSON
                       evidence exports
```

`app/config.py` reads `configs/app.toml` and the ignored `configs/local.toml`. Paths resolve relative to the combined project, preserving interpreter symlink paths so Python finds the correct virtual environment. Environment variables can override interpreters. The migrated applications keep their existing working directories and engine-relative model paths. `JEWEL_TRACKER_DIR` and `JEWEL_PACKET_DIR` always point to the new engines. OCR and the optional external reviewer receive explicit local locations.

`ProcessManager` launches with `subprocess.Popen`, a Qt polling timer, a new process session and no shell. A nonblocking `flock` permits one managed SOP at a time, including across launcher instances. Its descriptor is inherited by the engine, retaining ownership if the launcher unexpectedly disappears. Any descendant explicitly retaining the descriptor must exit before ownership can transfer. Most existing subprocess calls close inherited descriptors; engine-level close handlers remain responsible for waiting for their own workers and exporting evidence.

While an engine runs, both start buttons are disabled and normal launcher closure is refused. Closing the existing engine invokes its existing close handlers: purity drains workers, pools and model resources; packet sealing finishes/export evidence and may require a second window close after saving. The launcher waits for process exit and lock release, then displays Finished or Error. Unexpected termination cannot guarantee evidence integrity; review the session before continuing. No terminate/kill call exists in the production process manager.

The purity application's existing in-process `PacketStage` handoff remains intact. It shares/drains the existing camera worker and imports the copied packet modules. Both managed launch paths use the existing SLIC environment by default, preserving its installed versions. The normal environment is retained as a purity-only override. This feature is independent of the launcher's two start buttons and does not create a competing managed engine process.

Both original Git repositories and outer backup folders remain untouched. The new root Git repository contains code, safe examples, documentation and tests. Models, configuration with credentials, evidence, recordings, logs and environments are local assets. The new application's environment references are intentionally local dependencies rather than copied virtual environments.

Regression image fixtures are read from the untouched original Jeweltracker folder through `EMBSYS_REGRESSION_ASSETS`; tests keep their assertions and the new production tree does not receive historical evidence.

The process lock covers launches through this combined launcher. Direct invocation of engine scripts and original backup applications is outside its scope. Physical camera behavior and operator sequences must be validated on the bench. Optional Hailo, training tools, Qt6 acid UI, mouth lab and recordings reviewer remain available in source with their original dependencies; installing them is not part of launcher setup.

## Surveillance handoff

Workflow progress and evidence verification are independent. `packet_wait_reason` no longer checks rubbing, mask, acid, prediction, track count, active evidence cycles, gallery readiness or recent detection success. Reset lifecycle ownership remains a gate. `camera_released` requires the purity acquisition/stone/identity threads to have actually stopped before packet acquisition starts; a release event alone cannot override a live acquisition thread.

The existing `PacketStage` stops live acquisition without forcing a process exit. ContinuousMonitor closes the current recorded interval and drains recorded-video recovery, retention and PDF jobs. Manual live acid acquisition is cancelled without waiting in the GUI; already-running rub jobs may drain. Testing model caches remain alive until the old worker and auxiliary consumers have ended. Shared gold inference stays serialized by its existing dispatcher.

A bridge copies track data at handoff rather than sharing mutable tracks with background analysis. Existing IDs and the gallery are retained. Identity confidence is frozen at the camera barrier, preventing model-loading delay from inventing new uncertainty. Initially uncertain identities remain excluded from named pickup/placement ownership. Empty or partial identity sets stay LOCKED in surveillance mode and do not create a new batch. Gold-model failure during handoff degrades to unassigned recording without claiming absence or a match.

Packet recording begins from its first frame, including sessions with no IDs. Its storage lives outside the purity directory so background purity archival cannot prune it. Packet recording/camera shutdown occurs before final combined export waits for the purity completion event. Final purity snapshots are published under a lock; reports receive deep copies, preserving independent evidence outcomes and uncertainty. Unknown/uncertain recordings are retained as raw evidence. Completed manual rub arrays are retained as unconfirmed auxiliary evidence, rather than discarded or promoted to verified results. Purity status is VERIFIED only when its recorded verification fields support it; a prediction or completed packet stage alone never supplies that proof.

## Verified file cleanup

Unreferenced numbered/similarity entry points and historical snapshots are preserved under `development/legacy/`; their bytecode and duplicate version 3/packet snapshot assets were removed only after reference and content checks. Models referenced by optional packaging/training remain. The current packet pickup experiment modules remain at their original paths because tests import them dynamically.

Telemetry, packet output and local sample images were moved outside engine source directories, with compatible links. Purity captures remain at their physical original combined-project path: the unchanged evidence containment policy rejects symlinked parents. Saved evidence is retained and ignored by Git. `scripts/prepare_runtime.py` recreates these links during install and uses the same exclusive lock as ProcessManager; it refuses active ownership or ambiguous merges. No SOP source paths, algorithms or timing were changed.

The development runner uses the configured SLIC environment and ownership lock, and executes preserved source with its original engine file/import context so local model/config lookups still work. These explicit developer tools never run from the production launcher. `CLEANUP_MANIFEST.json` maps old assets to their preserved destinations or byte-identical canonicals; validation/audit understand that mapping without weakening required active-model checks. `CLEANUP_REPORT.md` records every initial engine file, reason and result.
