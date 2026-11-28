# EMBSYS SOP

The persistent three-engine application starts with `python3 main.py --mode production --hardware virtual`. Open **http://127.0.0.1:8765** for the queue and run `python3 ui/virtual_panel.py` separately for START/STOP/RESET.

Read the [production installation guide](docs/PRODUCTION_INSTALLATION.md), [architecture](docs/PRODUCTION_ARCHITECTURE.md), [engine integration report](docs/PRODUCTION_ENGINE_INTEGRATION.md), [dependencies](docs/PRODUCTION_DEPENDENCIES.md), [migration report](docs/PRODUCTION_MIGRATION_CLEANUP.md), [test results](docs/PRODUCTION_TEST_RESULTS.md) and [remaining physical validation](docs/PRODUCTION_KNOWN_LIMITATIONS.md).

Recorded-input measurements and pending physical tests are listed in [the acceptance matrix](docs/PRODUCTION_ACCEPTANCE_MATRIX.md). The installed service starts independently of the desktop; use the installation guide before starting a manual controller.

The sections below describe the preserved legacy launcher and earlier preparation phase. Production execution uses the persistent controller above.

Packet Releasing is now preserved under `engines/PacketReleasing/` for independent
development. The legacy launcher keeps it disabled; the validated new controller
enables all three adapters. See the
[three-engine preparation guide](docs/THREE_ENGINE_PREPARATION.md) and
[integration validation](docs/PACKET_RELEASING_VALIDATION.md).

```bash
./development/run_engine.sh purity
./development/run_engine.sh packet
./development/run_engine.sh releasing --camera 0
```

Run one engine at a time using these protected commands. Each opens its existing
PyQt interface under the same SOP camera-ownership lock. The purity-to-sealing
in-window handoff remains unchanged. Engine 3 uses its existing separate Python
environment; no detection or verification rules were changed during integration.

A PyQt5 desktop launcher for the existing purity testing and packet sealing applications. Both engines were copied from this PC's local working folders, including uncommitted changes and local models. The original folders are the preserved backups.

From this directory:

```bash
./scripts/install.sh
./run.sh
```

The installer discovers the existing Jeweltracker `.venv`, Jeweltracker `.venv-slic`, and packet sealing `.venv-ocr`. It writes interpreter locations into ignored `configs/local.toml` and inventories the installed versions in `configs/*-environment.txt`. It installs or upgrades **zero packages**. Existing environments remain necessary; keep the original directories in place. For a different installation, set `PURITY_PYTHON`, `PACKET_PYTHON`, and `PACKET_OCR_PYTHON` before running the installer. For headless setup use `QT_QPA_PLATFORM=offscreen ./scripts/install.sh`.

Click **PURITY TESTING** or **PACKET SEALING**. The existing engine window opens. Both start buttons remain disabled until it exits. Close the engine using its existing controls and allow evidence export to finish. Packet sealing may ask you to close its window again after saving. The launcher refuses to close while its workflow is active. No process is killed to switch workflows.

Both workflows default to the existing Jeweltracker SLIC environment so the in-window handoff has its required OpenCV modules. No package versions are changed. The original normal environment remains available through `PURITY_PYTHON` for purity-only operation; its missing OpenCV features prevent the integrated packet handoff.

In **Purity Testing**, connect the camera and let the detected stone settle clearly for half a second. The orange **STONE GRIP AREA** extends the stone box by **20% on each side**. Adjust **Stone grip margin** to cover corner grips and **Pinch distance** for wider bangle grips. Settings are saved locally. The area holds its last reliable position during occlusion and stays fixed from the pinch attempt through the jewel cycle. Margin changes take effect while idle. Wait for **Hand model: ready** before the first pickup.

A pickup candidate and a pinch inside this area start recording after half a second, with up to four seconds of pre-roll. A covered home can start a provisional recording. The standalone's **0.45-second recent-pinch tolerance** bridges brief missed hand detections. A raw gold mask confirming that a jewel remains at its home blocks that jewel's trigger. Reliable identities are retained, and uncertain pickups are saved as **UNASSIGNED** with provisional metadata. Stone-box movement alone cannot start a clip. The rubbed spot and acid still go through their usual checks.

The preview shows hand landmarks, the measured thumb–index gap, and progress toward the 0.5-second pickup trigger once the stone area is ready. **Pinch distance** defaults to **460 px** for the current bench trial. Green highlights indicate a grip within the distance limit and stone area. The message explains when jewel pickup is still pending.

Choose **Trigger hand** as **Any**, **Right**, or **Left** using the model labels shown beside the hands. The other hand stays visible with an ignored label when a specific hand is selected. Any mode preserves uncertain jewel attribution when two hands qualify together. The preview also shows whether each jewel's learned home appears gold, empty, or covered. Hand observations continue on fresh camera frames between the slower gold checks during rubbing.

The purity pickup hand model uses **0.25 detection and presence confidence thresholds**. Pinch distance, pickup evidence and hold timing remain separate trigger checks; lower confidence can also admit false hand detections.

A provisional pickup with exactly one candidate is shown under that jewel as **PROVISIONAL**. Double-click its table status or result to open the saved recording folder. Saved manifests retain an unassigned confirmed identity alongside the candidate ID; reports label the association as provisional. Multiple candidates stay unassigned until later evidence resolves ownership. Candidate evidence does not mark a jewel DONE or supply a confirmed per-jewel prediction to packet sealing. A confirmed test takes precedence in the jewel's table row.

Pickup ownership now retains the standalone recorder's departure, arrival and return evidence for each grip episode. A uniquely confirmed home departure (with the other eligible jewels still at home), or confidently recognized raw gold inside the stone area, can resolve a provisional pickup while its test is active. The original recording, rubbing evidence and acid reaction clock continue in the same cycle; ownership confirmation never starts another test. Confirmation records its reason and source observations in the manifest. A returned jewel can establish a new home after 0.8 seconds of clear, stable placement outside the work area. Multiple qualifying hands or conflicting arrivals remain unassigned. Closed cycles are not reassigned by later pickups.

Hand checks use current camera frames even when the last accepted gold observation is delayed. Gold masks older than 0.5 seconds cannot assert home occupancy or a gold arrival; fresh pixels at the learned home still support pickup checks. Ownership recovered during an acid reaction must be supported by observations from before its onset; a later arrival cannot inherit an earlier reaction. Unresolved identity crops are retried at most once every 0.4 seconds instead of caching an ambiguous result indefinitely. Confirmed stationary identities keep their cache.

PDF summaries include the candidate jewel's photo with the provisional label. New compressed clips use **AVI with MPEG-4/Xvid video**, avoiding the H.264-in-AVI stream rejected by this Ubuntu video pipeline. Original recorder clips remain Motion-JPEG AVI. `scripts/export_ubuntu_review.py SOURCE_SESSION NEW_OUTPUT_FOLDER` makes a separate review copy of an existing session with photos, linked AVI clips and preserved full jewel reports; it does not change the original evidence.

The stone placement check uses its **centre inside the test ROI**; its four corners may extend outside. This check documents verified jewel return. Whole-stone clearance and jewellery-box overlap are advisory logs, not gates for rubbed-mark inference or acid baseline preparation. Those paths use available stone geometry and checks on their own regions. The new-mark verification separately checks that the mark itself lies inside the stone and test ROI.

Acid confirmation and reaction visibility check only the detected acid ROI plus a **10% margin on each side**. Hands, cloth or glare elsewhere on the stone do not veto an otherwise visible acid patch when its position remains trackable. Two distinct clear patch frames confirm the application using the first qualifying frame's timestamp. The five-second reaction clock and recording continue through ROI interference; live and recorded checks report obstructed frames separately. Actual ROI obstructions or unavailable geometry can still leave the reaction incomplete and prevent a material prediction.

After an initial stone outline is established, rubbed-mark inference, acid confirmation, baseline capture and reaction-frame alignment reuse those coordinates on a stationary bed. Outline age alone does not block them. Detector bounds that overlap the established bounds by less than 65% IoU require coordinate recovery. The stone tracker can recover a displaced outline after three distinct detector-supported observations spanning at least **0.2 seconds**; recovery does not validate the old mark reference. When the stone detector is unavailable, continued alignment assumes the stone has stayed in place. Acid monitoring continues independently of missing rubbed-mask evidence. Jewel placement captures the baseline without whole-bed clearance or local-visibility gates; obstruction inside the actual acid ROI is still reported during comparison and classification. Each recording clears the previous cycle's mark-detection message.

The verified rubbed-mask source frame is the preferred acid comparison baseline, using that same unannotated image and its capture timestamp when it precedes the first acid candidate. Stationary bright rubbed streaks are baseline texture, not applicator rejections. If the saved baseline has skin or broad glare inside the actual acid ROI, a bounded cache of earlier pre-acid frames can replace it; post-candidate frames are never eligible. Exact rejection reasons, the selected source and recovery history are recorded. Current hand/applicator obstruction and patch persistence are still checked independently.

**Stone-area gold fallback** is off by default. If enabled, a fresh assigned gold detection inside the area can trigger a pickup candidate after the selected hand has been absent from the area for one second. An ignored hand does not block this fallback. A hand-model failure is shown explicitly and can use the enabled fallback. The preview distinguishes **PINCH HOLD** and **GOLD HOLD**, and saved trigger metadata records the source and any hand-model error. An unconfirmed pickup stays provisional.

To continue the **same batch**, use **Packet Sealing + OCR inside the purity window**. Missing rubbing, masks, acid detections, invalid classification, pending recovery and incomplete tests do not block that transition. The handoff stops and joins the purity camera before packet acquisition starts; recorded-video analysis and PDF work can continue in the background. The root launcher's separate PACKET SEALING button starts an independent session and remains disabled while another engine owns the camera.

Known J1/J2/J3 labels are retained without renumbering. Uncertain IDs remain uncertain and cannot own a named transfer. With no reliable IDs, packet surveillance records from the first frame with an explicit UNASSIGNED status. Reports contain separate purity and packet evidence statuses; completing packet sealing never passes an incomplete purity test. This application monitors evidence and does not approve or control a physical gold-loan transaction.

The separate OCR environment is preserved through `PACKET_OCR_PYTHON`. No internet is required to open the launcher or engines; OCR requires its existing local Paddle model cache. Startup never installs packages.

- Purity engine: `engines/Jeweltracker/jewel_tracker_5.py`
- Packet engine: `engines/Packetsealing_001/packet_app.py`
- Local cameras/ROI/settings: copied JSON files beside each engine, ignored by Git.
- Safe examples: `configs/*.example.json`. Copy examples only on a new installation; do not overwrite the migrated camera/ROI configuration.
- Local models: original engine-relative locations, ignored by Git. A clone alone cannot run inference; provision the local model assets separately.
- Purity evidence stays in the real, ignored `engines/Jeweltracker/captures/` directory because archive integrity rejects symlink parents. Telemetry and packet output live under ignored `runtime/`, with compatible engine-relative links. All existing saved evidence remains intact. Separate packet storage prevents purity archival from pruning an active packet session.
- Optional recordings reviewer: uses the existing external `puritytesting/acid_temporal_lab`, configured locally. Other standalone/legacy tools retain their original extra environment requirements.

Run validation:

```bash
python3 scripts/validate.py
python3 scripts/validate.py --quick
```

See the [verified per-file cleanup report](docs/CLEANUP_REPORT.md), [development tools](development/README.md), [surveillance handoff contract](docs/SURVEILLANCE_HANDOFF.md), [validation results](docs/VALIDATION.md), [architecture](docs/ARCHITECTURE.md), [cleanup/dependency report](docs/DEPENDENCY_REPORT.md), [security review](docs/SECURITY.md), and [change record](docs/CHANGELOG.md), and [directory tree](docs/TREE.md). Raw test logs are local under `outputs/validation/` and are excluded from Git. `scripts/cleanup.sh` removes only Python/test caches, preserving recordings and models.

The launcher locks managed workflows across launcher instances. Do not run either original backup application, or engine Python entry points directly, alongside this launcher; those external launches do not participate in its ownership lock. The copied legacy shell launchers now open the main launcher. Bench hardware validation is required before production use; consult the report for passed, failed and untested checks. No remote was configured and nothing was pushed.
