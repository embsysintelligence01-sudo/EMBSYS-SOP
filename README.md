# EMBSYS SOP

A PyQt5 desktop launcher for the existing purity testing and packet sealing applications. Both engines were copied from this PC's local working folders, including uncommitted changes and local models. The original folders are the preserved backups.

From this directory:

```bash
./scripts/install.sh
./run.sh
```

The installer discovers the existing Jeweltracker `.venv`, Jeweltracker `.venv-slic`, and packet sealing `.venv-ocr`. It writes interpreter locations into ignored `configs/local.toml` and inventories the installed versions in `configs/*-environment.txt`. It installs or upgrades **zero packages**. Existing environments remain necessary; keep the original directories in place. For a different installation, set `PURITY_PYTHON`, `PACKET_PYTHON`, and `PACKET_OCR_PYTHON` before running the installer. For headless setup use `QT_QPA_PLATFORM=offscreen ./scripts/install.sh`.

Click **PURITY TESTING** or **PACKET SEALING**. The existing engine window opens. Both start buttons remain disabled until it exits. Close the engine using its existing controls and allow evidence export to finish. Packet sealing may ask you to close its window again after saving. The launcher refuses to close while its workflow is active. No process is killed to switch workflows.

Both workflows default to the existing Jeweltracker SLIC environment so the in-window handoff has its required OpenCV modules. No package versions are changed. The original normal environment remains available through `PURITY_PYTHON` for purity-only operation; its missing OpenCV features prevent the integrated packet handoff.

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
