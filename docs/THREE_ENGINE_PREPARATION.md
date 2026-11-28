# Three-engine preparation

Prepared 2026-11-28. This integrates the existing desktop engines for development.
Packet Releasing is deliberately excluded from production launch selection. There
is no persistent queue or central transaction controller in this phase.

## Engine map and execution

| Engine | Independent launch | Existing entry point | UI/runtime | Evidence |
|---|---|---|---|---|
| 1. Purity Testing | `./development/run_engine.sh purity` | `engines/Jeweltracker/jewel_tracker_5.py` aliases `engines/purity/tracker.py` | Existing SLIC Python, PyQt5; OpenVINO, MediaPipe, ONNX Runtime, Torch | Physical `engines/Jeweltracker/captures/`; existing integrity/retention rules retained |
| 2. Packet Sealing | `./development/run_engine.sh packet` | `engines/Packetsealing_001/packet_app.py` uses `standalone.app` and existing `engines/sealing/` compatibility modules | Existing SLIC Python, PyQt5; separate OCR interpreter through `PACKET_OCR_PYTHON` | Existing engine-relative output link to `runtime/Packetsealing_001/outputs/` |
| 3. Packet Releasing | `./development/run_engine.sh releasing --camera 0` | `engines/PacketReleasing/automated_app/app.py` -> `packet_cut.desktop.app` -> original MainWindow/PacketWorker | Separate Python 3.12 environment, PyQt6; CPU Torch, Ultralytics, ONNX Runtime, PaddleOCR, ReportLab | `engines/PacketReleasing/packet_cut_outputs/automated_sessions/SES-…/`: JSON, Markdown, HTML, PDF, images and clips |

Commands run from the EMBSYS-SOP root. All three open their existing desktop
interfaces. The production `./run.sh` still exposes only purity and sealing.
`command('releasing')` rejects production selection; the developer runner alone
opts into `command('releasing', development=True)`. This is preparation gating,
not a production/debug mode implementation or security boundary.

The root developer runner resolves the interpreter and working directory,
acquires the existing SOP `flock`, checks for already-open `/dev/video*` devices,
and starts the engine as an isolated process. The ownership descriptor is
inherited, so the engine retains it if its parent disappears. The runner waits
for normal exit; it never kills an engine to switch. Production launches and
legacy developer tools use the same lock. The production process manager also
refuses launch when an unmanaged process visibly owns a camera. `/proc` scanning
is best-effort; permissions and future raw-script launches can bypass it.
Supported independent launches must use the provided protected commands.

Raw engine Python scripts, the preserved source launchers, and original backup
applications do not acquire this lock. Do not run them alongside a managed
engine. This phase does not patch their camera acquisition or detection code.
The original copied `run_automated.sh` expects a local `.venv-ubuntu`; the new
`engines/PacketReleasing/run_development.sh` is the supported migrated launcher.

## Preserved source and handoff

`configs/packet_releasing_snapshot.json` records SHA-256 for all 111 copied
working files (source, tests, documentation, requirements, seven models and
current ROI/baseline files and one regression image fixture) and a baseline for 892 existing engine source/script
files. Validation checks these bytes, including the source that was already
modified in EMBSYS-SOP before this integration. Nothing is reset to Git HEAD or
to an older backup. Original engine interfaces and algorithms remain unchanged.
No detection threshold, tracking identity, OCR rule, timing or evidence decision
was altered by this integration. Packet Releasing includes the earlier OCR
improvements as they existed at snapshot time.

Purity's existing in-window PacketStage handoff remains the same process and
batch. It stops/joins its acquisition before starting packet acquisition while
background evidence work drains. IDs, provisional attribution and independent
verification statuses remain governed by `docs/SURVEILLANCE_HANDOFF.md`. Starting
sealing independently still creates its existing independent session. Engine 3
has no automatic handoff, no inferred linkage to a sealing batch, and no physical
release approval. A matching OCR reading alone does not establish custody or
cross-engine identity verification.

## Dependencies and configuration

Interpreter paths are read from `configs/app.toml`, overridden by ignored
`configs/local.toml`, then by `PURITY_PYTHON`, `PACKET_PYTHON` or
`RELEASING_PYTHON`. `PACKET_OCR_PYTHON` remains the sealing OCR override.
The current purity/sealing environment remains the existing Jeweltracker
`.venv-slic`; releasing uses the original packet-cut-system `.venv-ubuntu`.
Environments were not moved, copied or upgraded. Keep those source installations
in place until a separate environment-provisioning phase. A source clone alone
cannot run these engines.

Installed distributions are inventoried in `configs/purity-environment.txt`,
`configs/packet-environment.txt` and `configs/releasing-environment.txt`.
Packet Releasing's original `requirements-linux.lock` is also preserved. Key
release versions: PyQt6 6.12.0, Torch 2.15.0+cpu, Ultralytics 8.4.211,
ONNX Runtime 1.31.1, PaddleOCR 3.7.0/Paddle 3.2.0, NumPy 2.2.6,
ReportLab 5.1.2. Existing purity/sealing use PyQt5 5.15.11, OpenVINO 2026.4.1,
Torch 2.14.1+cpu and ONNX Runtime 1.23.2. Do not merge these environments now.
Purity/sealing `pip check` reports Ultralytics' missing `opencv-python`
distribution name, although their installed `opencv-contrib-python` provides the
working `cv2` module and runtime import/UI/handoff checks pass. This pre-existing
metadata mismatch is recorded as a failed dependency audit; installing another
OpenCV wheel here could replace the contrib build needed by sealing, so no
package change was made in this preservation phase.
The release environment already contains two OpenCV distribution names; no
package removal or reinstall was attempted. Its dependency check passed.

The runner clears inherited Python/Qt plugin paths before launch. The existing
release entry point sets the Qt6 plugin directory to avoid OpenCV/Qt5 collision.
It preserves the original CPU, four-thread and mobile OCR environment defaults.
Paddle models require their existing local cache under
`~/.paddlex/official_models/en_PP-OCRv5_mobile_rec` and
`PP-OCRv5_mobile_det` (fallback). Startup installs no packages; uncached Paddle
assets would require separate provisioning. An unavailable OCR model must not
be treated as a verified ID.

Release workflow/inspection configuration JSON stays beside its modules. The
copied current testbed ROI and baseline assets remain at their original relative
locations. They are bench-specific and ignored by Git. New sessions use their
own new session IDs; copied calibration is not old session evidence. Historical
reports and recordings remain in the original packet-cut-system, not duplicated
into the new engine. Release's optional Windows logo path does not resolve on
Ubuntu; its existing empty-logo warning remains cosmetic and was not rewritten.

## Models and reusable candidates

| Consumer | Required assets/roles |
|---|---|
| Purity | Existing central registry: gold IR XML/BIN, stone model, rubbed-mark models, temporal acid detector/classifier, pH models, hand landmarks and SigLIP identity assets; preserve model-specific preprocessing and class maps |
| Sealing | Existing central registry: gold/identity context, packet, hand, strip, number-region detection and configured OCR; keep current sealing/mouth verification model settings |
| Releasing | Local `packet-seg.onnx`, `hand-seg.onnx`, `strip-seg.onnx`, `packet-no-obb.onnx`, `jeweltrack.onnx`, `scissor-seg.pt`, `cut-seg.pt`, cached mobile OCR; optional SAM weights remain unconfigured |

The existing central registry remains authoritative for purity/sealing. Release
keeps local model files and original model/config resolution. Its complete model
inventory and comparison is in `PACKET_RELEASING_MODEL_COMPARISON.json`.
Packet, hand, strip and number models are byte-identical to their sealing
registry counterparts. `jeweltrack.onnx` equals the registered legacy
`shared.gold.legacy.rojeweltrack.onnx`; this does not mean it is equivalent to the
active OpenVINO gold model. Cut/scissors have no identical registered counterpart.
No file or model object was deduplicated or switched to a different runtime.

Candidates for later common interfaces: latest-frame camera acquisition and
capture timestamps; inference scheduling and model lifecycle; packet/number
cropping and OCR; hand/strip masking; session metadata; image/clip storage and
PDF generation. They need explicit contracts for shapes, coordinates, class IDs,
confidence, timestamps and evidence ownership before reuse. Identical weights do
not guarantee interchangeable preprocessing or verification. Existing tracking,
gap confirmation, mouth rules and purity/sealing evidence code remain separate.
The planned 320-input MobileNet mouth classifier is not present or substituted.

## Validation and next-phase boundary

Run `python3 scripts/validate_releasing.py`. It checks the copy and existing
source baseline, protected launch paths, camera ownership, imports, dependencies,
Qt UI construction, real model loading/inference, release regressions and the
existing purity-to-sealing handoff without opening a camera. Results and
limitations are in `PACKET_RELEASING_VALIDATION.md`; private logs live under
`outputs/validation/packet-releasing/`.

Three historical `tests/test_integration.py` assertions fail against the current
pre-existing working tree: live camera/settings differ from the original backup;
a pre-existing `.venv-slic` directory exists under the legacy pickup standalone;
and migrated pickup code differs from the original pre-central-registry source.
They were retained, not weakened or used to reset current engine files. The new
checksum tests establish preservation against the working tree at integration
start. These older checks need reconciliation in a separately scoped cleanup.

Live camera switching, operator handling, complete evidence export and reopening
saved PDFs still require bench validation. Existing release mouth false-positive
and cut stability limitations remain unchanged. Passing camera-free tests is not
production acceptance. Browser frontend, persistent queue, controller, virtual
hardware panel and execution modes are future work; production enabling needs
explicit acceptance after bench validation and open findings are resolved.

Current focused results: release regressions 92 passed / 1 skipped (unavailable
historical Windows-only sample); new integration/ownership 9 passed; existing
launcher 5 passed; purity-to-sealing handoff 19 passed. Source/model checksum,
release dependency, imports, UI and real vision/OCR inference checks passed.
The historical integration assertions and purity/sealing dependency metadata
audit failures remain visible in the report and prevent a clean all-checks pass.
