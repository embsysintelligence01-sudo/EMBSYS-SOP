# Post-Cutting Packet Inspection & Jewellery Counting — Architecture, Integration & Model Report

## 1. Existing Workspace Audit

### Application Entry Points
- **PyQt6 Desktop Application**: `app_pyqt6.py` (`app-pyqt6.py`) → `packet_cut/desktop/app.py` → `packet_cut/desktop/widgets.py` (`MainWindow`)
- **CLI Application**: `app_cli.py` (`app-cli.py`) → `packet_cut/cli/workflow.py`

### Existing Camera Management
- In `packet_cut/desktop/worker.py` (`PacketWorker.run()`, lines 647–661), a single `cv2.VideoCapture` handle (`CAP_DSHOW` fallback on Windows, 640×480, `CAP_PROP_BUFFERSIZE=1`) is opened and owned for the lifetime of the worker thread.
- Frames are emitted to the UI via `frame_ready = pyqtSignal(QImage)` and cached in `self.last_frame`.

### Existing State Machine & Cut Confirmation Signal
In `packet_cut/desktop/worker.py`:
1. `01 FRONT SIDE · PACKET NUMBER` (`_front_ocr_stage`) → populates `self.packet_number`
2. `02 BACK SIDE · TOPSTRIP INSPECTION` (`_back_and_topstrip_stage`)
3. `03 BASELINE · INTACT PACKET MASK` (`_capture_baseline`)
4. `04 CUTTING · HAND / SCISSORS` (`_cut_activity_and_clear`)
5. `05 POST-CUT · SECOND SEGMENTATION` (`_postcut_verify` → `check_final_cut`)
6. `06 FINAL DECISION` → when `final_state == "CUT_CONFIRMED"` (line 674), currently runs `self._track_jewels(cap)` and then emits `self.completed.emit(final_state, final_reason)`.

**Safest Integration Point**: Right after `final_state == "CUT_CONFIRMED"` at `packet_cut/desktop/worker.py` line 674 — passing the **already-open `cap`** (or a `frame_provider` callback), **already-loaded `self.models` dictionary**, and **`self.packet_number`** into `PostCutInspectionPanel` / `InspectionController` so no duplicate camera stream or duplicate model load occurs.

---

## 2. Model Inspection & Audit Report

Inspected directly with ONNX Runtime (`['AzureExecutionProvider', 'CPUExecutionProvider']`) and Ultralytics in `models/`:

| Model File | Size | Task & Input | Classes | Status in New Module |
|---|---:|---|---|---|
| `models/packet-seg.onnx` | 47.22 MB | YOLOv8s-seg (`1×3×320×320`) | `{0: 'bag'}` | **REUSED** — Stage 1 packet ROI localization; Stage 2 packet-in-testbed exclusion |
| `models/jeweltrack.onnx` | 47.22 MB | YOLOv8s-seg (`1×3×320×320`) | `{0: 'gold'}` | **REUSED** — Stage 1 gold visibility inside packet interior; Stage 2 jewellery segmentation on testbed |
| `models/hand-seg.onnx` | 41.60 MB | YOLO26s-seg (`1×3×320×320`) | `{0: 'handseg - vdataset handseg-9w6ds'}` | **REUSED** — Stage 1 hand exclusion & occlusion detection; Stage 2 placement stability pause |
| `models/strip-seg.onnx` | 47.22 MB | YOLOv8s-seg (`1×3×320×320`) | `{0: 'TopStrip'}` | **EXCLUDED** — Never loaded or used in Stage 1 per requirements |
| `models/cut-seg.pt` | 6.11 MB | YOLOv8-seg | `{0: 'Cut', 1: 'Packet'}` | Existing cutting workflow only (untouched) |
| `models/scissor-seg.pt` | 6.11 MB | YOLOv8-seg | `{0: 'scissor'}` | Existing cutting workflow only (untouched) |
| `models/packet-no-obb.onnx` | 45.80 MB | YOLOv8s-obb (`1×3×640×640`) | `{0: 'packet-ocr'}` | Existing OCR workflow only (untouched) |
| `sam2.1_hiera_tiny.pt` / `mobile_sam.pt` | — | Optional SAM prompt segmentation | — | **MISSING from `models/`** — Optional hook implemented in `InstanceSeparator`; falls back gracefully to deterministic OpenCV watershed + overlap uncertainty detection |

### Known Limitations & Real-Data Observations
1. **No vertical top-down open-packet dataset in workspace**: `packet_cut_outputs/` contains flat overhead intact/cut packets (`baseline_frame.jpg`, `verification_debug/*_orig.jpg`) and one real gold cluster frame (`sample_1794906516686_sample_1_orig.jpg`). Live calibration of Stage 1 thresholds (`border_inset_ratio`, `min_interior_area_px`, `jewel_conf`) should be validated on live vertically-held packet video from the overhead station camera.
2. **Clustered / entangled jewellery**: On `sample_1794906516686_sample_1_orig.jpg`, `jeweltrack.onnx` detects the gold cluster with `conf=0.885` as a single merged segmentation mask. Touching items with a clean tangent neck (including figure-8 touching bangles) are split via marker-controlled watershed; heavily overlapping or tangled piles are intentionally flagged `COUNT_UNCERTAIN` rather than guessing.
3. **Optional SAM2.1 / MobileSAM**: Because neither `sam2.1_hiera_tiny.pt` nor `mobile_sam.pt` is present in `models/`, prompt-based SAM separation is inactive until a weight path is added to `inspection/config.json` (`models.sam_model`) or `INSPECTION_SAM_MODEL`.
4. **Execution Providers**: ONNX Runtime on this workstation exposes `CPUExecutionProvider` (and `AzureExecutionProvider`). Inference runs on CPU inside a dedicated `QThread` so the PyQt6 GUI remains responsive.

---

## 3. New Module Structure

```text
inspection/
├── __init__.py
├── config.json                        # Configurable model paths, ROIs, thresholds
├── post_cut_inspection.py             # GUI-independent PostCutInspectionEngine
├── inspection_controller.py           # PyQt6 QThread worker & signal/slot controller
├── inspection_state_machine.py        # Full 18-state post-CUT_CONFIRMED state machine
├── inspection_result.py               # Structured outcome dataclasses & JSON export
├── ui_panel.py                        # Independent PyQt6 PostCutInspectionPanel + standalone app
├── INTEGRATION_GUIDE.md               # This document
├── packet_inspection/
│   ├── __init__.py
│   ├── packet_roi.py                  # Packet localization, hand occlusion, rotation, lighting/blur checks
│   ├── interior_extractor.py          # Border/hand/printed-label exclusion -> visible interior depth mask
│   ├── gold_visibility.py             # jeweltrack.onnx + HSV/LAB glare & metallic texture verification
│   └── packet_status.py               # Temporal confirmation -> OPEN / CLOSED / UNKNOWN
├── jewellery_counting/
│   ├── __init__.py
│   ├── testbed_roi.py                 # Configurable fixed testbed ROI + green ROI calibration + empty check
│   ├── jewel_detector.py              # Testbed-only jeweltrack.onnx inference (excludes packet interiors)
│   ├── instance_separator.py          # Optional SAM + OpenCV watershed/hole-fill separation + COUNT_UNCERTAIN
│   ├── object_tracker.py              # Mask/Box IoU + Hungarian unique ID tracker (handles removal/re-placement)
│   └── count_manager.py               # Placement stability + appraiser confirmation gate
└── utils/
    ├── __init__.py
    ├── image_utils.py                 # Letterbox unpadding, YOLO mask extraction, quality & overlay helpers
    ├── inspection_config.py           # Dataclass config loader/saver + env overrides
    ├── model_loader.py                # Shared model registry (reuses host models, excludes TopStrip)
    └── temporal_verification.py       # Consecutive-frame hysteresis verifier
```

---

## 4. Exact Non-Destructive Integration Recipe (Connecting to `CUT_CONFIRMED`)

The existing Packet-Cut application has **not** been modified. When ready to connect the new module to `CUT_CONFIRMED`, make only these minimal additive hooks:

### Step A — Emit a `cut_confirmed` signal with shared resources in `packet_cut/desktop/worker.py`

1. Add a signal to `PacketWorker`:
```python
cut_confirmed = pyqtSignal(str, dict)  # (packet_id, shared_models)
```

2. Right after `final_state == "CUT_CONFIRMED"` (replacing the direct `_track_jewels` call only if desired, or running the new panel after releasing/pausing the capture loop):
```python
if final_state == "CUT_CONFIRMED":
    # Pass already-loaded models (excluding TopStrip) so nothing is reloaded.
    shared_models = {
        "packet": self.models.get("packet"),
        "jewel": self.models.get("jewel"),
        "hand": self.models.get("hand"),
    }
    # Option 1 (recommended): release `cap` cleanly before handing camera to PostCutInspectionPanel,
    # OR Option 2: pass `existing_capture=cap` directly into `InspectionController.start_on_cut_confirmed(...)`
    # while `PacketWorker` waits for completion so only ONE handle reads the camera at a time.
    self.cut_confirmed.emit(self.packet_number, shared_models)
```

### Step B — Open or switch to `PostCutInspectionPanel` in `packet_cut/desktop/widgets.py`

```python
from inspection.ui_panel import PostCutInspectionPanel

# Inside MainWindow._connect_worker():
self.worker.cut_confirmed.connect(self._launch_post_cut_inspection)

# Slot on MainWindow:
def _launch_post_cut_inspection(self, packet_id: str, shared_models: dict):
    self.post_cut_panel = PostCutInspectionPanel(shared_models=shared_models)
    self.post_cut_panel.inspection_finished.connect(self._on_post_cut_finished)
    self.post_cut_panel.show()
    self.post_cut_panel.start_inspection(
        packet_id=packet_id,
        cut_status="CUTTED",
        camera_source=self.worker.camera_id,  # Or pass existing_capture / frame_provider
        shared_models=shared_models,
    )

def _on_post_cut_finished(self, result_dict: dict):
    # Structured result matches Section 8 schema:
    # result_dict["packet_id"], result_dict["packet_inspection"]["status"],
    # result_dict["jewellery_counting"]["count"], result_dict["inspection_status"]
    print("Post-Cut Inspection Result:", result_dict)
```

### Standalone Execution (Independent of Existing App)
```bash
python -m inspection.ui_panel --camera 0 --packet-id PKT-1025
# Or with a recorded video:
python -m inspection.ui_panel --camera path/to/video.mp4 --packet-id PKT-1025
```

---

## 5. Configuration & Dependencies

### Configuration (`inspection/config.json`)
All model paths, execution providers, Stage 1 thresholds, and Stage 2 testbed ROI coordinates are configurable in `inspection/config.json` or via environment variables:
- `INSPECTION_PACKET_MODEL`, `INSPECTION_JEWEL_MODEL`, `INSPECTION_HAND_MODEL`, `INSPECTION_SAM_MODEL`
- `INSPECTION_DEVICE` (`cpu` or CUDA index)
- `INSPECTION_TESTBED_ROI` (e.g., `160,120,480,400`)

### Dependency Changes
- **Required**: No new mandatory dependencies beyond the existing `requirements.txt` (`ultralytics`, `opencv-python`, `numpy`, `torch`, `onnxruntime`, `PyQt6`).
- **Optional (already present in `.venv`)**: `scipy` is used for optimal Hungarian track matching (`scipy.optimize.linear_sum_assignment`), with a built-in deterministic greedy fallback when `scipy` is absent.

