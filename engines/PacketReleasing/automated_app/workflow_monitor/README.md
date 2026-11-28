# EMBSYS Packet-Cut — Automated Appraiser Workflow Monitoring, Compliance & Reporting

Click **START** once. The system automatically observes, processes, monitors compliance, generates real-time alerts, logs every action, and produces the final inspection & compliance report without manual stage progression.

## Two Independent Responsibilities

1. **Automated Packet Processing** (`workflow_monitor/automated_engine.py`):
   - Automatic packet detection & tracking (`models/packet-seg.onnx`)
   - Continuous OCR protection (`models/packet-no-obb.onnx` + `packet_cut/desktop/ocr.py` PaddleOCR) — locks the first reliable ID, preserves identity across all stages, supports post-alert recovery, and retains `UNRESOLVED` if unreadable (never assigns a wrong/stale ID)
   - Automatic pre-cut orientation & seal check (`models/strip-seg.onnx` — pre-cut only, strictly excluded from post-cut Stage 1) and intact baseline capture
   - Automatic cutting activity detection (`models/hand-seg.onnx`, `models/scissor-seg.pt`) and physical separation verification (`models/cut-seg.pt` + `packet_cut_core.py`)
   - Automatic post-cut OPEN/CLOSED packet inspection (`inspection/packet_inspection/` — gold visibility inside packet, never TopStrip)
   - Automatic green testbed empty check, jewellery detection, touching-item separation, overlap uncertainty detection, unique ID tracking, and auto-completion (`inspection/jewellery_counting/`)

2. **Appraiser Procedure Compliance Monitoring** (`workflow_monitor/procedure_monitor.py`, `alerts.py`, `activity_log.py`, `report_generator.py`):
   - Evaluates observed appraiser behavior against a 9-step prescribed SOP
   - Detects `COMPLIANT`, `SKIPPED`, `OUT_OF_ORDER`, `DEVIATION`, `RECOVERED`, and `UNVERIFIED` actions without halting packet processing
   - Emits deduplicated, timestamped real-time alerts linked to the session and packet ID, and tracks when the appraiser corrects each issue
   - Saves chronological activity logs with supporting evidence frames and automatically generates JSON + Markdown reports under `packet_cut_outputs/automated_sessions/<session_id>/`

## Launch Commands

```powershell
# Live USB camera (single START button)
.\.venv\Scripts\python.exe app_automated.py --camera 0

# Recorded video file
.\.venv\Scripts\python.exe app_automated.py --camera "path\to\video.mp4"

# Run automated workflow monitor tests
.\.venv\Scripts\python.exe -m unittest tests.test_automated_workflow_monitor -v
```

