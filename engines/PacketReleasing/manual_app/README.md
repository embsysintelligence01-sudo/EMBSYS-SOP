# EMBSYS Packet-Cut — Manual Application

Operator-guided applications for secure packet cutting verification, featuring manual controls and step-by-step confirmation.

## Applications Included

### 1. Manual PyQt6 Desktop Application (`app_pyqt6.py`)
Provides an interactive multi-button GUI where the operator manually controls stage progression:
- **`CONFIRM OCR`**: Manually verify and accept the scanned packet identifier.
- **`BACKSIDE READY`**: Confirm that the packet has been flipped to the seal side.
- **`RETRY`**: Reset and re-execute the active stage.
- **`NEXT STAGE`**: Advance to the next verification step.
- Visual metrics cards for Hand detection, Scissor presence, Background gap (px), and Piece count.

```powershell
# From project root:
.\.venv\Scripts\python.exe manual_app/app_pyqt6.py

# With custom camera index:
.\.venv\Scripts\python.exe manual_app/app_pyqt6.py --camera 0
```

### 2. Manual CLI Workflow (`app_cli.py`)
A keyboard-driven OpenCV dashboard interface:
- Visual overlays rendered via OpenCV HUD.
- Interactive keyboard shortcuts:
  - `SPACE` / `ENTER`: Proceed or accept read.
  - `R`: Retry active detection.
  - `Q` / `ESC`: Exit.

```powershell
# From project root:
.\.venv\Scripts\python.exe manual_app/app_cli.py

# Headless / CPU mode:
.\.venv\Scripts\python.exe manual_app/app_cli.py --camera 0 --cpu
```

## Shared Resources
Both applications use the shared computer vision models located at `../models/` (`packet-seg.onnx`, `hand-seg.onnx`, `scissor-seg.pt`, `strip-seg.onnx`, `packet-no-obb.onnx`, and `cut-seg.pt`).

