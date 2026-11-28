# EMBSYS Packet-Cut System

Computer vision inspection and compliance platform for gold loan packet appraisal.

## Repository Architecture

```
D:\Project\Embsys\Packet-cut-new\
├── models\                          # Shared CV models (ONNX & PyTorch weights)
├── .venv\                           # Shared Python virtual environment
│
├── automated_app\                   # Fully automated, event-driven desktop system
│   ├── app.py                       # Main PyQt6 inspection app
│   ├── app_compliance.py            # Standalone SOP & Compliance audit runner
│   ├── main.py                      # Default launcher
│   ├── packet_cut\                  # Event-driven state machine, camera worker, widgets
│   ├── inspection\                  # Post-cut packet mouth & jewellery counting module
│   ├── workflow_monitor\            # Real-time alerts, procedure compliance & reports
│   ├── packet_cut_outputs\          # Audit sessions, baseline images & testbed ROI
│   ├── tests\                       # Automated test suites
│   ├── packet_cut_core.py           # Core CV engine
│   └── README.md                    # Automated application documentation
│
└── manual_app\                      # Operator-guided manual desktop pipeline
    ├── app_pyqt6.py                 # Multi-button PyQt6 GUI (CONFIRM OCR, BACKSIDE READY, RETRY)
    ├── app_cli.py                   # Keyboard-driven OpenCV CLI dashboard
    ├── main.py                      # Default launcher
    ├── packet_cut_outputs\          # Sample baseline images & testbed ROI
    ├── packet_cut_core.py           # Core CV engine
    └── README.md                    # Manual application documentation
```

## Quick Start

### 1. Automated Application
```powershell
.\.venv\Scripts\python.exe automated_app/app.py
```

Click **Load Models / Check Ready**, then **Draw ROI** to select the testbed on the camera preview. Click **START** to begin. Use **Retry Current Stage** to recheck the active stage or **STOP** to save the session. **Copy Compliance Report** copies the generated Markdown report.

### 2. Standalone Compliance & Audit Monitor
```powershell
.\.venv\Scripts\python.exe automated_app/app_compliance.py
```

### 3. Manual Application (Multi-Button Controls)
```powershell
.\.venv\Scripts\python.exe manual_app/app_pyqt6.py
```

### 4. Run Test Suite
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s automated_app/tests -p "test_*.py"
```
