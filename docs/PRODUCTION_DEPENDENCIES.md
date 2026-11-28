# Standalone dependency report

All production interpreters are local to this checkout:

- `runtime/environments/controller`: Python 3.10, FastAPI 0.162.5, Uvicorn 0.54.0,
  psutil 7.2.2, HTTPX 0.29.0, ReportLab 5.1.0 and pypdf 7.2.0; no Qt, Torch or camera acquisition in the API.
- `runtime/environments/purity`: copied working SLIC environment, used by purity,
  sealing and the virtual panel. Existing PyQt5, OpenVINO, ONNX Runtime,
  MediaPipe, Torch and original inference dependencies are unchanged.
- `runtime/environments/ocr`: copied working isolated sealing OCR environment.
- `runtime/environments/releasing`: copied working Python 3.12 release environment,
  PyQt6, CPU Torch, ONNX Runtime, PaddleOCR and ReportLab.
- `runtime/python/`: copied CPython 3.12 base required by the release environment.
- `runtime/model-cache/paddlex`: copied cached OCR assets. Startup does not install
  packages or require a model download on this provisioned PC.

`configs/local.toml` points at these local environments. Model binaries remain
ignored. `models/shared`, `models/purity`, `models/sealing`, `models/releasing` are
registered with hashes in `configs/model_registry.json`. Release's original
`models/` path is a compatibility link, preserving static and dynamic resolution.
Its original model directory is retained under `runtime/backups/releasing-models`.

`deployment/provision_local.py` copies installed environments and repairs their
Python/shebang paths without upgrading engine packages. `requirements-linux.lock`
for releasing and environment inventories are preserved. The backend requirements
are pinned separately in `deployment/requirements-controller.txt`. Existing
working engine installations are the offline provisioning source on this PC;
another PC needs those environment assets or an appropriately supplied wheelhouse.
This is not a universal self-contained executable or a new-platform package build.

OS prerequisites on this Ubuntu PC include Python 3.10, standard C/C++ runtime,
OpenGL/EGL/GLib libraries for OpenCV, FFmpeg, camera access and the Qt desktop
libraries for optional panel/debug clients. The production engine hosts do not
require X11, Wayland, a monitor or a desktop login. The user account must retain
its camera permissions. System libraries and USB device drivers are not vendored.

Purity/sealing's pre-existing `pip check` distribution-name mismatch remains:
Ultralytics asks for `opencv-python` while the working environment supplies
`opencv-contrib-python`. Runtime imports, models and regressions are tested.
Installing an overlapping wheel would replace the contrib build; it was not
done. The release environment also retains its original installed OpenCV
packages. No source-engine dependency version was upgraded to silence metadata.

The optional legacy external recordings reviewer remains a development-only
configuration, not a production dependency. Its source installation is retained.
No release MobileNet classifier was added: that model has not been provided.
No serial driver/protocol is assumed for uncommissioned physical hardware.
