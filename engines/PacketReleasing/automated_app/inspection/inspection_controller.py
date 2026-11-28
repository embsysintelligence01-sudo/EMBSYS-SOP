"""PyQt6 background QThread worker and signal/slot controller for Post-Cutting Inspection.

Key guarantees:
- Runs all inference inside a background QThread; never blocks the main GUI thread.
- Reuses an existing camera stream / frame_provider callback when integrated into
  Packet-Cut so no duplicate camera handles are opened.
- Reuses already-loaded models via `shared_models` to avoid duplicate loading.
- Handles missing models and camera failures gracefully via Qt error signals.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QImage

from .inspection_result import PostCutInspectionResult
from .inspection_state_machine import InspectionState
from .post_cut_inspection import PostCutInspectionEngine
from .utils.inspection_config import InspectionConfig


FrameProvider = Callable[[], Tuple[bool, Optional[np.ndarray]]]


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class InspectionWorker(QThread):
    """Background worker executing Stage 1 & Stage 2 inspection frames."""

    frame_ready = pyqtSignal(QImage)
    state_changed = pyqtSignal(str, str)  # (state_name, message)
    packet_status_updated = pyqtSignal(dict)
    jewellery_count_updated = pyqtSignal(dict)
    result_updated = pyqtSignal(dict)
    model_report_ready = pyqtSignal(dict)
    inspection_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        packet_id: str,
        cut_status: str = "CUTTED",
        config: Optional[InspectionConfig] = None,
        shared_models: Optional[Dict[str, Any]] = None,
        camera_source: Optional[Union[int, str]] = None,
        frame_provider: Optional[FrameProvider] = None,
        existing_capture: Optional[cv2.VideoCapture] = None,
        auto_advance_to_stage2: bool = False,
        frame_interval_sec: float = 0.03,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.packet_id = packet_id
        self.cut_status = cut_status
        self.config = config if config is not None else InspectionConfig.load()
        self.shared_models = shared_models
        self.camera_source = camera_source
        self.frame_provider = frame_provider
        self.existing_capture = existing_capture
        self.auto_advance_to_stage2 = auto_advance_to_stage2
        self.frame_interval_sec = max(0.005, float(frame_interval_sec))

        self._lock = threading.Lock()
        self._stop_requested = False
        self._pending_commands: list[Tuple[str, Any]] = []
        self._engine: Optional[PostCutInspectionEngine] = None
        self.last_frame: Optional[np.ndarray] = None

    def stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def enqueue_command(self, command: str, payload: Any = None) -> None:
        with self._lock:
            self._pending_commands.append((command, payload))

    def request_proceed_stage2(self, require_empty_check: bool = True) -> None:
        self.enqueue_command("proceed_stage2", require_empty_check)

    def request_retry_stage1(self) -> None:
        self.enqueue_command("retry_stage1")

    def request_retry_stage2(self, require_empty_check: bool = False) -> None:
        self.enqueue_command("retry_stage2", require_empty_check)

    def request_confirm_count(self, override_count: Optional[int] = None) -> None:
        self.enqueue_command("confirm_count", override_count)

    def request_update_testbed_roi(self, roi: Tuple[int, int, int, int]) -> None:
        self.enqueue_command("set_roi", roi)

    def request_auto_calibrate_roi(self) -> None:
        self.enqueue_command("auto_calibrate_roi")

    def _pop_commands(self) -> list[Tuple[str, Any]]:
        with self._lock:
            cmds = list(self._pending_commands)
            self._pending_commands.clear()
            return cmds

    def _is_stopped(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _open_owned_capture(self) -> Optional[cv2.VideoCapture]:
        if self.frame_provider is not None or self.existing_capture is not None:
            return None
        src = 0 if self.camera_source is None else self.camera_source
        if isinstance(src, int):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open camera or video source '{src}'. "
                "Check camera connection or pass an existing frame provider."
            )
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _read_next_frame(self, owned_cap: Optional[cv2.VideoCapture]) -> Tuple[bool, Optional[np.ndarray]]:
        if self.frame_provider is not None:
            return self.frame_provider()
        cap = self.existing_capture if self.existing_capture is not None else owned_cap
        if cap is None:
            return False, None
        return cap.read()

    def _handle_commands(self, engine: PostCutInspectionEngine, last_frame: Optional[np.ndarray]) -> None:
        for cmd, payload in self._pop_commands():
            try:
                if cmd == "proceed_stage2":
                    engine.proceed_to_jewellery_counting(require_empty_check=bool(payload))
                    self.state_changed.emit(engine.state_machine.state.value, "Switched to Stage 2: Jewellery Counting")
                elif cmd == "retry_stage1":
                    engine.retry_packet_inspection()
                    self.state_changed.emit(engine.state_machine.state.value, "Retrying Stage 1 packet inspection")
                elif cmd == "retry_stage2":
                    engine.retry_jewellery_counting(require_empty_check=bool(payload))
                    self.state_changed.emit(engine.state_machine.state.value, "Retrying Stage 2 jewellery counting")
                elif cmd == "confirm_count":
                    res = engine.confirm_final_count(override_count=payload)
                    self.jewellery_count_updated.emit(res.jewellery_counting.to_dict())
                    self.result_updated.emit(res.to_dict())
                    if res.inspection_status == "COMPLETED":
                        self.state_changed.emit(
                            InspectionState.INSPECTION_COMPLETED.value,
                            f"Inspection completed: {res.packet_inspection.status}, count={res.jewellery_counting.count}",
                        )
                        self.inspection_completed.emit(res.to_dict())
                elif cmd == "set_roi" and payload is not None:
                    engine.stage2.roi_manager.set_roi(tuple(payload))
                elif cmd == "auto_calibrate_roi" and last_frame is not None:
                    detected = engine.stage2.roi_manager.detect_green_testbed_roi(last_frame)
                    if detected is not None:
                        engine.stage2.roi_manager.set_roi(detected)
            except Exception as exc:
                self.error_occurred.emit(f"Command '{cmd}' error: {exc}")

    def run(self) -> None:
        owned_cap: Optional[cv2.VideoCapture] = None
        try:
            self._engine = PostCutInspectionEngine(
                config=self.config,
                shared_models=self.shared_models,
                auto_load_models=True,
            )
            self.model_report_ready.emit(self._engine.model_report.to_dict())
            result = self._engine.on_cut_confirmed(self.packet_id, cut_status=self.cut_status)
            self.state_changed.emit(
                self._engine.state_machine.state.value,
                f"Post-cut inspection activated for packet {self.packet_id}",
            )
            self.result_updated.emit(result.to_dict())

            owned_cap = self._open_owned_capture()
            last_state = self._engine.state_machine.state

            while not self._is_stopped():
                self._handle_commands(self._engine, self.last_frame)

                if self._engine.state_machine.state == InspectionState.INSPECTION_COMPLETED:
                    break

                ok, frame = self._read_next_frame(owned_cap)
                if not ok or frame is None:
                    # End of video file or temporary camera drop
                    if isinstance(self.camera_source, str):
                        time.sleep(0.05)
                        break
                    time.sleep(0.05)
                    continue

                self.last_frame = frame
                step = self._engine.process_frame(
                    frame,
                    auto_advance_to_stage2=self.auto_advance_to_stage2,
                )

                self.frame_ready.emit(bgr_to_qimage(step.annotated_frame))

                if step.state != last_state:
                    last_state = step.state
                    self.state_changed.emit(step.state.value, step.message)

                self.packet_status_updated.emit(step.result.packet_inspection.to_dict())
                self.jewellery_count_updated.emit(step.result.jewellery_counting.to_dict())
                self.result_updated.emit(step.result.to_dict())

                time.sleep(self.frame_interval_sec)

        except Exception as exc:
            self.error_occurred.emit(f"{type(exc).__name__}: {exc}")
        finally:
            # IMPORTANT: Never release `existing_capture` owned by the host application!
            if owned_cap is not None:
                owned_cap.release()


class InspectionController(QObject):
    """High-level Qt signal/slot facade bridging `CUT_CONFIRMED` and the UI panel."""

    frame_ready = pyqtSignal(QImage)
    state_changed = pyqtSignal(str, str)
    packet_status_updated = pyqtSignal(dict)
    jewellery_count_updated = pyqtSignal(dict)
    result_updated = pyqtSignal(dict)
    model_report_ready = pyqtSignal(dict)
    inspection_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        config: Optional[InspectionConfig] = None,
        shared_models: Optional[Dict[str, Any]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.config = config if config is not None else InspectionConfig.load()
        self.shared_models = shared_models
        self._worker: Optional[InspectionWorker] = None
        self._last_result: Optional[Dict[str, Any]] = None

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        return self._last_result

    def start_on_cut_confirmed(
        self,
        packet_id: str,
        cut_status: str = "CUTTED",
        camera_source: Optional[Union[int, str]] = None,
        frame_provider: Optional[FrameProvider] = None,
        existing_capture: Optional[cv2.VideoCapture] = None,
        shared_models: Optional[Dict[str, Any]] = None,
        auto_advance_to_stage2: bool = False,
    ) -> None:
        """Start post-cutting inspection in a background thread upon CUT_CONFIRMED."""
        self.stop()
        models = shared_models if shared_models is not None else self.shared_models
        self._worker = InspectionWorker(
            packet_id=packet_id,
            cut_status=cut_status,
            config=self.config,
            shared_models=models,
            camera_source=camera_source,
            frame_provider=frame_provider,
            existing_capture=existing_capture,
            auto_advance_to_stage2=auto_advance_to_stage2,
            parent=self,
        )
        self._worker.frame_ready.connect(self.frame_ready)
        self._worker.state_changed.connect(self.state_changed)
        self._worker.packet_status_updated.connect(self.packet_status_updated)
        self._worker.jewellery_count_updated.connect(self.jewellery_count_updated)
        self._worker.result_updated.connect(self._on_result_updated)
        self._worker.model_report_ready.connect(self.model_report_ready)
        self._worker.inspection_completed.connect(self.inspection_completed)
        self._worker.error_occurred.connect(self.error_occurred)
        self._worker.start()

    def _on_result_updated(self, data: Dict[str, Any]) -> None:
        self._last_result = data
        self.result_updated.emit(data)

    def proceed_to_jewellery_counting(self, require_empty_check: bool = True) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_proceed_stage2(require_empty_check=require_empty_check)

    def retry_packet_inspection(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_retry_stage1()

    def retry_jewellery_counting(self, require_empty_check: bool = False) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_retry_stage2(require_empty_check=require_empty_check)

    def confirm_final_count(self, override_count: Optional[int] = None) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_confirm_count(override_count=override_count)

    def update_testbed_roi(self, roi: Tuple[int, int, int, int]) -> None:
        self.config.jewellery_counting.testbed_roi = roi
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_update_testbed_roi(roi)

    def auto_calibrate_testbed_roi(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_auto_calibrate_roi()

    def stop(self) -> None:
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker.stop()
                self._worker.wait(5000)
            self._worker = None

