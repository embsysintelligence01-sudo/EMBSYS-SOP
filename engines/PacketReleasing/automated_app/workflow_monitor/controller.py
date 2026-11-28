"""PyQt6 background QThread worker and signal/slot controller for Automated Monitoring.

Executes all vision inference, procedure compliance evaluation, alert generation,
logging, and automatic report creation off the main GUI thread.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QImage

from inspection.inspection_controller import bgr_to_qimage
from inspection.utils.inspection_config import InspectionConfig

from .activity_log import ActivityEvent
from .alerts import AlertRecord
from .automated_engine import AutomatedWorkflowEngine
from .config import WorkflowMonitorConfig

FrameProvider = Callable[[], Tuple[bool, Optional[np.ndarray]]]


class AutomatedWorkflowWorker(QThread):
    """Single-START background thread running `AutomatedWorkflowEngine`."""

    session_started = pyqtSignal(str)
    frame_ready = pyqtSignal(QImage)
    stage_updated = pyqtSignal(str, str, str, str)  # (stage, packet_id, identity_status, message)
    alert_raised = pyqtSignal(dict)
    alert_resolved = pyqtSignal(dict)
    activity_logged = pyqtSignal(dict)
    compliance_updated = pyqtSignal(dict)
    operational_updated = pyqtSignal(dict)
    report_ready = pyqtSignal(dict, dict)  # (report_dict, report_paths)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        camera_source: Union[int, str] = 0,
        workflow_config: Optional[WorkflowMonitorConfig] = None,
        inspection_config: Optional[InspectionConfig] = None,
        preloaded_models: Optional[Dict[str, Any]] = None,
        frame_provider: Optional[FrameProvider] = None,
        frame_interval_sec: float = 0.03,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.camera_source = camera_source
        self.wf_cfg = workflow_config if workflow_config is not None else WorkflowMonitorConfig.load()
        self.insp_cfg = inspection_config if inspection_config is not None else InspectionConfig.load()
        self.preloaded_models = preloaded_models
        self.frame_provider = frame_provider
        self.frame_interval_sec = max(0.005, float(frame_interval_sec))

        self._lock = threading.Lock()
        self._stop_requested = False
        self._engine: Optional[AutomatedWorkflowEngine] = None

    def stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def _is_stopped(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        if self.frame_provider is not None:
            return None
        src = self.camera_source
        if isinstance(src, int):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera or video source '{src}'. "
                "Verify camera connection or supply a valid video file path."
            )
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        try:
            self._engine = AutomatedWorkflowEngine(
                workflow_config=self.wf_cfg,
                inspection_config=self.insp_cfg,
                preloaded_models=self.preloaded_models,
                auto_load_models=True,
                on_alert=lambda rec: self.alert_raised.emit(rec.to_dict()),
                on_alert_resolved=lambda rec: self.alert_resolved.emit(rec.to_dict()),
                on_activity=lambda ev: self.activity_logged.emit(ev.to_dict()),
            )

            session_id = self._engine.start_session()
            self.session_started.emit(session_id)
            cap = self._open_camera()

            completed_naturally = False
            while not self._is_stopped():
                if self.frame_provider is not None:
                    ok, frame = self.frame_provider()
                else:
                    assert cap is not None
                    ok, frame = cap.read()

                if not ok or frame is None:
                    if isinstance(self.camera_source, str) or self.frame_provider is not None:
                        break
                    time.sleep(0.05)
                    continue

                step = self._engine.process_frame(frame)
                self.frame_ready.emit(bgr_to_qimage(step.annotated_frame))
                self.stage_updated.emit(
                    step.stage.value,
                    step.packet_id,
                    step.identity_status,
                    step.status_message,
                )
                self.compliance_updated.emit(step.compliance_summary)
                self.operational_updated.emit(step.operational_snapshot)

                if step.completed and step.final_report is not None:
                    completed_naturally = True
                    self.report_ready.emit(step.final_report, step.report_paths)
                    break

                time.sleep(self.frame_interval_sec)

            if not completed_naturally and self._engine is not None:
                report = self._engine.stop_and_generate_report()
                self.compliance_updated.emit(report["appraiser_compliance"])
                self.report_ready.emit(report, dict(self._engine._report_paths))

        except Exception as exc:
            self.error_occurred.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if cap is not None:
                cap.release()


class AutomatedWorkflowController(QObject):
    """High-level Qt controller exposing a single `start()` entry point."""

    session_started = pyqtSignal(str)
    frame_ready = pyqtSignal(QImage)
    stage_updated = pyqtSignal(str, str, str, str)
    alert_raised = pyqtSignal(dict)
    alert_resolved = pyqtSignal(dict)
    activity_logged = pyqtSignal(dict)
    compliance_updated = pyqtSignal(dict)
    operational_updated = pyqtSignal(dict)
    report_ready = pyqtSignal(dict, dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        camera_source: Union[int, str] = 0,
        workflow_config: Optional[WorkflowMonitorConfig] = None,
        inspection_config: Optional[InspectionConfig] = None,
        preloaded_models: Optional[Dict[str, Any]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.camera_source = camera_source
        self.wf_cfg = workflow_config if workflow_config is not None else WorkflowMonitorConfig.load()
        self.insp_cfg = inspection_config if inspection_config is not None else InspectionConfig.load()
        self.preloaded_models = preloaded_models
        self._worker: Optional[AutomatedWorkflowWorker] = None
        self.last_report: Optional[Dict[str, Any]] = None
        self.last_report_paths: Dict[str, Optional[str]] = {}

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def start(self, frame_provider: Optional[FrameProvider] = None) -> None:
        self.stop()
        self.last_report = None
        self.last_report_paths = {}
        self._worker = AutomatedWorkflowWorker(
            camera_source=self.camera_source,
            workflow_config=self.wf_cfg,
            inspection_config=self.insp_cfg,
            preloaded_models=self.preloaded_models,
            frame_provider=frame_provider,
            parent=self,
        )
        self._worker.session_started.connect(self.session_started)
        self._worker.frame_ready.connect(self.frame_ready)
        self._worker.stage_updated.connect(self.stage_updated)
        self._worker.alert_raised.connect(self.alert_raised)
        self._worker.alert_resolved.connect(self.alert_resolved)
        self._worker.activity_logged.connect(self.activity_logged)
        self._worker.compliance_updated.connect(self.compliance_updated)
        self._worker.operational_updated.connect(self.operational_updated)
        self._worker.report_ready.connect(self._on_report_ready)
        self._worker.error_occurred.connect(self.error_occurred)
        self._worker.start()

    def _on_report_ready(self, report: Dict[str, Any], paths: Dict[str, Optional[str]]) -> None:
        self.last_report = report
        self.last_report_paths = paths
        self.report_ready.emit(report, paths)

    def stop(self) -> None:
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker.stop()
                self._worker.wait(8000)
            self._worker = None

