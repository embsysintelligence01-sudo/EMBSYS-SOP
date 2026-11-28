"""Automated background QThread worker for the EMBSYS Packet-Cut Desktop App.

Guarantees:
- Loads all models ONCE into the shared `UnifiedModelManager` cache.
- Single camera acquisition pipeline (flushes stale buffered frames on open).
- Processes strictly ONE state per fresh camera frame via `AutomatedPacketPipeline`.
- STOP always saves completed events, partial/provisional results, reached stage,
  evidence images, and JSON/Markdown reports before releasing camera resources.
"""

from __future__ import annotations

import threading
import sys
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from inspection.utils.inspection_config import InspectionConfig
from workflow_monitor.config import WorkflowMonitorConfig
from workflow_monitor.models import UnifiedModelManager

from .automation import AutoPipelineConfig, AutomatedPacketPipeline
from .capture import LatestFrameCapture

FrameProvider = Callable[[], Tuple[bool, Optional[np.ndarray]]]


def _bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class PacketWorker(QThread):
    """Fully automated START-to-STOP background worker for MainWindow."""

    acquisition_stopped = pyqtSignal()
    frame_ready = pyqtSignal(QImage)
    analysis_frame_ready = pyqtSignal(QImage, float, dict)
    live_monitoring_updated = pyqtSignal(dict)
    compliance_updated = pyqtSignal(dict)
    system_monitoring_updated = pyqtSignal(dict)
    alert_notification = pyqtSignal(dict)
    alert_resolved = pyqtSignal(dict)
    activity_event = pyqtSignal(dict)
    final_report_ready = pyqtSignal(dict, dict)
    session_finished = pyqtSignal(str)
    error = pyqtSignal(str)
    setup_ready = pyqtSignal(QImage, str)

    def __init__(
        self,
        camera_id: Union[int, str] = 0,
        device: Any = "cpu",
        model_manager: Optional[UnifiedModelManager] = None,
        pipeline_config: Optional[AutoPipelineConfig] = None,
        workflow_config: Optional[WorkflowMonitorConfig] = None,
        inspection_config: Optional[InspectionConfig] = None,
        frame_provider: Optional[FrameProvider] = None,
        frame_interval_sec: float = 0.03,
        parent=None,
        *,
        prepare_only: bool = False,
        visualization: bool = True,
        manual_roi: Optional[Tuple[int, int, int, int]] = None,
        empty_reference: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self.device = device
        self.wf_cfg = workflow_config if workflow_config is not None else WorkflowMonitorConfig.load()
        self.wf_cfg.models.device = str(device)
        self.insp_cfg = inspection_config if inspection_config is not None else InspectionConfig.load()
        self.insp_cfg.models.device = str(device)
        self.pipe_cfg = pipeline_config if pipeline_config is not None else AutoPipelineConfig()

        self.model_manager = (
            model_manager if model_manager is not None else UnifiedModelManager(self.wf_cfg)
        )
        self.frame_provider = frame_provider
        self.frame_interval_sec = max(0.005, float(frame_interval_sec))
        self.visualization = visualization
        self.pipe_cfg.visualization = visualization
        self.prepare_only = prepare_only
        self.manual_roi = manual_roi
        self.empty_reference = empty_reference.copy() if empty_reference is not None else None

        self._lock = threading.Lock()
        self._stop_requested = False
        self._retry_stage: Optional[str] = None
        self.analysis_fps = 0.0
        self.last_snapshot = None
        self.last_frame: Optional[np.ndarray] = None
        self.pipeline: Optional[AutomatedPacketPipeline] = None

    def stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def request_stage_retry(self, stage: str) -> None:
        with self._lock:
            self._retry_stage = stage

    def _is_stopped(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _emit_system_status(
        self,
        model_status: str,
        active_model: str,
        camera_status: str,
        errors: str = "0 (None)",
        recovery: str = "Nominal",
    ) -> None:
        self.system_monitoring_updated.emit({
            "model_loading_status": model_status,
            "active_inference_model": active_model,
            "camera_status": camera_status,
            "processing_errors": errors,
            "automatic_recovery_status": recovery,
        })

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        if self.frame_provider is not None:
            return None
        if isinstance(self.camera_id, int):
            backend = cv2.CAP_ANY
            if sys.platform == "win32":
                backend = cv2.CAP_DSHOW
            elif sys.platform.startswith("linux"):
                backend = cv2.CAP_V4L2
            cap = cv2.VideoCapture(self.camera_id, backend)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.camera_id)
        else:
            cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Cannot open camera source '{self.camera_id}'. "
                "Verify USB webcam connection or supply a valid video path."
            )

        if isinstance(self.camera_id, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Flush stale buffered webcam frames so pre-START frames never count as evidence
            for _ in range(3):
                cap.grab()
        return cap

    def run(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        capture = None
        capture_sequence = 0
        analysis_rate = None
        last_analysis_finished = None
        try:
            if self.prepare_only:
                avail = self.model_manager.load_all_models()
                required = set(self.model_manager.ALL_KEYS) - {"sam"}
                missing = sorted(required - avail["loaded"].keys())
                if missing and not self.pipe_cfg.pass_by:
                    details = ", ".join(f"{key} ({avail['missing'].get(key, 'unavailable')})" for key in missing)
                    raise RuntimeError(f"Models unavailable: {details}")
                cap = self._open_camera()
                if self.frame_provider is not None:
                    ok, frame = self.frame_provider()
                else:
                    assert cap is not None
                    ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Could not read a camera frame for ROI setup.")
                self.last_frame = frame.copy()
                details = f" | Unavailable checks: {', '.join(missing)}" if missing else ""
                self.setup_ready.emit(_bgr_to_qimage(frame), f"{len(avail['loaded'])} loaded ({avail['device']}){details}")
                return

            # 1. Load models ONCE (reused if already cached in shared model_manager)
            if len(self.model_manager.loaded) == 0:
                self._emit_system_status(
                    model_status="Loading models (one-time startup)...",
                    active_model="Initializing",
                    camera_status="STARTING",
                )
            avail = self.model_manager.ensure_loaded()

            # 2. Initialize strict event-driven pipeline (model loading NEVER counts as stage evidence)
            self.pipeline = AutomatedPacketPipeline(
                model_manager=self.model_manager,
                pipeline_config=self.pipe_cfg,
                workflow_config=self.wf_cfg,
                inspection_config=self.insp_cfg,
                on_alert=lambda rec: self.alert_notification.emit(rec.to_dict()),
                on_alert_resolved=lambda rec: self.alert_resolved.emit(rec.to_dict()),
                on_activity=lambda ev: self.activity_event.emit(ev.to_dict()),
            )
            self.pipeline.refresh_shared_models()
            cut_file = self.pipeline._cut_model_filename
            print(f"[AUTOMATION] Models ready ({len(avail['loaded'])} loaded). Cut model: {cut_file}")

            self._emit_system_status(
                model_status=f"{len(avail['loaded'])} loaded ({avail['device']}) | Cut: {cut_file}",
                active_model="Ready",
                camera_status="CONNECTING",
            )

            # 3. Open single camera stream and start session showing "Clear the testbed."
            self.pipeline.start_session()
            cap = self._open_camera()
            live_source = isinstance(self.camera_id, int) or str(self.camera_id).startswith(('/dev/video', 'rtsp://', 'http://', 'https://'))
            if self.pipe_cfg.pass_by and self.frame_provider is None and live_source:
                capture = LatestFrameCapture(cap, (lambda frame: self.frame_ready.emit(_bgr_to_qimage(frame))) if self.visualization else None)
                capture.start()
            if self.manual_roi is not None:
                x1, y1, x2, y2 = self.manual_roi
                local_roi = (0, 0, x2 - x1, y2 - y1)
                self.pipeline.testbed_roi.set_roi(local_roi)
                self.pipeline.cutting_roi = local_roi
            if self.empty_reference is not None:
                self.pipeline.set_empty_reference(self.empty_reference)
            self.pipeline.camera_status = "ACTIVE"
            self.live_monitoring_updated.emit({
                "stage": self.pipeline.stage.value,
                "instruction": self.pipeline._instruction,
                "packet_id": "UNRESOLVED",
                "identity_status": "UNRESOLVED",
                "active_model": "Waiting for fresh frame",
                "processing_status": self.pipeline._instruction,
                "packet_detection": "Waiting for fresh frame",
                "cutting_verification": f"NOT_STARTED [{cut_file}]",
                "packet_condition": "Mouth: UNKNOWN | Gold: None",
                "jewellery_count": "0 (waiting)",
            })

            consecutive_read_failures = 0
            while not self._is_stopped():
                frame_captured_at = None
                if self.frame_provider is not None:
                    ok, frame = self.frame_provider()
                elif capture is not None:
                    sample = capture.next_frame(capture_sequence)
                    if sample is None and not capture.closed:
                        continue
                    if sample is None:
                        raise RuntimeError(capture.error or 'Camera stream ended.')
                    capture_sequence, frame, frame_captured_at = sample
                    ok = True
                else:
                    assert cap is not None
                    ok, frame = cap.read()

                if not ok or frame is None:
                    consecutive_read_failures += 1
                    if isinstance(self.camera_id, str) or self.frame_provider is not None:
                        break
                    self.pipeline.camera_status = f"RECOVERING (miss #{consecutive_read_failures})"
                    self.pipeline.recovery.total_retries += 1
                    if consecutive_read_failures >= 30:
                        self.pipeline.camera_status = "DISCONNECTED"
                        self.error.emit("Camera stream lost after 30 consecutive read failures.")
                        break
                    time.sleep(0.05)
                    continue

                if consecutive_read_failures > 0:
                    self.pipeline.recovery.recovered_errors += 1
                    consecutive_read_failures = 0
                self.pipeline.camera_status = "ACTIVE"

                self.last_frame = frame.copy()
                with self._lock:
                    retry_stage = self._retry_stage
                    self._retry_stage = None
                if retry_stage is not None:
                    self.pipeline.retry_current_stage(retry_stage)
                # Strictly ONE state evaluated on this fresh frame
                analysis_started = time.monotonic()
                if self.manual_roi is None:
                    snap = self.pipeline.process_frame(frame, now_sec=frame_captured_at)
                    preview = snap.annotated_frame
                else:
                    height, width = frame.shape[:2]
                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        raise RuntimeError("The selected ROI is outside the camera frame. Check readiness and redraw the ROI.")
                    region = frame[y1:y2, x1:x2].copy()
                    snap = self.pipeline.process_frame(region, now_sec=frame_captured_at)
                    preview = frame.copy() if self.visualization else frame
                    if self.visualization:
                        preview[y1:y2, x1:x2] = snap.annotated_frame

                self.last_snapshot = snap
                finished = time.monotonic()
                elapsed = finished - analysis_started
                cadence = elapsed if last_analysis_finished is None else finished - last_analysis_finished
                last_analysis_finished = finished
                instant_rate = 1 / max(cadence, .001)
                analysis_rate = instant_rate if analysis_rate is None else .75 * analysis_rate + .25 * instant_rate
                self.analysis_fps = analysis_rate
                if capture is None and self.visualization:
                    self.frame_ready.emit(_bgr_to_qimage(preview))
                elif capture is not None:
                    snap.system_monitoring['camera_status'] = (
                        f'LIVE {capture.preview_rate:.0f} fps | checks {analysis_rate:.1f} fps'
                    )
                    if self.visualization and snap.live_monitoring.get('cutting_detection_fresh'):
                        self.analysis_frame_ready.emit(_bgr_to_qimage(snap.annotated_frame),
                                                       frame_captured_at or finished, dict(snap.live_monitoring))
                self.live_monitoring_updated.emit(snap.live_monitoring)
                self.compliance_updated.emit(snap.compliance_monitoring)
                self.system_monitoring_updated.emit(snap.system_monitoring)

                if snap.completed and snap.final_report is not None:
                    self.final_report_ready.emit(snap.final_report, snap.report_paths)
                    self.session_finished.emit("COMPLETED")
                    break

                time.sleep(self.frame_interval_sec)

            # STOP or stream end ALWAYS saves full/partial result & evidence
            if capture is not None:
                capture.stop()
                capture.join(2)
            if capture is not None:
                while capture.is_alive():
                    capture.join(.2)
            if cap is not None:
                cap.release()
                cap = None
            self.acquisition_stopped.emit()
            if self.pipeline is not None and self.pipeline._final_report is None:
                report = self.pipeline.stop_session()
                self.final_report_ready.emit(report, dict(self.pipeline._report_paths))
                self._emit_system_status(
                    model_status=f"{len(avail['loaded'])} loaded (cached) | Cut: {cut_file}",
                    active_model="Stopped",
                    camera_status="STOPPED & RELEASED",
                    errors=f"{self.pipeline.recovery.total_inference_errors} ({self.pipeline.recovery.last_error})",
                    recovery=self.pipeline.recovery.last_recovery,
                )
                self.session_finished.emit("STOPPED")

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            self._emit_system_status(
                model_status="Error",
                active_model="Stopped",
                camera_status="ERROR",
                errors=err_msg,
                recovery="Stopped safely",
            )
            self.error.emit(err_msg)
            if capture is not None:
                capture.stop()
                while capture.is_alive():
                    capture.join(.2)
            if cap is not None:
                cap.release()
                cap = None
            self.acquisition_stopped.emit()
            # Preserve partial evidence and flush pending clips on a camera or
            # processing failure, just as STOP does.
            if self.pipeline is not None and self.pipeline.activity_logger is not None and self.pipeline._final_report is None:
                try:
                    report = self.pipeline.stop_session()
                    self.final_report_ready.emit(report, dict(self.pipeline._report_paths))
                except Exception as save_error:
                    self.error.emit(f"Could not save partial report: {save_error}")
            self.session_finished.emit("ERROR")
        finally:
            if capture is not None:
                capture.stop()
                capture.join(2)
            if cap is not None:
                cap.release()
            if capture is not None and capture.is_alive():
                capture.join(.5)
