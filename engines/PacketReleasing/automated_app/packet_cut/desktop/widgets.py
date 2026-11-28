"""PyQt6 dashboard for automated EMBSYS packet-cut inspection."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from pathlib import Path
import time
from PyQt6.QtCore import QRect, QRectF, Qt, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QRubberBand,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from workflow_monitor.config import WorkflowMonitorConfig
from workflow_monitor.models import UnifiedModelManager
from workflow_monitor.report_generator import FinalReportGenerator

from .automation import AutoPipelineConfig
from .state_machine import AutoStage
from .worker import FrameProvider, PacketWorker


DASHBOARD_STYLESHEET = """
QWidget#root { background:#0b1220; color:#e8eef8; }
QLabel { color:#e8eef8; }
QLabel#appTitle { font-size:20px; font-weight:800; letter-spacing:0.8px; }
QLabel#subTitle { color:#93a4bd; font-size:11px; }
QLabel#liveBadge { background:#12382f; color:#67e8b2; padding:6px 12px; border-radius:10px; font-weight:800; font-size:11px; }
QLabel#instructionBanner { background:#102a42; border:1px solid #38bdf8; color:#e0f2fe; border-radius:8px; padding:7px 12px; font-weight:800; font-size:13px; }
QLabel#alertBanner { background:#4b1d24; border:1px solid #f87171; color:#fecaca; border-radius:8px; padding:7px 12px; font-weight:800; font-size:12px; }
QLabel#video { background:#050a12; border:1px solid #27364b; border-radius:12px; color:#8091a9; font-size:14px; }
QFrame#sectionCard { background:#121e30; border:1px solid #25364d; border-radius:10px; }
QLabel#sectionTitle { color:#38bdf8; font-size:10px; font-weight:900; letter-spacing:1.0px; }
QLabel#fieldCaption { color:#8295b1; font-size:10px; font-weight:700; }
QLabel#fieldValue { color:#f8fafc; font-size:12px; font-weight:800; }
QLabel#highlightValue { color:#61e6ad; font-size:13px; font-weight:900; }
QPlainTextEdit#displayLog {
    background:#0c1624;
    border:1px solid #223247;
    border-radius:7px;
    color:#cbd5e1;
    font-size:11px;
    padding:4px;
}
QPushButton#startBtn {
    background:#176b56;
    border:2px solid #34d399;
    border-radius:8px;
    padding:8px 22px;
    color:#ffffff;
    font-size:13px;
    font-weight:900;
}
QPushButton#startBtn:hover { background:#1c8469; }
QPushButton#startBtn:disabled { background:#122923; border-color:#23463c; color:#607a72; }
QPushButton#restartBtn {
    background:#1e3a5f;
    border:2px solid #38bdf8;
    border-radius:8px;
    padding:8px 22px;
    color:#ffffff;
    font-size:13px;
    font-weight:900;
}
QPushButton#restartBtn:hover { background:#254877; }
QPushButton#restartBtn:disabled { background:#162436; border-color:#2a4563; color:#5c708a; }
QPushButton#stopBtn {
    background:#7a2335;
    border:2px solid #f87171;
    border-radius:8px;
    padding:8px 22px;
    color:#ffffff;
    font-size:13px;
    font-weight:900;
}
QPushButton#stopBtn:hover { background:#962b41; }
QPushButton#stopBtn:disabled { background:#2a1720; border-color:#4b2833; color:#775a63; }
QPushButton#themeBtn {
    background:#1e3a5f;
    border:2px solid #64748b;
    border-radius:8px;
    padding:6px 14px;
    color:#e8eef8;
    font-size:12px;
    font-weight:800;
}
QPushButton#themeBtn:hover { background:#2d4a6f; }
"""

LIGHT_STYLESHEET = """
QWidget#root { background:#f0f4f8; color:#1e293b; }
QLabel { color:#1e293b; }
QLabel#appTitle { font-size:20px; font-weight:800; letter-spacing:0.8px; color:#0f172a; }
QLabel#subTitle { color:#64748b; font-size:11px; }
QLabel#liveBadge { background:#d1fae5; color:#065f46; padding:6px 12px; border-radius:10px; font-weight:800; font-size:11px; }
QLabel#instructionBanner { background:#dbeafe; border:1px solid #3b82f6; color:#1e3a8a; border-radius:8px; padding:7px 12px; font-weight:800; font-size:13px; }
QLabel#alertBanner { background:#fee2e2; border:1px solid #ef4444; color:#991b1b; border-radius:8px; padding:7px 12px; font-weight:800; font-size:12px; }
QLabel#video { background:#ffffff; border:1px solid #cbd5e1; border-radius:12px; color:#64748b; font-size:14px; }
QFrame#sectionCard { background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; }
QLabel#sectionTitle { color:#2563eb; font-size:10px; font-weight:900; letter-spacing:1.0px; }
QLabel#fieldCaption { color:#64748b; font-size:10px; font-weight:700; }
QLabel#fieldValue { color:#1e293b; font-size:12px; font-weight:800; }
QLabel#highlightValue { color:#059669; font-size:13px; font-weight:900; }
QPlainTextEdit#displayLog {
    background:#f8fafc;
    border:1px solid #e2e8f0;
    border-radius:7px;
    color:#334155;
    font-size:11px;
    padding:4px;
}
QPushButton#startBtn {
    background:#059669;
    border:2px solid #34d399;
    border-radius:8px;
    padding:8px 22px;
    color:#ffffff;
    font-size:13px;
    font-weight:900;
}
QPushButton#startBtn:hover { background:#047857; }
QPushButton#startBtn:disabled { background:#d1fae5; border-color:#a7f3d0; color:#6ee7b7; }
QPushButton#restartBtn {
    background:#2563eb;
    border:2px solid #60a5fa;
    border-radius:8px;
    padding:8px 22px;
    color:#ffffff;
    font-size:13px;
    font-weight:900;
}
QPushButton#restartBtn:hover { background:#1d4ed8; }
QPushButton#restartBtn:disabled { background:#dbeafe; border-color:#93c5fd; color:#93c5fd; }
QPushButton#stopBtn {
    background:#dc2626;
    border:2px solid #f87171;
    border-radius:8px;
    padding:8px 22px;
    color:#ffffff;
    font-size:13px;
    font-weight:900;
}
QPushButton#stopBtn:hover { background:#b91c1c; }
QPushButton#stopBtn:disabled { background:#fee2e2; border-color:#fca5a5; color:#fca5a5; }
QPushButton#themeBtn {
    background:#e2e8f0;
    border:2px solid #94a3b8;
    border-radius:8px;
    padding:6px 14px;
    color:#1e293b;
    font-size:12px;
    font-weight:800;
}
QPushButton#themeBtn:hover { background:#cbd5e1; }
"""


class ROISelectionLabel(QLabel):
    def __init__(self, image: QImage) -> None:
        super().__init__()
        self.image = image
        pixmap = QPixmap.fromImage(image).scaled(
            800, 560, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.origin = event.position().toPoint()
            self.band.setGeometry(QRect(self.origin, self.origin))
            self.band.show()

    def mouseMoveEvent(self, event) -> None:
        if self.origin is not None:
            self.band.setGeometry(QRect(self.origin, event.position().toPoint()).normalized().intersected(self.rect()))

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.origin is not None:
            self.mouseMoveEvent(event)
            self.origin = None

    def selected_roi(self) -> Optional[Tuple[int, int, int, int]]:
        box = self.band.geometry()
        if self.band.isHidden() or box.width() < 5 or box.height() < 5:
            return None
        sx = self.image.width() / self.width()
        sy = self.image.height() / self.height()
        return (round(box.x() * sx), round(box.y() * sy),
                round((box.right() + 1) * sx), round((box.bottom() + 1) * sy))


class MainWindow(QMainWindow):

    def __init__(
        self,
        camera: Union[int, str] = 0,
        device: Any = "cpu",
        model_manager: Optional[UnifiedModelManager] = None,
        pipeline_config: Optional[AutoPipelineConfig] = None,
        frame_provider: Optional[FrameProvider] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("EMBSYS · Packet-Cut Fully Automated Inspection")
        self.resize(1420, 860)
        self.setMinimumSize(1100, 680)

        self._camera = camera
        self._device = device
        self._pipeline_config = pipeline_config
        self._pass_by = pipeline_config.pass_by if pipeline_config is not None else True
        self._frame_provider = frame_provider

        wf_cfg = WorkflowMonitorConfig.load()
        wf_cfg.models.device = str(device)
        self._shared_model_manager = (
            model_manager if model_manager is not None else UnifiedModelManager(wf_cfg)
        )
        self.worker: Optional[PacketWorker] = None
        self._last_pixmap: Optional[QPixmap] = None
        self._detection_pixmap = None
        self._detection_time = None
        self._detection_label = ''
        self._setup_image: Optional[QImage] = None
        self._setup_frame = None
        self._empty_reference = None
        self._manual_roi: Optional[Tuple[int, int, int, int]] = None
        self._report_text = ""
        self._visual_report_path = None
        self._pdf_report_path = None
        self._restart_requested = False

        self._build_ui()
        self._is_dark_theme = True
        self.setStyleSheet(DASHBOARD_STYLESHEET)
        self.sys_model_load.setText("Waiting for readiness check")
        self.live_badge.setText("SETUP REQUIRED")

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(8)

        # ==============================================================
        # HEADER + MAIN CONTROLS
        # ==============================================================
        header = QHBoxLayout()
        # LOGO on the top‑left corner
        logo_label = QLabel()
        logo_path = "C:/Users/acer/.gemini/antigravity/brain/298517ee-e4ac-4920-8c3c-3b3aab7370ad/.user_uploaded/media_1795174641710_193d8910.png"
        logo_pix = QPixmap(logo_path).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        logo_label.setPixmap(logo_pix)
        logo_label.setFixedSize(logo_pix.size())
        header.addWidget(logo_label, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)


        
        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        title = QLabel("EMBSYS PACKET-CUT · FULLY AUTOMATED INSPECTION")
        title.setObjectName("appTitle")
        subtitle = QLabel("Automated 8-stage vision pipeline · Appraiser compliance monitoring")
        subtitle.setObjectName("subTitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header.addLayout(title_col, 1)

        self.live_badge = QLabel("● READY")
        self.live_badge.setObjectName("liveBadge")
        header.addWidget(self.live_badge)

        self.start_btn = QPushButton("START")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setMinimumWidth(115)

        self.restart_btn = QPushButton("RESTART")
        self.restart_btn.setObjectName("restartBtn")
        self.restart_btn.setMinimumWidth(115)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setMinimumWidth(115)
        self.stop_btn.setEnabled(False)
        self.check_btn = QPushButton("Load Models / Check Ready")
        self.roi_btn = QPushButton("Draw ROI")
        self.retry_btn = QPushButton("Refresh Pending Checks" if self._pass_by else "Retry Current Stage")
        self.roi_btn.setEnabled(False)
        self.retry_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.restart_btn.setEnabled(False)

        header.addWidget(self.start_btn)
        header.addWidget(self.restart_btn)
        header.addWidget(self.stop_btn)

        self.theme_btn = QPushButton("☀")
        self.theme_btn.setObjectName("themeBtn")
        self.theme_btn.setFixedWidth(40)
        header.addWidget(self.theme_btn)

        # preview_controls are added below the video widget (see left_col section)


        outer.addLayout(header)

        # Non-interactive automatic guidance instruction banner (display-only)
        self.instruction_banner = QLabel("Check readiness and select the work area. Release observations can start at any point.")
        self.instruction_banner.setObjectName("instructionBanner")
        self.instruction_banner.setWordWrap(True)
        self.instruction_banner.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        outer.addWidget(self.instruction_banner)

        # Non-interactive real-time alert notification banner (never requires click)
        self.alert_banner = QLabel("")
        self.alert_banner.setObjectName("alertBanner")
        self.alert_banner.setWordWrap(True)
        self.alert_banner.setVisible(False)
        outer.addWidget(self.alert_banner)

        # ==============================================================
        # MAIN DASHBOARD BODY (ALL DISPLAY-ONLY)
        # ==============================================================
        body = QHBoxLayout()
        body.setSpacing(10)
        outer.addLayout(body, 1)

        # --------------------------------------------------------------
        # LEFT COLUMN: LIVE MONITORING
        # --------------------------------------------------------------
        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        body.addLayout(left_col, 7)

        self.video = QLabel("Click START to begin fully automated monitoring")
        self.video.setObjectName("video")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(540, 380)
        self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_col.addWidget(self.video, 1)

        # Preview‑area controls (below video, above live card)
        preview_controls = QHBoxLayout()
        preview_controls.setSpacing(8)
        preview_controls.addWidget(self.check_btn)
        preview_controls.addWidget(self.roi_btn)
        preview_controls.addWidget(self.retry_btn)
        left_col.addLayout(preview_controls)

        live_card = self._make_section_card("LIVE MONITORING")
        lg = QGridLayout()
        lg.setSpacing(6)
        self.lm_stage = self._value_label("IDLE", highlight=True)
        self.lm_packet_id = self._value_label("UNRESOLVED", highlight=True)
        self.lm_active_model = self._value_label("None")
        self.lm_status = self._value_label("System ready — click START.")
        self.lm_packet_det = self._value_label("—")
        self.lm_cut_verify = self._value_label("PENDING")
        self.lm_packet_cond = self._value_label("UNKNOWN")
        self.lm_jewel_count = self._value_label("0", highlight=True)

        self._add_kv(lg, 0, 0, "Current Stage", self.lm_stage)
        self._add_kv(lg, 0, 1, "Packet ID", self.lm_packet_id)
        self._add_kv(lg, 0, 2, "Active Model", self.lm_active_model)
        self._add_kv(lg, 0, 3, "Jewellery Count", self.lm_jewel_count)
        self._add_kv(lg, 1, 0, "Packet Detection", self.lm_packet_det)
        self._add_kv(lg, 1, 1, "Cutting Verification", self.lm_cut_verify)
        self._add_kv(lg, 1, 2, "Packet OPEN / CLOSED", self.lm_packet_cond)
        self._add_kv(lg, 1, 3, "Processing Status", self.lm_status)
        live_card.layout().addLayout(lg)
        left_col.addWidget(live_card)

        # --------------------------------------------------------------
        # RIGHT COLUMN: APPRAISER COMPLIANCE + SYSTEM MONITORING + FINAL REPORT
        # --------------------------------------------------------------
        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        body.addLayout(right_col, 5)

        # 1. APPRAISER COMPLIANCE (display-only)
        comp_card = self._make_section_card("APPRAISER COMPLIANCE")
        cg = QGridLayout()
        cg.setSpacing(5)
        self.cmp_observed = self._value_label("Waiting for START")
        self.cmp_expected = self._value_label("Click START to begin automated inspection.")
        self.cmp_correct = self._value_label("0 / 9", highlight=True)
        self.cmp_skipped = self._value_label("0")
        self.cmp_deviations = self._value_label("0")

        self._add_kv(cg, 0, 0, "Latest Observed Action", self.cmp_observed, colspan=2)
        self._add_kv(cg, 0, 2, "Correct / Expected", self.cmp_correct)
        self._add_kv(cg, 1, 0, "Expected Procedure", self.cmp_expected, colspan=2)
        self._add_kv(cg, 1, 2, "Skipped / Deviations", self.cmp_skipped)
        comp_card.layout().addLayout(cg)

        self.compliance_log = self._make_readonly_display(max_height=135)
        self.compliance_log.setPlainText("Compliance alerts and chronological event log will appear automatically.")
        comp_card.layout().addWidget(self.compliance_log)
        self.copy_compliance_btn = QPushButton("Copy to Clipboard")
        self.copy_compliance_btn.clicked.connect(self._copy_compliance)
        comp_card.layout().addWidget(self.copy_compliance_btn)
        right_col.addWidget(comp_card, 2)

        # 2. SYSTEM MONITORING (display-only)
        sys_card = self._make_section_card("SYSTEM MONITORING")
        sg = QGridLayout()
        sg.setSpacing(5)
        self.sys_model_load = self._value_label("Not loaded yet (loads once on START)")
        self.sys_active_model = self._value_label("Idle")
        self.sys_camera = self._value_label("IDLE")
        self.sys_errors = self._value_label("0 (None)")
        self.sys_recovery = self._value_label("Nominal")

        self._add_kv(sg, 0, 0, "Model Loading Status", self.sys_model_load)
        self._add_kv(sg, 0, 1, "Active Inference Model", self.sys_active_model)
        self._add_kv(sg, 0, 2, "Camera Status", self.sys_camera)
        self._add_kv(sg, 1, 0, "Processing Errors", self.sys_errors)
        self._add_kv(sg, 1, 1, "Automatic Recovery Status", self.sys_recovery, colspan=2)
        sys_card.layout().addLayout(sg)
        right_col.addWidget(sys_card, 1)

        # 3. FINAL REPORT (display-only, auto-populated on completion or STOP)
        rep_card = self._make_section_card("FINAL REPORT (AUTO-GENERATED)")
        rg = QGridLayout()
        rg.setSpacing(5)
        self.fr_packet = self._value_label("—")
        self.fr_cut = self._value_label("—")
        self.fr_cond = self._value_label("—")
        self.fr_count = self._value_label("—")
        self.fr_compliance = self._value_label("—")
        self.fr_overall = self._value_label("INACTIVE", highlight=True)

        self._add_kv(rg, 0, 0, "Packet ID", self.fr_packet)
        self._add_kv(rg, 0, 1, "Cutting Result", self.fr_cut)
        self._add_kv(rg, 0, 2, "Packet Condition", self.fr_cond)
        self._add_kv(rg, 1, 0, "Jewellery Count", self.fr_count)
        self._add_kv(rg, 1, 1, "Compliance Summary", self.fr_compliance)
        self._add_kv(rg, 1, 2, "Overall Status", self.fr_overall)
        rep_card.layout().addLayout(rg)

        self.final_report_view = self._make_readonly_display(max_height=175)
        self.final_report_view.setPlainText(
            "Final report (mistakes, deviations, evidence references, and summary) populates automatically upon completion or STOP."
        )
        rep_card.layout().addWidget(self.final_report_view)
        self.copy_report_btn = QPushButton("Copy Compliance Report")
        self.copy_report_btn.setEnabled(False)
        rep_card.layout().addWidget(self.copy_report_btn)
        self.open_report_btn = QPushButton("Open Visual Report")
        self.open_report_btn.setEnabled(False)
        self.open_pdf_btn = QPushButton("Open PDF Report")
        self.open_pdf_btn.setEnabled(False)
        report_actions = QHBoxLayout()
        report_actions.addWidget(self.open_report_btn)
        report_actions.addWidget(self.open_pdf_btn)
        rep_card.layout().addLayout(report_actions)
        right_col.addWidget(rep_card, 2)

        # Wire setup and session controls
        self.start_btn.clicked.connect(self.start_inspection)
        self.restart_btn.clicked.connect(self.restart_inspection)
        self.stop_btn.clicked.connect(self.stop_inspection)
        self.check_btn.clicked.connect(self._check_readiness)
        self.roi_btn.clicked.connect(self._draw_roi)
        self.retry_btn.clicked.connect(self._retry_stage)
        self.copy_report_btn.clicked.connect(self._copy_report)
        self.open_report_btn.clicked.connect(self._open_visual_report)
        self.open_pdf_btn.clicked.connect(self._open_pdf_report)
        self.theme_btn.clicked.connect(self._toggle_theme)

    def _toggle_theme(self) -> None:
        """Switch between dark and light themes."""
        self._is_dark_theme = not self._is_dark_theme
        if self._is_dark_theme:
            self.setStyleSheet(DASHBOARD_STYLESHEET)
            self.theme_btn.setText("☀")
        else:
            self.setStyleSheet(LIGHT_STYLESHEET)
            self.theme_btn.setText("🌙")

    @staticmethod
    def _make_section_card(title_text: str) -> QFrame:
        card = QFrame()
        card.setObjectName("sectionCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 7, 10, 7)
        lay.setSpacing(4)
        cap = QLabel(title_text)
        cap.setObjectName("sectionTitle")
        lay.addWidget(cap)
        return card

    @staticmethod
    def _value_label(text: str, highlight: bool = False) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("highlightValue" if highlight else "fieldValue")
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        return lbl

    @staticmethod
    def _add_kv(grid: QGridLayout, row: int, col: int, caption: str, value_lbl: QLabel, colspan: int = 1) -> None:
        box = QVBoxLayout()
        box.setSpacing(1)
        cap = QLabel(caption.upper())
        cap.setObjectName("fieldCaption")
        box.addWidget(cap)
        box.addWidget(value_lbl)
        grid.addLayout(box, row, col, 1, colspan)

    @staticmethod
    def _make_readonly_display(max_height: int = 150) -> QPlainTextEdit:
        view = QPlainTextEdit()
        view.setObjectName("displayLog")
        view.setReadOnly(True)
        view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        view.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        view.setMaximumHeight(max_height)
        return view

    # ==============================================================
    # SETUP AND SESSION CONTROLS
    # ==============================================================
    def _check_readiness(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self._setup_image = None
        self._setup_frame = None
        self._empty_reference = None
        self._manual_roi = None
        self._render_pixmap()
        self.alert_banner.hide()
        self.check_btn.setEnabled(False)
        self.roi_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.restart_btn.setEnabled(False)
        self.live_badge.setText("CHECKING SYSTEM")
        self.instruction_banner.setText("Loading models and checking the camera…")
        self.worker = PacketWorker(
            camera_id=self._camera, device=self._device, model_manager=self._shared_model_manager,
            frame_provider=self._frame_provider, parent=self, prepare_only=True,
        )
        self.worker.setup_ready.connect(self._on_setup_ready)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_setup_finished)
        self.worker.start()

    def _on_setup_ready(self, image: QImage, model_status: str) -> None:
        self._setup_image = image
        if self.worker is not None and self.worker.last_frame is not None:
            self._setup_frame = self.worker.last_frame.copy()
        self._on_frame_ready(image)
        self.sys_model_load.setText(model_status)
        self.sys_camera.setText("Preview captured")
        self.sys_errors.setText("0 (None)")
        self.live_badge.setText("MODELS & CAMERA READY")
        self.instruction_banner.setText("Draw ROI around the work area. Empty-bed calibration is optional; missing checks will not block observation.")

    def _on_setup_finished(self) -> None:
        if self.sender() is not self.worker:
            return
        self.check_btn.setEnabled(True)
        self.roi_btn.setEnabled(self._setup_image is not None)
        if self._setup_image is None:
            self.live_badge.setText("SETUP REQUIRED")
            self.instruction_banner.setText("Resolve the system error and check readiness again.")

    def _draw_roi(self) -> None:
        if self._setup_image is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Draw Testbed ROI")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select the work area. You can start with a packet already present." if self._pass_by else
                               "Select the EMPTY testbed. Fixed markings become the background reference."))
        calibrate = None
        if self._pass_by:
            calibrate = QCheckBox("This preview is empty of loose objects: use it to calibrate fixed markings (optional)")
            layout.addWidget(calibrate)
        preview = ROISelectionLabel(self._setup_image)
        layout.addWidget(preview)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(lambda: dialog.accept() if preview.selected_roi() is not None else None)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._manual_roi = preview.selected_roi()
            self._empty_reference = None
            if self._setup_frame is not None and (not self._pass_by or calibrate.isChecked()):
                x1, y1, x2, y2 = self._manual_roi
                self._empty_reference = self._setup_frame[y1:y2, x1:x2].copy()
            self._on_frame_ready(self._setup_image)
            self.start_btn.setEnabled(True)
            self.live_badge.setText("READY")
            self.instruction_banner.setText("ROI selected. Click START to begin inspection.")

    def _retry_stage(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stage_retry(self.lm_stage.text())

    def _copy_compliance(self) -> None:
        lines = [
            "Appraiser Compliance",
            f"Latest Observed Action: {self.cmp_observed.text()}",
            f"Expected Procedure: {self.cmp_expected.text()}",
            f"Correct / Expected: {self.cmp_correct.text()}",
            f"Skipped / Deviations: {self.cmp_skipped.text()}",
        ]
        log = self.compliance_log.toPlainText().strip()
        if log:
            lines.extend(("", log))
        QApplication.clipboard().setText("\n".join(lines))
        self.statusBar().showMessage("Appraiser compliance copied to clipboard.", 3000)

    def _copy_report(self) -> None:
        if self._report_text:
            QApplication.clipboard().setText(self._report_text)
            self.statusBar().showMessage("Compliance report copied to clipboard.", 3000)

    def _open_visual_report(self) -> None:
        if self._visual_report_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self._visual_report_path).resolve())))

    def _open_pdf_report(self) -> None:
        if self._pdf_report_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self._pdf_report_path).resolve())))

    def start_inspection(self) -> None:
        if self._manual_roi is None or (self.worker is not None and self.worker.isRunning()):
            return

        self.start_btn.setEnabled(False)
        self.check_btn.setEnabled(False)
        self.roi_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.restart_btn.setEnabled(True)
        self.copy_report_btn.setEnabled(False)
        self.open_report_btn.setEnabled(False)
        self._visual_report_path = None
        self.open_pdf_btn.setEnabled(False)
        self._pdf_report_path = None
        self._report_text = ""
        self._detection_pixmap = None
        self._detection_time = None
        self.live_badge.setText("● AUTOMATED SESSION RUNNING")
        self.alert_banner.setVisible(False)
        self.compliance_log.clear()
        self.final_report_view.setPlainText("Automated session in progress — final report will populate automatically.")
        self.fr_overall.setText("RUNNING")

        self.worker = PacketWorker(
            camera_id=self._camera,
            device=self._device,
            model_manager=self._shared_model_manager,
            pipeline_config=self._pipeline_config,
            frame_provider=self._frame_provider,
            parent=self,
            manual_roi=self._manual_roi,
            empty_reference=self._empty_reference,
        )
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.analysis_frame_ready.connect(self._on_detection_frame)
        self.worker.live_monitoring_updated.connect(self._on_live_monitoring)
        self.worker.compliance_updated.connect(self._on_compliance)
        self.worker.system_monitoring_updated.connect(self._on_system_monitoring)
        self.worker.alert_notification.connect(self._on_alert_raised)
        self.worker.alert_resolved.connect(self._on_alert_resolved)
        self.worker.activity_event.connect(self._on_activity_event)
        self.worker.final_report_ready.connect(self._on_final_report)
        self.worker.session_finished.connect(self._on_session_finished)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def stop_inspection(self) -> None:
        self.stop_btn.setEnabled(False)
        self.retry_btn.setEnabled(False)
        self.live_badge.setText("● STOPPING SAFELY...")
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
        self.restart_btn.setEnabled(True)

    def restart_inspection(self) -> None:
        """Safely stops any active inspection and immediately restarts a fresh session from Stage 1."""
        if self._manual_roi is None:
            return
        self.restart_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.live_badge.setText("● RESTARTING...")
        if self.worker is not None and self.worker.isRunning():
            self._restart_requested = True
            self.worker.stop()
            return

        # Reset telemetry & status displays to pristine Stage 1 state
        self.lm_stage.setText("01 · WAIT FOR EMPTY TESTBED")
        self.lm_packet_id.setText("UNRESOLVED")
        self.lm_active_model.setText("packet-seg.onnx")
        self.lm_status.setText("Restarting session from Stage 1...")
        self.lm_packet_det.setText("—")
        self.lm_cut_verify.setText("PENDING")
        self.lm_packet_cond.setText("UNKNOWN")
        self.lm_jewel_count.setText("0")
        self.cmp_observed.setText("Restarting...")
        self.cmp_expected.setText("Clear the testbed for empty check.")
        self.cmp_correct.setText("0 / 9")
        self.cmp_skipped.setText("0")
        self.instruction_banner.setText("Restarting inspection... Please clear the testbed.")

        # Start fresh session
        self.start_inspection()
        self.restart_btn.setEnabled(True)

    # ==============================================================
    # DISPLAY-ONLY SIGNAL SLOTS
    # ==============================================================
    def _on_frame_ready(self, image: QImage) -> None:
        self._last_pixmap = QPixmap.fromImage(image)
        self._render_pixmap()

    def _on_detection_frame(self, image: QImage, captured_at: float, metadata: Dict[str, Any]) -> None:
        hand, scissors = metadata.get('hand_detected'), metadata.get('scissor_detected')
        if (hand or scissors) and time.monotonic() - captured_at <= 1.5:
            self._detection_pixmap = QPixmap.fromImage(image)
            self._detection_time = captured_at
            self._detection_label = ('Hand + scissors' if hand and scissors else
                                     'Hand detected' if hand else 'Scissors detected')
        else:
            self._detection_pixmap = None
        self._render_pixmap()

    def _render_pixmap(self) -> None:
        if self._last_pixmap is None:
            return
        pixmap = self._last_pixmap.scaled(
            self.video.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if self._manual_roi is not None and not pixmap.isNull():
            x1, y1, x2, y2 = self._manual_roi
            sx = pixmap.width() / self._last_pixmap.width()
            sy = pixmap.height() / self._last_pixmap.height()
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor("#38bdf8"), 2))
            painter.drawRect(
                QRectF(x1 * sx, y1 * sy, (x2 - x1 - 1) * sx, (y2 - y1 - 1) * sy)
            )
            painter.end()
        if self._detection_pixmap is not None and self._detection_time is not None:
            age = time.monotonic() - self._detection_time
            if 0 <= age <= 1.5:
                width = min(240, max(140, int(pixmap.width() * .42)))
                height = min(175, int(pixmap.height() * .42))
                inset = self._detection_pixmap.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio,
                                                       Qt.TransformationMode.FastTransformation)
                painter = QPainter(pixmap)
                painter.fillRect(8, 8, inset.width() + 4, inset.height() + 34, QColor('#102432'))
                painter.drawPixmap(10, 40, inset)
                painter.setPen(QColor('#ecf6ff'))
                painter.setFont(QFont('Sans Serif', 8))
                painter.drawText(12, 21, self._detection_label)
                painter.drawText(12, 34, f'Last analysed frame · {age:.1f}s ago')
                painter.end()
        self.video.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._render_pixmap()

    def _on_live_monitoring(self, data: Dict[str, Any]) -> None:
        self.lm_stage.setText(str(data.get("stage", "—")))
        self.retry_btn.setEnabled(
            self.stop_btn.isEnabled() and self.lm_stage.text() in {
                stage.value for stage in AutoStage if stage not in {AutoStage.IDLE, AutoStage.STOPPED_OR_SAVED}
            }
        )
        if data.get("instruction"):
            self.instruction_banner.setText(str(data["instruction"]))
        self.lm_packet_id.setText(f"{data.get('packet_id', 'UNRESOLVED')} ({data.get('identity_status', 'UNRESOLVED')})")
        self.lm_active_model.setText(str(data.get("active_model", "—")))
        self.lm_status.setText(str(data.get("processing_status", "—")))
        self.lm_packet_det.setText(str(data.get("packet_detection", "—")))
        self.lm_cut_verify.setText(str(data.get("cutting_verification", "—")))
        self.lm_packet_cond.setText(str(data.get("packet_condition", "—")))
        self.lm_jewel_count.setText(str(data.get("jewellery_count", "0")))

    def _on_compliance(self, data: Dict[str, Any]) -> None:
        self.cmp_observed.setText(str(data.get("latest_observed_action", "—")))
        self.cmp_expected.setText(str(data.get("expected_procedure", "—")))
        self.cmp_correct.setText(f"{data.get('correct_actions', 0)} / {data.get('total_expected', 9)}")
        self.cmp_skipped.setText(
            f"Skipped: {data.get('skipped_actions', 0)} | Dev: {data.get('confirmed_deviations', 0)}"
        )

    def _on_system_monitoring(self, data: Dict[str, Any]) -> None:
        self.sys_model_load.setText(str(data.get("model_loading_status", "—")))
        self.sys_active_model.setText(str(data.get("active_inference_model", "—")))
        self.sys_camera.setText(str(data.get("camera_status", "—")))
        self.sys_errors.setText(str(data.get("processing_errors", "0")))
        self.sys_recovery.setText(str(data.get("automatic_recovery_status", "Nominal")))

    def _on_alert_raised(self, alert: Dict[str, Any]) -> None:
        sev = alert.get("severity", "WARNING")
        msg = alert.get("message", "")
        aid = alert.get("alert_id", "")
        self.alert_banner.setText(f"ALERT [{sev}] {msg}")
        self.alert_banner.setVisible(True)
        self.compliance_log.appendPlainText(f"[ALERT {aid} · {sev}] {msg}")

    def _on_alert_resolved(self, alert: Dict[str, Any]) -> None:
        aid = alert.get("alert_id", "")
        note = alert.get("correction_note", "Corrected")
        self.alert_banner.setVisible(False)
        self.compliance_log.appendPlainText(f"[CORRECTED {aid}] {note}")

    def _on_activity_event(self, ev: Dict[str, Any]) -> None:
        self.compliance_log.appendPlainText(
            f"[{ev.get('compliance_status')}] {ev.get('procedure_step')}: {ev.get('actual_action')}"
        )

    def _on_final_report(self, report: Dict[str, Any], paths: Dict[str, Optional[str]]) -> None:
        self._visual_report_path = paths.get("html")
        self.open_report_btn.setEnabled(bool(self._visual_report_path))
        self._pdf_report_path = paths.get("pdf")
        self.open_pdf_btn.setEnabled(bool(self._pdf_report_path))
        self._report_text = FinalReportGenerator.render_markdown(report)
        self.copy_report_btn.setEnabled(True)
        si = report.get("session_information", {})
        op = report.get("operational_results", {})
        comp = report.get("appraiser_compliance", {})
        fs = report.get("final_summary", {})
        devs = report.get("deviation_analysis", [])

        ocr = op.get("ocr_identification", {})
        cut = op.get("cutting_verification", {})
        pkt = op.get("packet_open_closed_inspection", {})
        jwl = op.get("jewellery_detection_and_counting", {})

        self.fr_packet.setText(f"{si.get('packet_id', 'UNRESOLVED')} ({si.get('packet_identity_status', 'UNRESOLVED')})")
        self.fr_cut.setText(f"{cut.get('state', 'UNVERIFIED')} [{op.get('cut_model_filename', 'cut-seg.pt')}]")
        mouth_status = pkt.get("mouth_status", pkt.get("status"))
        if mouth_status == "OPEN":
            gold_status = "VISIBLE" if pkt.get("gold_visible") else "NOT VISIBLE"
            self.fr_cond.setText(f"Mouth OPEN | Gold {gold_status}")
        elif mouth_status == "CLOSED":
            self.fr_cond.setText("Mouth CLOSED")
        else:
            self.fr_cond.setText("Mouth verification pending")
        self.fr_count.setText(
            f"{jwl.get('provisional_count', jwl.get('count', 0))} ({jwl.get('count_status', '—')})"
        )
        self.fr_compliance.setText(
            f"{comp.get('correctly_followed_procedures', 0)}/{comp.get('total_expected_procedures', 9)} OK · "
            f"Mistakes: {len(report.get('observed_mistakes', []))}"
        )
        stage_reached = si.get("stage_reached", "—")
        overall = "COMPLETED" if fs.get("packet_processing_completed") else f"SAVED ({stage_reached})"
        self.fr_overall.setText(overall)

        lines = [
            f"Stage Reached: {stage_reached} | Strip: {op.get('strip_result', '—')} | Cut Model: {op.get('cut_model_filename', '—')}",
            f"Summary: {fs.get('summary_text', '')}",
            f"Duration: {si.get('total_processing_duration_seconds', 0):.1f}s | OCR conf: {ocr.get('confidence', 0.0):.0%}",
        ]
        mistakes = report.get("observed_mistakes", [])
        unverified = report.get("camera_unverifiable_actions", [])
        if mistakes:
            lines.append("Observed Mistakes:")
            for m in mistakes:
                lines.append(f"  - [{m.get('code')}] {m.get('message')}")
        if unverified:
            lines.append("Camera-Unverifiable Actions:")
            for u in unverified:
                lines.append(f"  - [{u.get('code')}] {u.get('reason')}")
        if devs:
            lines.append("Procedure Deviations:")
            for d in devs:
                corr = "CORRECTED" if d.get("corrected") else "UNCORRECTED"
                lines.append(f"  - [{d.get('deviation_type')}] {d.get('code')} ({corr}): {d.get('actual_action')}")
        if op.get("missing_completion_event_rule"):
            lines.append(f"Completion Rule Note: {op['missing_completion_event_rule']}")
        if paths.get("json") or paths.get("markdown"):
            lines.append(f"Saved JSON: {paths.get('json') or '—'}")
            lines.append(f"Saved Markdown: {paths.get('markdown') or '—'}")
            lines.append(f"Visual Report: {paths.get('html') or '—'}")
            lines.append(f"PDF Report: {paths.get('pdf') or 'Unavailable'}")
        lines.extend(str(error) for error in report.get("report_export_errors", []))

        self.final_report_view.setPlainText("\n".join(lines))

    def _on_session_finished(self, status: str) -> None:
        self.live_badge.setText(f"● {status}")
        self.stop_btn.setEnabled(False)
        self.retry_btn.setEnabled(False)

    def _on_worker_finished(self) -> None:
        if self.sender() is not self.worker:
            return
        if self._restart_requested:
            self._restart_requested = False
            self.restart_inspection()
            return
        self.start_btn.setEnabled(self._manual_roi is not None)
        self.restart_btn.setEnabled(self._manual_roi is not None)
        self.check_btn.setEnabled(True)
        self.roi_btn.setEnabled(self._setup_image is not None)

    def _on_error(self, message: str) -> None:
        # Non-interactive error display (NEVER opens a modal QMessageBox!)
        self.sys_errors.setText(message)
        self.alert_banner.setText(f"SYSTEM ERROR: {message}")
        self.alert_banner.setVisible(True)
        self.compliance_log.appendPlainText(f"[SYSTEM ERROR] {message}")
        self.stop_btn.setEnabled(False)
        self.retry_btn.setEnabled(False)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._restart_requested = False
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            if not self.worker.wait(8000):
                event.ignore()
                return
        self._shared_model_manager.unload_all_models()
        event.accept()
