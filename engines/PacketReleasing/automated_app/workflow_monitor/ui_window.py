"""Single-START PyQt6 Desktop Interface for Automated Workflow Monitoring & Compliance.

Click START once — no manual stage-progression or confirmation buttons required.
Displays:
- Live camera preview with real-time overlays & alert HUD
- Automatic stage progression tracker & OCR packet identity status
- Real-Time Alerts panel (active vs corrected alerts)
- Live Appraiser Compliance & Activity Log feed
- Automatic Final Inspection & Compliance Report viewer + JSON/Markdown export
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from inspection.utils.inspection_config import InspectionConfig

from .config import WorkflowMonitorConfig
from .controller import AutomatedWorkflowController, FrameProvider
from .report_generator import FinalReportGenerator


WINDOW_STYLESHEET = """
QWidget#root { background:#0b1220; color:#e8eef8; }
QLabel { color:#e8eef8; }
QLabel#appTitle { font-size:21px; font-weight:800; letter-spacing:0.8px; }
QLabel#appSub { color:#93a4bd; font-size:11px; }
QLabel#stageBadge { background:#12382f; color:#67e8b2; padding:6px 12px; border-radius:10px; font-weight:800; font-size:11px; }
QLabel#videoFeed { background:#050a12; border:1px solid #27364b; border-radius:12px; color:#8091a9; font-size:14px; }
QFrame#card { background:#121e30; border:1px solid #25364d; border-radius:11px; }
QLabel#cardCap { color:#8295b1; font-size:10px; font-weight:800; letter-spacing:0.9px; }
QLabel#kpiLabel { color:#93a4bd; font-size:11px; }
QLabel#kpiVal { color:#ffffff; font-size:15px; font-weight:800; }
QLabel#alertBanner { background:#4b1d24; border:1px solid #f87171; color:#fecaca; border-radius:9px; padding:8px 12px; font-weight:800; font-size:12px; }
QListWidget, QTextEdit { background:#0d1726; border:1px solid #24344a; border-radius:8px; color:#dce7f5; font-size:11px; padding:6px; }
QTabWidget::pane { border:1px solid #25364d; border-radius:8px; background:#121e30; }
QTabBar::tab { background:#162438; color:#9fb2cc; padding:7px 14px; border-top-left-radius:7px; border-top-right-radius:7px; font-weight:700; font-size:11px; }
QTabBar::tab:selected { background:#1d334f; color:#61e6ad; }
QPushButton { background:#1b2a40; border:1px solid #344963; border-radius:8px; padding:8px 14px; color:#e8eef8; font-weight:700; font-size:12px; }
QPushButton:hover { background:#263b58; }
QPushButton:disabled { color:#526176; background:#121b29; border-color:#202d3d; }
QPushButton#startBtn { background:#176b56; border:2px solid #34d399; color:#ffffff; font-size:14px; font-weight:900; }
QPushButton#startBtn:hover { background:#1c8469; }
QPushButton#restartBtn { background:#1e3a5f; border:2px solid #38bdf8; color:#ffffff; font-size:14px; font-weight:900; }
QPushButton#restartBtn:hover { background:#254877; }
QPushButton#restartBtn:disabled { background:#162436; border-color:#2a4563; color:#5c708a; }
QPushButton#stopBtn { background:#632b38; border:0; color:#ffffff; font-weight:800; }
QPushButton#stopBtn:hover { background:#7a3545; }
"""


class AutomatedMonitoringWindow(QMainWindow):
    """Single-START automated appraiser workflow monitoring & compliance window."""

    def __init__(
        self,
        camera_source: Union[int, str] = 0,
        workflow_config: Optional[WorkflowMonitorConfig] = None,
        inspection_config: Optional[InspectionConfig] = None,
        preloaded_models: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("EMBSYS Packet-Cut · Automated Appraiser Workflow Monitor")
        self.resize(1380, 820)
        self.setMinimumSize(1080, 660)

        self.controller = AutomatedWorkflowController(
            camera_source=camera_source,
            workflow_config=workflow_config,
            inspection_config=inspection_config,
            preloaded_models=preloaded_models,
            parent=self,
        )
        self._last_pixmap: Optional[QPixmap] = None
        self._build_ui()
        self._connect_signals()
        self.setStyleSheet(WINDOW_STYLESHEET)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(10)

        # Header + Single START Control
        header = QHBoxLayout()
        tbox = QVBoxLayout()
        title = QLabel("EMBSYS PACKET-CUT · AUTOMATED WORKFLOW & COMPLIANCE MONITOR")
        title.setObjectName("appTitle")
        sub = QLabel("Click START once — automated packet processing, OCR protection, procedure monitoring, alerts & final compliance report")
        sub.setObjectName("appSub")
        tbox.addWidget(title)
        tbox.addWidget(sub)
        header.addLayout(tbox, 1)

        self.stage_badge = QLabel("● IDLE")
        self.stage_badge.setObjectName("stageBadge")
        header.addWidget(self.stage_badge)

        self.start_btn = QPushButton("▶  START")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setMinimumWidth(130)
        header.addWidget(self.start_btn)

        self.restart_btn = QPushButton("↺  RESTART")
        self.restart_btn.setObjectName("restartBtn")
        self.restart_btn.setMinimumWidth(130)
        header.addWidget(self.restart_btn)

        self.stop_btn = QPushButton("■  End Session")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        header.addWidget(self.stop_btn)
        outer.addLayout(header)

        # Live Real-Time Alert Banner (hidden when no active alert)
        self.alert_banner = QLabel("")
        self.alert_banner.setObjectName("alertBanner")
        self.alert_banner.setWordWrap(True)
        self.alert_banner.setVisible(False)
        outer.addWidget(self.alert_banner)

        # Main split
        body = QHBoxLayout()
        body.setSpacing(12)
        outer.addLayout(body, 1)

        # Left: Live Camera Feed + Status
        left = QVBoxLayout()
        left.setSpacing(6)
        self.video_label = QLabel("Click START to begin fully automated camera monitoring")
        self.video_label.setObjectName("videoFeed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(540, 400)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left.addWidget(self.video_label, 1)

        self.live_status_label = QLabel("System ready. Only one click (START) is required.")
        self.live_status_label.setObjectName("appSub")
        self.live_status_label.setWordWrap(True)
        left.addWidget(self.live_status_label)
        body.addLayout(left, 7)

        # Right: KPI Summary + Tabbed Live Monitor & Automatic Report
        right = QVBoxLayout()
        right.setSpacing(8)
        body.addLayout(right, 5)

        # Operational + Compliance KPI Card
        kpi_card = QFrame()
        kpi_card.setObjectName("card")
        kg = QGridLayout(kpi_card)
        kg.setContentsMargins(12, 8, 12, 8)
        kg.setSpacing(6)

        self.session_val = QLabel("—")
        self.packet_id_val = QLabel("UNRESOLVED")
        self.cut_val = QLabel("NOT_STARTED")
        self.postcut_val = QLabel("UNKNOWN")
        self.count_val = QLabel("—")
        self.compliance_val = QLabel("0 / 9")
        for lbl in (self.session_val, self.packet_id_val, self.cut_val, self.postcut_val, self.count_val, self.compliance_val):
            lbl.setObjectName("kpiVal")

        kg.addWidget(self._kpi_label("SESSION ID"), 0, 0)
        kg.addWidget(self.session_val, 1, 0)
        kg.addWidget(self._kpi_label("PACKET ID (OCR)"), 0, 1)
        kg.addWidget(self.packet_id_val, 1, 1)
        kg.addWidget(self._kpi_label("CUT STATUS"), 0, 2)
        kg.addWidget(self.cut_val, 1, 2)
        kg.addWidget(self._kpi_label("PACKET OPEN/CLOSED"), 2, 0)
        kg.addWidget(self.postcut_val, 3, 0)
        kg.addWidget(self._kpi_label("JEWELLERY COUNT"), 2, 1)
        kg.addWidget(self.count_val, 3, 1)
        kg.addWidget(self._kpi_label("COMPLIANCE"), 2, 2)
        kg.addWidget(self.compliance_val, 3, 2)
        right.addWidget(kpi_card)

        # Tabs: Alerts & Procedure | Activity Log | Final Report
        self.tabs = QTabWidget()

        # Tab 1: Real-Time Alerts + Prescribed Procedure Checklist
        tab_monitor = QWidget()
        tm_lay = QVBoxLayout(tab_monitor)
        tm_lay.setContentsMargins(8, 8, 8, 8)
        tm_lay.setSpacing(6)
        tm_lay.addWidget(self._caption("REAL-TIME PROCEDURAL ALERTS"))
        self.alerts_list = QListWidget()
        self.alerts_list.setMaximumHeight(175)
        tm_lay.addWidget(self.alerts_list)
        tm_lay.addWidget(self._caption("PRESCRIBED PROCEDURE COMPLIANCE"))
        self.steps_list = QListWidget()
        tm_lay.addWidget(self.steps_list, 1)
        self.tabs.addTab(tab_monitor, "Alerts & Compliance")

        # Tab 2: Complete Chronological Activity Log
        tab_log = QWidget()
        tl_lay = QVBoxLayout(tab_log)
        tl_lay.setContentsMargins(8, 8, 8, 8)
        tl_lay.addWidget(self._caption("CHRONOLOGICAL APPRAISER ACTIVITY LOG"))
        self.activity_list = QListWidget()
        tl_lay.addWidget(self.activity_list, 1)
        self.tabs.addTab(tab_log, "Activity Log")

        # Tab 3: Automatic Final Report Viewer
        tab_report = QWidget()
        tr_lay = QVBoxLayout(tab_report)
        tr_lay.setContentsMargins(8, 8, 8, 8)
        tr_lay.setSpacing(6)
        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setPlainText("Final inspection & compliance report will be generated automatically when the session finishes.")
        tr_lay.addWidget(self.report_view, 1)
        self.export_report_btn = QPushButton("Export Report (JSON)")
        self.export_report_btn.setEnabled(False)
        tr_lay.addWidget(self.export_report_btn)
        self.tabs.addTab(tab_report, "Final Report")

        right.addWidget(self.tabs, 1)

    @staticmethod
    def _kpi_label(text: str) -> QLabel:
        l = QLabel(text)
        l.setObjectName("kpiLabel")
        return l

    @staticmethod
    def _caption(text: str) -> QLabel:
        l = QLabel(text)
        l.setObjectName("cardCap")
        return l

    def _connect_signals(self) -> None:
        self.start_btn.clicked.connect(lambda: self.start_monitoring())
        self.restart_btn.clicked.connect(self.restart_monitoring)
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.export_report_btn.clicked.connect(self._export_report)

        self.controller.session_started.connect(self._on_session_started)
        self.controller.frame_ready.connect(self._on_frame_ready)
        self.controller.stage_updated.connect(self._on_stage_updated)
        self.controller.alert_raised.connect(self._on_alert_raised)
        self.controller.alert_resolved.connect(self._on_alert_resolved)
        self.controller.activity_logged.connect(self._on_activity_logged)
        self.controller.compliance_updated.connect(self._on_compliance_updated)
        self.controller.operational_updated.connect(self._on_operational_updated)
        self.controller.report_ready.connect(self._on_report_ready)
        self.controller.error_occurred.connect(self._on_error)

    def start_monitoring(self, frame_provider: Optional[FrameProvider] = None) -> None:
        self.alerts_list.clear()
        self.activity_list.clear()
        self.steps_list.clear()
        self.alert_banner.setVisible(False)
        self.export_report_btn.setEnabled(False)
        self.report_view.setPlainText("Session running automatically — report will populate upon completion.")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.tabs.setCurrentIndex(0)
        self.controller.start(frame_provider=frame_provider)

    def stop_monitoring(self) -> None:
        self.stop_btn.setEnabled(False)
        self.live_status_label.setText("Ending session and generating final compliance report…")
        self.controller.stop()
        self.start_btn.setEnabled(True)
        self.restart_btn.setEnabled(True)

    def restart_monitoring(self) -> None:
        """Safely ends any ongoing workflow monitor session and starts a new session."""
        self.restart_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stage_badge.setText("● RESTARTING...")
        self.controller.stop()
        self.start_monitoring()
        self.restart_btn.setEnabled(True)

    def _on_session_started(self, session_id: str) -> None:
        self.session_val.setText(session_id)
        self.stage_badge.setText("● MONITORING ACTIVE")

    def _on_frame_ready(self, image: QImage) -> None:
        self._last_pixmap = QPixmap.fromImage(image)
        self._render_pixmap()

    def _render_pixmap(self) -> None:
        if self._last_pixmap is None:
            return
        self.video_label.setPixmap(
            self._last_pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._render_pixmap()

    def _on_stage_updated(self, stage: str, packet_id: str, identity_status: str, message: str) -> None:
        self.stage_badge.setText(f"● {stage}")
        self.packet_id_val.setText(f"{packet_id} ({identity_status})")
        color = "#34d399" if identity_status in ("RESOLVED", "RECOVERED") else "#fbbf24"
        self.packet_id_val.setStyleSheet(f"color:{color}; font-size:14px; font-weight:800;")
        self.live_status_label.setText(message)

    def _on_alert_raised(self, alert: Dict[str, Any]) -> None:
        sev = alert.get("severity", "WARNING")
        msg = alert.get("message", "")
        aid = alert.get("alert_id", "")
        self.alert_banner.setText(f"⚠ [{sev}] {msg}")
        self.alert_banner.setVisible(True)
        item = QListWidgetItem(f"[{aid}] [{sev}] {msg}")
        self.alerts_list.insertItem(0, item)

    def _on_alert_resolved(self, alert: Dict[str, Any]) -> None:
        aid = alert.get("alert_id", "")
        note = alert.get("correction_note", "Corrected")
        self.alerts_list.insertItem(0, QListWidgetItem(f"✓ [{aid}] CORRECTED — {note}"))
        self.alert_banner.setVisible(False)

    def _on_activity_logged(self, ev: Dict[str, Any]) -> None:
        line = (
            f"[{ev.get('compliance_status')}] {ev.get('procedure_step')} — "
            f"{ev.get('actual_action')}"
        )
        self.activity_list.insertItem(0, QListWidgetItem(line))

    def _on_compliance_updated(self, comp: Dict[str, Any]) -> None:
        ok = comp.get("correctly_followed_procedures", 0)
        tot = comp.get("total_expected_procedures", 9)
        devs = comp.get("confirmed_deviations", 0)
        self.compliance_val.setText(f"{ok}/{tot}  (Dev: {devs})")

        self.steps_list.clear()
        for st in comp.get("procedure_steps", []):
            self.steps_list.addItem(
                f"{st['order']}. [{st['status']}] {st['title']}"
            )

    def _on_operational_updated(self, op: Dict[str, Any]) -> None:
        cut = op.get("cut", {})
        pkt = op.get("post_cut_packet", {})
        jwl = op.get("jewellery", {})

        self.cut_val.setText(str(cut.get("state", "NOT_STARTED")))
        self.postcut_val.setText(str(pkt.get("status", "UNKNOWN")))
        j_status = str(jwl.get("count_status", "NOT_STARTED"))
        if j_status == "COUNT_UNCERTAIN":
            self.count_val.setText("COUNT_UNCERTAIN")
        elif jwl.get("count") is not None:
            self.count_val.setText(f"{jwl['count']} ({j_status})")
        else:
            self.count_val.setText(f"{jwl.get('live_count', 0)} (live)")

    def _on_report_ready(self, report: Dict[str, Any], paths: Dict[str, Optional[str]]) -> None:
        md = FinalReportGenerator.render_markdown(report)
        if paths.get("json") or paths.get("markdown"):
            md += f"\n---\nSaved JSON: `{paths.get('json')}`\nSaved Markdown: `{paths.get('markdown')}`\n"
        self.report_view.setPlainText(md)
        self.export_report_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.restart_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stage_badge.setText("● REPORT READY")
        self.tabs.setCurrentIndex(2)

    def _export_report(self) -> None:
        if not self.controller.last_report:
            return
        sid = self.controller.last_report["session_information"]["session_id"]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Compliance & Inspection Report",
            f"{sid}_report.json",
            "JSON Files (*.json)",
        )
        if path:
            Path(path).write_text(json.dumps(self.controller.last_report, indent=2), encoding="utf-8")

    def _on_error(self, msg: str) -> None:
        self.live_status_label.setText(f"ERROR: {msg}")
        self.start_btn.setEnabled(True)
        self.restart_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.warning(self, "Automated Monitor Warning", msg)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.controller.stop()
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description="EMBSYS Packet-Cut Automated Workflow & Compliance Monitor")
    parser.add_argument("--camera", default="0", help="Camera index (default: 0) or video file path")
    parser.add_argument("--config", default=None, help="Optional workflow_monitor config.json path")
    parser.add_argument("--inspection-config", default=None, help="Optional inspection config.json path")
    args = parser.parse_args()

    cam_src: Union[int, str] = int(args.camera) if str(args.camera).isdigit() else str(args.camera)
    wf_cfg = WorkflowMonitorConfig.load(args.config) if args.config else WorkflowMonitorConfig.load()
    insp_cfg = InspectionConfig.load(args.inspection_config) if args.inspection_config else InspectionConfig.load()

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = AutomatedMonitoringWindow(
        camera_source=cam_src,
        workflow_config=wf_cfg,
        inspection_config=insp_cfg,
    )
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

