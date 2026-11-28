"""Separate PyQt6 Post-Cutting Inspection Panel & standalone launcher.

Matches the Embsys dark slate/emerald visual design while remaining completely
independent of the existing cutting UI (`packet_cut/desktop/widgets.py`).

Displays:
1. PACKET INFORMATION (Packet ID, Cutting status, State)
2. PACKET INSPECTION (Live camera preview, Packet ROI & Gold overlays, OPEN/CLOSED/UNKNOWN)
3. JEWELLERY COUNTING (Testbed ROI, individual boxes/masks, unique IDs, live count, uncertainty indicator)
4. FINAL RESULT (Packet ID, Packet condition, Gold visibility, Jewellery count, Inspection status)
5. Appraiser confirmation controls
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .inspection_controller import FrameProvider, InspectionController
from .utils.inspection_config import InspectionConfig


PANEL_STYLESHEET = """
QWidget#inspectionRoot { background:#0b1220; color:#e8eef8; }
QLabel { color:#e8eef8; }
QLabel#panelTitle { font-size:20px; font-weight:800; letter-spacing:0.8px; }
QLabel#panelSubTitle { color:#93a4bd; font-size:11px; }
QLabel#stateBadge { background:#12382f; color:#67e8b2; padding:6px 12px; border-radius:10px; font-weight:700; font-size:11px; }
QLabel#videoPreview { background:#050a12; border:1px solid #27364b; border-radius:12px; color:#8091a9; font-size:14px; }
QFrame#sectionCard { background:#121e30; border:1px solid #25364d; border-radius:11px; }
QLabel#sectionCaption { color:#8295b1; font-size:10px; font-weight:800; letter-spacing:0.9px; }
QLabel#fieldLabel { color:#93a4bd; font-size:11px; }
QLabel#fieldValue { color:#ffffff; font-size:14px; font-weight:800; }
QLabel#bigCount { color:#61e6ad; font-size:28px; font-weight:900; }
QLabel#uncertaintyBanner { background:#4b2218; border:1px solid #f59e0b; color:#fde68a; border-radius:8px; padding:6px 10px; font-weight:800; font-size:12px; }
QListWidget { background:#0d1726; border:1px solid #24344a; border-radius:8px; color:#dce7f5; font-size:11px; padding:4px; }
QPushButton { background:#1b2a40; border:1px solid #344963; border-radius:8px; padding:7px 11px; color:#e8eef8; font-weight:700; font-size:12px; }
QPushButton:hover { background:#263b58; }
QPushButton:disabled { color:#526176; background:#121b29; border-color:#202d3d; }
QPushButton#primaryBtn { background:#176b56; border:0; color:#ffffff; font-weight:800; }
QPushButton#primaryBtn:hover { background:#1c8469; }
QPushButton#dangerBtn { background:#632b38; border:0; color:#ffffff; }
QPushButton#dangerBtn:hover { background:#7a3545; }
QProgressBar { border:1px solid #293a50; border-radius:5px; text-align:center; background:#101a29; color:#dce8f8; font-size:11px; font-weight:700; }
QProgressBar::chunk { background:#22a982; border-radius:4px; }
"""


class PostCutInspectionPanel(QWidget):
    """Embeddable PyQt6 panel for Post-Cutting Packet Inspection & Jewellery Counting."""

    inspection_finished = pyqtSignal(dict)

    def __init__(
        self,
        config: Optional[InspectionConfig] = None,
        shared_models: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("inspectionRoot")
        self.config = config if config is not None else InspectionConfig.load()
        self.controller = InspectionController(self.config, shared_models=shared_models, parent=self)
        self._last_pixmap: Optional[QPixmap] = None
        self._build_ui()
        self._connect_signals()
        self.setStyleSheet(PANEL_STYLESHEET)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # Header
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("POST-CUTTING PACKET INSPECTION & JEWELLERY COUNTING")
        title.setObjectName("panelTitle")
        subtitle = QLabel("Stage 1: Gold Visibility (OPEN / CLOSED / UNKNOWN)  ·  Stage 2: Testbed Jewellery Counting")
        subtitle.setObjectName("panelSubTitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.state_badge = QLabel("● IDLE")
        self.state_badge.setObjectName("stateBadge")
        header.addWidget(self.state_badge)
        root.addLayout(header)

        # Main split: Left live video preview, Right structured status cards
        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, 1)

        # Left column: Live Camera Preview + Stage Message
        left = QVBoxLayout()
        left.setSpacing(6)
        self.video_label = QLabel("Post-cutting live inspection feed will appear here")
        self.video_label.setObjectName("videoPreview")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left.addWidget(self.video_label, 1)

        self.live_note = QLabel("Waiting for CUT_CONFIRMED activation event…")
        self.live_note.setObjectName("panelSubTitle")
        self.live_note.setWordWrap(True)
        left.addWidget(self.live_note)
        body.addLayout(left, 7)

        # Right column: 4 required panels + controls
        right = QVBoxLayout()
        right.setSpacing(8)
        body.addLayout(right, 5)

        # 1. PACKET INFORMATION
        pkt_info_card = self._make_card("PACKET INFORMATION")
        pi_grid = QGridLayout()
        pi_grid.setSpacing(6)
        self.pkt_id_val = QLabel("—")
        self.pkt_id_val.setObjectName("fieldValue")
        self.cut_status_val = QLabel("—")
        self.cut_status_val.setObjectName("fieldValue")
        pi_grid.addWidget(self._field_label("Packet ID"), 0, 0)
        pi_grid.addWidget(self.pkt_id_val, 1, 0)
        pi_grid.addWidget(self._field_label("Cutting Status"), 0, 1)
        pi_grid.addWidget(self.cut_status_val, 1, 1)
        pkt_info_card.layout().addLayout(pi_grid)
        right.addWidget(pkt_info_card)

        # 2. PACKET INSPECTION (Stage 1)
        stage1_card = self._make_card("PACKET INSPECTION (STAGE 1)")
        s1_grid = QGridLayout()
        s1_grid.setSpacing(6)
        self.pkt_status_val = QLabel("UNKNOWN")
        self.pkt_status_val.setObjectName("fieldValue")
        self.gold_vis_val = QLabel("—")
        self.gold_vis_val.setObjectName("fieldValue")
        self.s1_conf_val = QLabel("0%")
        self.s1_conf_val.setObjectName("fieldValue")
        self.s1_detail_val = QLabel("Hold cut packet vertically under overhead camera.")
        self.s1_detail_val.setObjectName("fieldLabel")
        self.s1_detail_val.setWordWrap(True)
        s1_grid.addWidget(self._field_label("Packet Status"), 0, 0)
        s1_grid.addWidget(self.pkt_status_val, 1, 0)
        s1_grid.addWidget(self._field_label("Gold Visibility"), 0, 1)
        s1_grid.addWidget(self.gold_vis_val, 1, 1)
        s1_grid.addWidget(self._field_label("Confidence"), 0, 2)
        s1_grid.addWidget(self.s1_conf_val, 1, 2)
        stage1_card.layout().addLayout(s1_grid)
        stage1_card.layout().addWidget(self.s1_detail_val)
        right.addWidget(stage1_card)

        # 3. JEWELLERY COUNTING (Stage 2)
        stage2_card = self._make_card("JEWELLERY COUNTING (STAGE 2 — GREEN TESTBED)")
        s2_top = QHBoxLayout()
        count_col = QVBoxLayout()
        count_col.addWidget(self._field_label("Live Count"))
        self.live_count_val = QLabel("0")
        self.live_count_val.setObjectName("bigCount")
        count_col.addWidget(self.live_count_val)
        s2_top.addLayout(count_col, 1)

        status_col = QVBoxLayout()
        status_col.addWidget(self._field_label("Count Status"))
        self.count_status_val = QLabel("NOT_STARTED")
        self.count_status_val.setObjectName("fieldValue")
        status_col.addWidget(self.count_status_val)
        self.stability_bar = QProgressBar()
        self.stability_bar.setRange(0, 100)
        self.stability_bar.setValue(0)
        self.stability_bar.setFormat("Placement stability: %p%")
        self.stability_bar.setFixedHeight(18)
        status_col.addWidget(self.stability_bar)
        s2_top.addLayout(status_col, 2)
        stage2_card.layout().addLayout(s2_top)

        self.uncertainty_banner = QLabel("⚠ COUNT_UNCERTAIN — Separate overlapping jewellery on testbed")
        self.uncertainty_banner.setObjectName("uncertaintyBanner")
        self.uncertainty_banner.setWordWrap(True)
        self.uncertainty_banner.setVisible(False)
        stage2_card.layout().addWidget(self.uncertainty_banner)

        self.articles_list = QListWidget()
        self.articles_list.setMaximumHeight(95)
        stage2_card.layout().addWidget(self.articles_list)
        right.addWidget(stage2_card)

        # 4. FINAL RESULT
        final_card = self._make_card("FINAL RESULT")
        fr_grid = QGridLayout()
        fr_grid.setSpacing(6)
        self.final_pkt_val = QLabel("—")
        self.final_pkt_val.setObjectName("fieldValue")
        self.final_cond_val = QLabel("UNKNOWN")
        self.final_cond_val.setObjectName("fieldValue")
        self.final_gold_val = QLabel("—")
        self.final_gold_val.setObjectName("fieldValue")
        self.final_count_val = QLabel("—")
        self.final_count_val.setObjectName("fieldValue")
        self.final_status_val = QLabel("IN_PROGRESS")
        self.final_status_val.setObjectName("fieldValue")

        fr_grid.addWidget(self._field_label("Packet ID"), 0, 0)
        fr_grid.addWidget(self.final_pkt_val, 1, 0)
        fr_grid.addWidget(self._field_label("Packet Condition"), 0, 1)
        fr_grid.addWidget(self.final_cond_val, 1, 1)
        fr_grid.addWidget(self._field_label("Gold Visibility"), 0, 2)
        fr_grid.addWidget(self.final_gold_val, 1, 2)
        fr_grid.addWidget(self._field_label("Jewellery Count"), 2, 0)
        fr_grid.addWidget(self.final_count_val, 3, 0)
        fr_grid.addWidget(self._field_label("Inspection Status"), 2, 1, 1, 2)
        fr_grid.addWidget(self.final_status_val, 3, 1, 1, 2)
        final_card.layout().addLayout(fr_grid)
        right.addWidget(final_card)

        # 5. APPRAISER CONTROLS
        controls_card = QFrame()
        controls_card.setObjectName("sectionCard")
        cg = QGridLayout(controls_card)
        cg.setContentsMargins(10, 8, 10, 8)
        cg.setSpacing(6)

        self.proceed_stage2_btn = QPushButton("Proceed to Jewellery Counting →")
        self.proceed_stage2_btn.setObjectName("primaryBtn")
        self.proceed_stage2_btn.setEnabled(False)

        self.confirm_count_btn = QPushButton("✓ Confirm Final Count")
        self.confirm_count_btn.setObjectName("primaryBtn")
        self.confirm_count_btn.setEnabled(False)

        self.retry_s1_btn = QPushButton("Retry Packet Check")
        self.retry_s2_btn = QPushButton("Recount Jewellery")
        self.calibrate_roi_btn = QPushButton("Auto-Detect Green ROI")
        self.export_btn = QPushButton("Export Result JSON")
        self.stop_btn = QPushButton("■ Stop Inspection")
        self.stop_btn.setObjectName("dangerBtn")

        cg.addWidget(self.proceed_stage2_btn, 0, 0, 1, 2)
        cg.addWidget(self.confirm_count_btn, 1, 0, 1, 2)
        cg.addWidget(self.retry_s1_btn, 2, 0)
        cg.addWidget(self.retry_s2_btn, 2, 1)
        cg.addWidget(self.calibrate_roi_btn, 3, 0)
        cg.addWidget(self.export_btn, 3, 1)
        cg.addWidget(self.stop_btn, 4, 0, 1, 2)
        right.addWidget(controls_card)

    @staticmethod
    def _make_card(caption_text: str) -> QFrame:
        card = QFrame()
        card.setObjectName("sectionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        cap = QLabel(caption_text)
        cap.setObjectName("sectionCaption")
        layout.addWidget(cap)
        return card

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("fieldLabel")
        return lbl

    def _connect_signals(self) -> None:
        self.controller.frame_ready.connect(self._on_frame_ready)
        self.controller.state_changed.connect(self._on_state_changed)
        self.controller.packet_status_updated.connect(self._on_packet_status_updated)
        self.controller.jewellery_count_updated.connect(self._on_jewellery_count_updated)
        self.controller.result_updated.connect(self._on_result_updated)
        self.controller.inspection_completed.connect(self._on_inspection_completed)
        self.controller.error_occurred.connect(self._on_error)

        self.proceed_stage2_btn.clicked.connect(lambda: self.controller.proceed_to_jewellery_counting(True))
        self.confirm_count_btn.clicked.connect(self._on_confirm_clicked)
        self.retry_s1_btn.clicked.connect(self.controller.retry_packet_inspection)
        self.retry_s2_btn.clicked.connect(lambda: self.controller.retry_jewellery_counting(False))
        self.calibrate_roi_btn.clicked.connect(self.controller.auto_calibrate_testbed_roi)
        self.export_btn.clicked.connect(self._on_export_clicked)
        self.stop_btn.clicked.connect(self.stop_inspection)

    def start_inspection(
        self,
        packet_id: str,
        cut_status: str = "CUTTED",
        camera_source: Optional[Union[int, str]] = None,
        frame_provider: Optional[FrameProvider] = None,
        existing_capture: Optional[Any] = None,
        shared_models: Optional[Dict[str, Any]] = None,
        auto_advance_to_stage2: bool = False,
    ) -> None:
        self.pkt_id_val.setText(str(packet_id))
        self.cut_status_val.setText(str(cut_status))
        self.final_pkt_val.setText(str(packet_id))
        self.controller.start_on_cut_confirmed(
            packet_id=packet_id,
            cut_status=cut_status,
            camera_source=camera_source,
            frame_provider=frame_provider,
            existing_capture=existing_capture,
            shared_models=shared_models,
            auto_advance_to_stage2=auto_advance_to_stage2,
        )

    def stop_inspection(self) -> None:
        self.controller.stop()
        self.state_badge.setText("● STOPPED")
        self.live_note.setText("Post-cutting inspection stopped.")

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

    def _on_state_changed(self, state_name: str, message: str) -> None:
        self.state_badge.setText(f"● {state_name}")
        self.live_note.setText(message)
        can_proceed = state_name in ("PACKET_OPEN", "PACKET_CLOSED")
        self.proceed_stage2_btn.setEnabled(can_proceed)

    def _on_packet_status_updated(self, data: Dict[str, Any]) -> None:
        status = str(data.get("status", "UNKNOWN"))
        gold_vis = data.get("gold_visible")
        conf = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", ""))

        self.pkt_status_val.setText(status)
        color = {"OPEN": "#34d399", "CLOSED": "#38bdf8", "UNKNOWN": "#fbbf24"}.get(status, "#fbbf24")
        self.pkt_status_val.setStyleSheet(f"color:{color}; font-size:16px; font-weight:900;")

        gold_text = "YES" if gold_vis is True else ("NO" if gold_vis is False else "UNKNOWN")
        self.gold_vis_val.setText(gold_text)
        self.s1_conf_val.setText(f"{conf:.0%}")
        self.s1_detail_val.setText(reason)

        self.final_cond_val.setText(status)
        self.final_gold_val.setText(gold_text)

    def _on_jewellery_count_updated(self, data: Dict[str, Any]) -> None:
        count_status = str(data.get("count_status", "NOT_STARTED"))
        live_count = int(data.get("live_count", 0))
        final_count = data.get("count")
        stable = int(data.get("stable_frames", 0))
        req = max(1, int(data.get("required_stable_frames", 8)))
        pct = min(100, int(round((stable / float(req)) * 100.0)))

        self.count_status_val.setText(count_status)
        self.stability_bar.setValue(pct)

        is_uncertain = count_status == "COUNT_UNCERTAIN"
        self.uncertainty_banner.setVisible(is_uncertain)
        if is_uncertain:
            self.live_count_val.setText("?")
            self.live_count_val.setStyleSheet("color:#f59e0b; font-size:28px; font-weight:900;")
            self.uncertainty_banner.setText(f"⚠ {data.get('reason', 'COUNT_UNCERTAIN')}")
            self.confirm_count_btn.setEnabled(False)
            self.final_count_val.setText("COUNT_UNCERTAIN")
        else:
            shown = final_count if final_count is not None else live_count
            self.live_count_val.setText(str(shown))
            self.live_count_val.setStyleSheet("color:#61e6ad; font-size:28px; font-weight:900;")
            self.confirm_count_btn.setEnabled(count_status == "PENDING_CONFIRMATION")
            self.final_count_val.setText(str(shown) if final_count is not None else f"{live_count} (live)")

        self.articles_list.clear()
        for art in data.get("articles", []):
            tag = " [SPLIT]" if art.get("separated_from_cluster") else ""
            if art.get("uncertain"):
                tag += " [UNCERTAIN]"
            self.articles_list.addItem(
                f"{art['article_id']}  |  conf={art['confidence']:.2f}  |  area={art['area_px']}px{tag}"
            )

    def _on_result_updated(self, data: Dict[str, Any]) -> None:
        self.pkt_id_val.setText(str(data.get("packet_id", "—")))
        self.cut_status_val.setText(str(data.get("cut_status", "—")))
        self.final_pkt_val.setText(str(data.get("packet_id", "—")))
        insp_status = str(data.get("inspection_status", "IN_PROGRESS"))
        self.final_status_val.setText(insp_status)
        color = {
            "COMPLETED": "#34d399",
            "AWAITING_CONFIRMATION": "#38bdf8",
            "UNCERTAIN": "#f87171",
        }.get(insp_status, "#e8eef8")
        self.final_status_val.setStyleSheet(f"color:{color}; font-size:15px; font-weight:900;")

    def _on_confirm_clicked(self) -> None:
        self.controller.confirm_final_count()

    def _on_inspection_completed(self, data: Dict[str, Any]) -> None:
        self.confirm_count_btn.setEnabled(False)
        self.proceed_stage2_btn.setEnabled(False)
        self.inspection_finished.emit(data)

    def _on_export_clicked(self) -> None:
        data = self.controller.last_result
        if not data:
            QMessageBox.information(self, "Export Result", "No inspection result available yet.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Post-Cutting Inspection Result",
            f"{data.get('packet_id', 'inspection')}_result.json",
            "JSON Files (*.json)",
        )
        if file_path:
            Path(file_path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _on_error(self, message: str) -> None:
        self.live_note.setText(f"ERROR: {message}")
        QMessageBox.warning(self, "Post-Cutting Inspection Warning", message)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.controller.stop()
        event.accept()


class PostCutInspectionWindow(QMainWindow):
    """Standalone window hosting `PostCutInspectionPanel` for independent runs/tests."""

    def __init__(
        self,
        packet_id: str = "PKT-TEST",
        camera_source: Union[int, str] = 0,
        config: Optional[InspectionConfig] = None,
        auto_advance_to_stage2: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"Embsys · Post-Cutting Inspection ({packet_id})")
        self.resize(1280, 760)
        self.setMinimumSize(980, 620)
        self.panel = PostCutInspectionPanel(config=config, parent=self)
        self.setCentralWidget(self.panel)
        self._packet_id = packet_id
        self._camera_source = camera_source
        self._auto_advance = auto_advance_to_stage2

    def launch(self) -> None:
        self.show()
        self.panel.start_inspection(
            packet_id=self._packet_id,
            cut_status="CUTTED",
            camera_source=self._camera_source,
            auto_advance_to_stage2=self._auto_advance,
        )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.panel.stop_inspection()
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Post-Cutting Inspection UI")
    parser.add_argument("--camera", default="0", help="Camera index or video path")
    parser.add_argument("--packet-id", default="PKT-1025", help="Packet ID from upstream OCR")
    parser.add_argument("--config", default=None, help="Path to inspection config.json")
    parser.add_argument("--auto-stage2", action="store_true", help="Auto-advance from Stage 1 to Stage 2")
    args = parser.parse_args()

    cam_src: Union[int, str] = int(args.camera) if str(args.camera).isdigit() else str(args.camera)
    cfg = InspectionConfig.load(args.config) if args.config else InspectionConfig.load()

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = PostCutInspectionWindow(
        packet_id=args.packet_id,
        camera_source=cam_src,
        config=cfg,
        auto_advance_to_stage2=args.auto_stage2,
    )
    win.launch()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

