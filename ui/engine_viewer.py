#!/usr/bin/env python3
"""Existing engine UI as a read-only viewer in its own process; closing it cannot stop surveillance."""

import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import JEWEL, PACKET, RELEASING


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine", choices=("purity", "packet", "releasing"), required=True
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.engine == "releasing":
        sys.path[:0] = [str(RELEASING / "automated_app"), str(RELEASING), str(ROOT)]
        from PyQt6 import QtCore, QtGui, QtWidgets
        from packet_cut.desktop.widgets import MainWindow
        from PyQt6.QtCore import QLibraryInfo

        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.path(
            QLibraryInfo.LibraryPath.PluginsPath
        )
        application = QtWidgets.QApplication([])
        window = MainWindow()
    else:
        sys.path[:0] = [str(JEWEL), str(PACKET), str(ROOT)]
        from PyQt5 import QtCore, QtGui, QtWidgets
        import jewel_tracker_5
        from PyQt5.QtCore import QLibraryInfo

        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
            QLibraryInfo.PluginsPath
        )
        application = QtWidgets.QApplication([])
        if args.engine == "purity":
            window = jewel_tracker_5.MainWindow(auto_connect=False)
            window.rub_timer.stop()
            window.packet_stage.timer.stop()
            window.preview_timer.stop()
            from ui.purity_diagnostics import PurityView

            purity_view = PurityView(window, jewel_tracker_5)
        else:
            from engines.sealing.workflow import MainWindow

            window = MainWindow(auto_connect=False)
            window.preview_timer.stop()
    window.setWindowTitle(
        window.windowTitle() + " · Controller diagnostics (read only)"
    )
    for kind in (
        QtWidgets.QAbstractButton,
        QtWidgets.QAbstractSpinBox,
        QtWidgets.QComboBox,
        QtWidgets.QLineEdit,
    ):
        for control in window.findChildren(kind):
            control.blockSignals(True)
            control.setEnabled(False)

    def refresh():
        if (args.directory / "closed.json").exists():
            window.close()
            return
        path = args.directory / "frame.jpg"
        if path.exists():
            image = QtGui.QImage(str(path))
            if not image.isNull():
                if args.engine == "releasing":
                    window._on_frame_ready(image)
                else:
                    window.video.set_frame(image)
        try:
            data = json.loads((args.directory / "status.json").read_text())
        except (OSError, ValueError):
            return
        if args.engine == "releasing":
            window._on_live_monitoring(data.get("live", {}))
            window._on_system_monitoring(data.get("system", {}))
            window._on_compliance(data.get("compliance", {}))
        elif args.engine == "purity":
            window.lbl_status.setText(data.get("status", ""))
            purity_view.update(data)
        else:
            window.notice.setText(data.get("status", ""))
            if data.get("sealing"):
                window.show_state(data["sealing"])
        for control in window.findChildren(QtWidgets.QAbstractButton):
            control.setEnabled(False)

    timer = QtCore.QTimer()
    timer.timeout.connect(refresh)
    timer.start(200)
    window.show()
    if args.smoke:
        QtCore.QTimer.singleShot(200, window.close)
    return application.exec() if args.engine == "releasing" else application.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
