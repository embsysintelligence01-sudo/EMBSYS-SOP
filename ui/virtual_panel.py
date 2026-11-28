#!/usr/bin/env python3
"""Virtual hardware only: HTTP commands and state, no engine/model/camera/SQLite access."""

import argparse, os, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from PyQt5 import QtCore, QtWidgets
except ImportError:
    from app.config import interpreter

    os.execv(
        str(interpreter("purity")),
        [str(interpreter("purity")), __file__, *sys.argv[1:]],
    )
from hardware.virtual import VirtualAdapter
from hardware.interface import HardwareCommand


class NetworkWorker(QtCore.QThread):
    done = QtCore.pyqtSignal(str, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        import queue, threading

        self.requests = queue.Queue(maxsize=16)
        self.halt = threading.Event()

    def run(self):
        while not self.halt.is_set():
            item = self.requests.get()
            if item is None:
                return
            key, operation = item
            try:
                self.done.emit(key, operation(), None)
            except Exception as exc:
                self.done.emit(key, None, str(exc))

    def stop(self):
        self.halt.set()
        self.requests.put(None)


class Panel(QtWidgets.QWidget):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.pending = None
        self.retries = 0
        self.polling = False
        self.last = {}
        self.blink = False
        self.callbacks = {}
        self.closed = False
        self.network = NetworkWorker(self)
        self.network.done.connect(self.response)
        self.network.start()
        self.setWindowTitle("EMBSYS · Virtual hardware")
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self.setFixedWidth(390)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        title = QtWidgets.QLabel("EMBSYS SOP")
        title.setStyleSheet("font-weight:700;font-size:18px")
        layout.addWidget(title)
        self.lcd = QtWidgets.QLabel("Connecting to controller…")
        self.lcd.setWordWrap(True)
        self.lcd.setMinimumHeight(125)
        self.lcd.setStyleSheet(
            "background:#152821;color:#a7e6bb;font-family:monospace;padding:15px;border-radius:6px"
        )
        layout.addWidget(self.lcd)
        leds = QtWidgets.QHBoxLayout()
        self.leds = {}
        for key, label in [
            ("ready", "READY"),
            ("running", "RUNNING"),
            ("fault", "FAULT"),
        ]:
            widget = QtWidgets.QLabel("● " + label)
            self.leds[key] = widget
            leds.addWidget(widget)
        layout.addLayout(leds)
        buttons = QtWidgets.QHBoxLayout()
        self.buttons = {}
        for name, color in [
            ("START", "#227d4f"),
            ("RESET", "#aa7c10"),
            ("STOP", "#ae303c"),
        ]:
            button = QtWidgets.QPushButton(name)
            button.setMinimumHeight(50)
            button.setStyleSheet(
                "color:white;font-weight:700;background:" + color + ";border-radius:5px"
            )
            button.clicked.connect(lambda checked=False, n=name: self.command(n))
            button.setEnabled(False)
            buttons.addWidget(button)
            self.buttons[name] = button
        layout.addLayout(buttons)
        self.ack = QtWidgets.QLabel("")
        self.ack.setWordWrap(True)
        layout.addWidget(self.ack)
        layout.addWidget(
            QtWidgets.QLabel("Closing this panel leaves surveillance running.")
        )
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(500)
        self.flash = QtCore.QTimer(self)
        self.flash.timeout.connect(self.render)
        self.flash.start(400)
        self.poll()

    def request(self, operation, callback):
        if self.closed:
            return
        key = uuid.uuid4().hex
        self.callbacks[key] = callback
        self.network.requests.put((key, operation))

    def response(self, key, result, error):
        callback = self.callbacks.pop(key, None)
        if callback is not None and not self.closed:
            callback(result, error)

    def closeEvent(self, event):
        self.closed = True
        self.timer.stop()
        self.flash.stop()
        self.network.stop()
        self.network.wait(4000)
        event.accept()

    def poll(self):
        if self.polling:
            return
        self.polling = True

        def done(result, error):
            self.polling = False
            if error:
                self.last = {}
                self.lcd.setText("CONTROLLER OFFLINE\n" + error)
            else:
                self.last = result
            self.render()

        self.request(self.adapter.snapshot, done)

    def render(self):
        self.blink = not self.blink
        state = self.last
        if state:
            current = state.get("current") or {}
            following = state.get("next") or {}
            names = {
                "purity": "Purity Testing",
                "packet": "Packet Sealing",
                "releasing": "Packet Releasing",
            }
            self.lcd.setText(
                "Loan: "
                + str(current.get("loan_id") or following.get("loan_id") or "—")
                + "\nCurrent: "
                + names.get(current.get("engine"), "—")
                + "\nNext: "
                + names.get(following.get("engine"), "—")
                + "\nStatus: "
                + state.get("state", "OFFLINE")
            )
        for name, button in self.buttons.items():
            button.setEnabled(
                bool(state.get("controls", {}).get(name.lower()))
                and self.pending is None
            )
        for key, widget in self.leds.items():
            on = state.get("leds", {}).get(key, False)
            if key == "running" and state.get("leds", {}).get("blink"):
                on = self.blink
            color = (
                {"ready": "#29985a", "running": "#218ce3", "fault": "#d33e49"}[key]
                if on
                else "#8b959e"
            )
            widget.setStyleSheet("font-weight:700;color:" + color)

    def command(self, name):
        if self.pending:
            return
        if (
            name == "RESET"
            and QtWidgets.QMessageBox.question(
                self,
                "Confirm RESET",
                "Interrupt this attempt and retain its incomplete evidence? A new attempt will be prepared.",
            )
            != QtWidgets.QMessageBox.Yes
        ):
            return
        self.pending = HardwareCommand(
            name, confirmed=name == "RESET", command_id=uuid.uuid4().hex
        )
        self.retries = 0
        self.submit()

    def submit(self):
        self.render()
        command = self.pending

        def done(result, error):
            if error:
                self.retries += 1
                if self.retries < 3:
                    self.ack.setText(
                        "Waiting for acknowledgement; retrying the same command ID…"
                    )
                    QtCore.QTimer.singleShot(1000, self.submit)
                    return
                self.ack.setText("Command not acknowledged: " + error)
            else:
                self.ack.setText(
                    "Acknowledged " + command.name + " · " + result.get("state", "")
                )
            self.pending = None
            self.poll()

        self.request(lambda: self.adapter.submit(command), done)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    app = QtWidgets.QApplication([])
    panel = Panel(VirtualAdapter(args.url, os.environ.get("EMBSYS_API_TOKEN", "")))
    panel.show()
    if args.smoke:
        QtCore.QTimer.singleShot(600, panel.close)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
