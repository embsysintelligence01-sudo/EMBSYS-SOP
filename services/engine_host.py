#!/usr/bin/env python3
"""QCoreApplication engine host: no QWidget, display, camera service or alternate inference."""

from pathlib import Path
import argparse
import json
import os
import queue
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import JEWEL, PACKET, RELEASING
from services.ownership import CameraOwnership
from services.evidence import publish


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("pledge", "releasing"), required=True)
    parser.add_argument("--mode", choices=("production", "debug"), required=True)
    args = parser.parse_args()
    if args.family == "pledge":
        sys.path[:0] = [str(JEWEL), str(PACKET), str(ROOT)]
        from PyQt5 import QtCore
    else:
        sys.path[:0] = [str(RELEASING / "automated_app"), str(RELEASING), str(ROOT)]
        from PyQt6 import QtCore
    from services.adapters import PledgeAdapter, ReleaseAdapter
    from services.debug_stream import DebugStream

    application = QtCore.QCoreApplication([])
    messages = queue.Queue(maxsize=32)
    write_lock = threading.Lock()
    camera = CameraOwnership()

    class Host(QtCore.QObject):
        def __init__(self):
            super().__init__()
            self.exit_requested = False
            self.current = None
            self.stream = None
            self.last_health = 0
            self.adapter = (
                PledgeAdapter if args.family == "pledge" else ReleaseAdapter
            )(self.emit, args.mode == "debug")
            budget = int(os.environ.get("OMP_NUM_THREADS", "2"))
            if args.family == "pledge":
                self.adapter.tracker.ORT_THREADS = budget
                self.adapter.tracker.JEWEL_MODEL_THREADS = budget
            from services.model_budget import apply_cpu_budget

            apply_cpu_budget(budget)
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(100)
            self.timer.timeout.connect(self.tick)
            self.timer.start()
            import psutil

            self.process = psutil.Process()
            self.process.cpu_percent()

        def emit(self, event, attempt_id, **data):
            if event == "released":
                camera.release()
            if (
                event == "finished"
                and self.stream
                and self.stream.attempt_id == attempt_id
            ):
                self.stream.close()
            with write_lock:
                try:
                    print(
                        "@SOP@"
                        + json.dumps(
                            dict(event=event, attempt_id=attempt_id, **data),
                            default=str,
                        ),
                        flush=True,
                    )
                except (BrokenPipeError, OSError):
                    self.exit_requested = True
                    self.adapter.stop()

        def tick(self):
            try:
                try:
                    message = messages.get_nowait()
                except queue.Empty:
                    message = None
                if message:
                    if message["command"] == "START":
                        self.current = message
                        camera.acquire()
                        self.adapter.start(message)
                        if args.mode == "debug":
                            if self.stream:
                                self.stream.close()
                            self.stream = DebugStream(message, args.family)
                    elif message["command"] == "STOP":
                        self.adapter.stop()
                    elif message["command"] == "EXIT":
                        self.exit_requested = True
                self.adapter.tick()
                if self.stream:
                    self.stream.update(self.adapter)
                if self.current and time.monotonic() - self.last_health >= 1:
                    self.last_health = time.monotonic()
                    worker = getattr(self.adapter, "worker", None)
                    grabber = getattr(worker, "grabber", None)
                    import re

                    matches = re.search(r"([0-9.]+) detection FPS", self.adapter.status)
                    fps = (
                        float(matches.group(1))
                        if matches
                        else getattr(worker, "analysis_fps", None)
                    )
                    self.emit(
                        "health",
                        self.current["attempt_id"],
                        rss_mb=round(self.process.memory_info().rss / 1048576, 1),
                        cpu_percent=self.process.cpu_percent(),
                        status=self.adapter.status,
                        analysis_fps=fps,
                        camera_connected=bool(
                            getattr(grabber, "connected", camera.fd is not None)
                        ),
                    )
                if self.exit_requested and self.adapter.close():
                    if self.stream:
                        self.stream.close()
                    camera.release()
                    application.quit()
            except Exception as exc:
                aid = self.current["attempt_id"] if self.current else None
                self.emit("error", aid, error=f"{type(exc).__name__}: {exc}")
                self.adapter.stop()
                if not self.adapter.busy():
                    self.emit("released", aid)
                    self.emit(
                        "finished",
                        aid,
                        evidence_status="REVIEW_REQUIRED",
                        reports={},
                        error=str(exc),
                    )
                    camera.release()
                    application.quit()

    host = Host()

    def read():
        for line in sys.stdin:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            messages.put(message)
        messages.put({"command": "EXIT"})

    threading.Thread(target=read, name="controller-input", daemon=True).start()
    code = application.exec() if args.family == "releasing" else application.exec_()
    camera.release()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
