"""Exercise the real read-only Qt view with worker results, without a camera."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "engines/Jeweltracker"),
    str(ROOT / "engines/Packetsealing_001"),
    str(ROOT),
]
import numpy as np
from PyQt5 import QtWidgets
import jewel_tracker_5 as tracker
from services.adapters import PledgeAdapter
from services.debug_stream import DebugStream
from services.diagnostics import PurityDiagnostics, decode_image, image_payload
from ui.purity_diagnostics import PurityView

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class PurityDiagnosticsTests(unittest.TestCase):
    def test_worker_snapshot_to_existing_ui_with_images_and_no_camera(self):
        image = np.full((800, 1000, 3), (20, 80, 170), np.uint8)
        diagnostic = PurityDiagnostics()
        diagnostic.event(
            dict(
                kind="hand_model", message="Hand model: ready", detail="existing worker"
            )
        )
        diagnostic.event(dict(kind="reference", photo=image, timestamp=100))
        diagnostic.event(dict(kind="begin", timestamp=101))
        diagnostic.event(
            dict(kind="contact", photo=image, timestamp=102, confirmed=True)
        )
        diagnostic.event(dict(kind="baseline", photo=image, timestamp=103))
        diagnostic.event(dict(kind="mask_decision", detected=False, pending=False))
        diagnostic.event(
            dict(
                kind="phase",
                phase="WAIT_FOR_ACID",
                jewel_id=1,
                message="Waiting for acid",
                remaining=3.5,
            )
        )
        diagnostic.log("Stone loaded", "info")
        worker = SimpleNamespace(
            continuous_monitor=SimpleNamespace(directory="/test/attempt/purity/session")
        )
        record = dict(
            prediction=None,
            rubbing_status="CONFIRMED_LIVE",
            close_reason="INACTIVITY_TIMEOUT",
            record_path="/test/manifest.json",
        )
        adapter = SimpleNamespace(
            status="LOCKED",
            diagnostics=diagnostic,
            worker=worker,
            table=[[1, image, True, image, False, False, record, False, image]],
            debug_frame=lambda: image.copy(),
        )
        with tempfile.TemporaryDirectory() as directory:
            stream = DebugStream.__new__(DebugStream)
            stream.directory, stream.last = Path(directory), 0
            stream.update(adapter)
            data = json.loads((Path(directory) / "status.json").read_text())
            self.assertLess((Path(directory) / "status.json").stat().st_size, 200_000)
            self.assertLessEqual(decode_image(data["table"][0][1]).shape[1], 128)
            # A viewer must never construct an inference worker or camera.
            with (
                patch.object(
                    tracker, "Worker", side_effect=AssertionError("duplicate worker")
                ),
                patch.object(
                    tracker, "RTSPGrabber", side_effect=AssertionError("second camera")
                ),
            ):
                window = tracker.MainWindow(auto_connect=False)
                window.rub_timer.stop()
                window.preview_timer.stop()
                window.packet_stage.timer.stop()
                view = PurityView(window, tracker)
                initial_logs = window.log.count()
                view.update(data)
                self.assertEqual(window.hand_model_status.text(), "Hand model: ready")
                self.assertIn("Baseline: CAPTURED", window.before_instruction.text())
                self.assertIn(
                    "Before stone: CAPTURED", window.before_instruction.text()
                )
                self.assertFalse(window.before_preview.pixmap().isNull())
                self.assertFalse(window.contact_preview.pixmap().isNull())
                for column in (1, 3, 6):
                    self.assertFalse(
                        window.table.cellWidget(0, column).pixmap().isNull()
                    )
                self.assertEqual(window.table.item(0, 4).text(), "INCOMPLETE")
                self.assertIn("NOT DETECTED", window.rub_instruction.text())
                self.assertIn("Waiting for acid", window.acid_instruction.text())
                self.assertIn(
                    "/test/attempt/purity/session", window.recordings_location.text()
                )
                self.assertEqual(window.log.count(), initial_logs + 1)
                view.update(data)
                self.assertEqual(window.log.count(), initial_logs + 1)
                data["purity"]["workflow"].append(
                    dict(
                        sequence=1000,
                        payload=dict(
                            kind="result",
                            record=dict(
                                cycle_id="older", close_reason="INACTIVITY_TIMEOUT"
                            ),
                        ),
                    )
                )
                data["purity"]["cycle_id"] = "active"
                data["purity"]["latest_token"] = "active"
                phase = window.phase_label.text()
                view.update(data)
                self.assertEqual(window.phase_label.text(), phase)
                data["purity"]["cycle_id"] = None
                view.update(data)
                self.assertEqual(window.phase_label.text(), phase)
                data["purity"]["latest_token"] = "older"
                view.update(data)
                self.assertIn("FINALIZED", window.phase_label.text())
                self.assertIn("Acid not confirmed", window.phase_label.text())
                self.assertIsNone(window.worker)
                window.close()

    def test_latest_events_bounded_and_new_cycle_clears_stale_captures(self):
        diagnostic = PurityDiagnostics()
        for i in range(200):
            diagnostic.event(dict(kind="status", message=str(i)))
            diagnostic.log(str(i), "info")
        diagnostic.event(
            dict(kind="baseline", timestamp=1, photo=np.zeros((10, 10, 3), np.uint8))
        )
        diagnostic.event(
            dict(kind="contact", timestamp=1, photo=np.zeros((10, 10, 3), np.uint8))
        )
        diagnostic.event(dict(kind="begin", timestamp=2))
        self.assertEqual(len(diagnostic.logs), 40)
        self.assertNotIn("contact", diagnostic.workflow)
        self.assertNotIn("baseline", diagnostic.workflow)
        self.assertEqual(diagnostic.workflow["status"]["payload"]["message"], "199")

    def test_original_hand_overlay_used_on_copy_without_inference(self):
        image = np.zeros((480, 640, 3), np.uint8)
        points = np.tile([200.0, 200.0], (21, 1))
        points[8] = [240.0, 200.0]
        preview = dict(
            shape=image.shape[:2],
            timestamp=1.0,
            hands=[
                dict(
                    points=points,
                    selected=True,
                    pinched=True,
                    in_area=True,
                    handedness="Right",
                    distance_px=40,
                )
            ],
            pinch_px=100,
            selection="Any",
            hold_seconds=0.5,
            state="Holding",
        )
        adapter = PledgeAdapter.__new__(PledgeAdapter)
        adapter.worker = SimpleNamespace(
            grabber=SimpleNamespace(latest=lambda: (image.copy(), 1.1)),
            preview=SimpleNamespace(render=lambda frame, stamp: frame),
            recording_hand_preview=preview,
        )
        adapter.current = {"activity": {"engine": "purity"}}
        adapter.tracker = tracker
        result = adapter.debug_frame()
        self.assertGreater(np.count_nonzero(result), 0)
        self.assertEqual(np.count_nonzero(image), 0)
        # Original freshness rule still prevents stale landmarks.
        stale = tracker.draw_hand_preview(image.copy(), 3.0, preview)
        self.assertEqual(np.count_nonzero(stale), 0)


if __name__ == "__main__":
    unittest.main()
