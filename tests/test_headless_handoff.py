"""Two-START uses existing handoff snapshots and the original TrackingBridge."""

import sys, unittest, tempfile, threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "engines/Jeweltracker/tests"),
    str(ROOT / "engines/Jeweltracker"),
    str(ROOT / "engines/Packetsealing_001"),
    str(ROOT),
]
from services.handoff import SurveillanceHandoff, recovered_handoff
from services.adapters import PledgeAdapter
import test_packet_stage as fixtures


class HeadlessHandoffTests(unittest.TestCase):
    def track(self):
        return fixtures.PacketStageTests().track(7)

    def test_freeze_final_results_and_uncertainty_without_widget(self):
        from PyQt5.QtCore import QCoreApplication

        app = QCoreApplication.instance() or QCoreApplication([])
        with tempfile.TemporaryDirectory() as directory:
            worker = SimpleNamespace(
                tracks=[self.track()],
                mode="LOCKED",
                lock=threading.Lock(),
                packet_capture_stopped=threading.Event(),
                isRunning=lambda: True,
                frame_wh=(640, 480),
                continuous_monitor=SimpleNamespace(
                    directory=directory, tests=[], finalization_errors=[]
                ),
                id_model=SimpleNamespace(
                    ready=True,
                    labels=[7],
                    index=SimpleNamespace(vecs=np.array([[0.1, 0.2]])),
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "barrier"):
                SurveillanceHandoff(worker, directory)
            worker.packet_capture_stopped.set()
            context = SurveillanceHandoff(worker, directory)
            self.assertIs(context.payload()["worker"], worker)
            self.assertFalse(context.ready.is_set())
            self.assertEqual(context.testing["jewels"][0]["id"], 7)
            context.finish()
            self.assertTrue(context.ready.is_set())
            recovered = recovered_handoff(Path(directory) / "identity.json", 0, {}, [])
            self.assertEqual(recovered["testing"]["jewels"][0]["id"], 7)
            self.assertEqual(
                recovered["testing"]["jewels"][0]["identity_status"], "UNCERTAIN"
            )
            self.assertEqual(recovered["worker"].tracks, [])

    def test_sealing_starts_during_purity_background_drain(self):
        adapter = PledgeAdapter(lambda *a, **kw: None)
        old = SimpleNamespace(
            isRunning=lambda: True, packet_capture_stopped=threading.Event(), source=0
        )
        old.packet_capture_stopped.set()
        adapter.worker = adapter.purity = old
        expected = {"worker": old, "testing": {"jewels": []}}
        adapter.context = SimpleNamespace(payload=lambda: expected)
        fake = Mock()
        adapter.PacketWorker = Mock(return_value=fake)
        adapter.source_settings = lambda: (0, {}, [])
        with tempfile.TemporaryDirectory() as directory:
            adapter.start(
                {
                    "activity": {"engine": "packet", "linked": 1},
                    "attempt_id": "second",
                    "directory": directory,
                }
            )
        self.assertIs(adapter.PacketWorker.call_args.kwargs["handoff"], expected)
        fake.start.assert_called_once()

    def test_missing_history_uses_existing_unassigned_handoff_not_a_new_batch(self):
        payload = recovered_handoff(None, 0, {}, [])
        self.assertEqual(payload["worker"].mode, "LOCKED")
        self.assertEqual(payload["worker"].tracks, [])
        self.assertEqual(payload["testing"]["identity_status"], "UNASSIGNED")
        self.assertEqual(payload["testing"]["analysis_status"], "INTERRUPTED")

    def test_sealing_stop_during_startup_cannot_release_before_real_capture_barrier(
        self,
    ):
        events = []
        adapter = PledgeAdapter(lambda event, aid, **kwargs: events.append(event))
        worker = SimpleNamespace(
            running=False,
            grabber=None,
            capture_stopped=threading.Event(),
            isRunning=lambda: True,
        )
        adapter.worker = worker
        adapter.current = {"attempt_id": "early-stop", "activity": {"engine": "packet"}}
        adapter.tick()
        self.assertNotIn("released", events)
        alive = [True]
        worker.grabber = SimpleNamespace(is_alive=lambda: alive[0], connected=False)
        worker.capture_stopped.set()
        adapter.tick()
        self.assertNotIn("released", events)
        alive[0] = False
        adapter.tick()
        self.assertEqual(events, ["released"])
