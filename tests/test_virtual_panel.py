import os, time, unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets
from ui.virtual_panel import Panel

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeAdapter:
    def __init__(self):
        self.commands = []
        self.state = {
            "state": "READY",
            "current": None,
            "next": {"loan_id": "TEST", "engine": "purity"},
            "controls": {"start": True, "stop": False, "reset": False},
            "leds": {"ready": True, "running": False, "fault": False, "blink": False},
        }

    def snapshot(self):
        return self.state

    def submit(self, command):
        self.commands.append(command.payload())
        return {"acknowledged": True, "state": "STARTING"}


class VirtualPanelTests(unittest.TestCase):
    def wait(self, predicate):
        until = time.monotonic() + 3
        while time.monotonic() < until:
            APP.processEvents()
            if predicate():
                return
            time.sleep(0.01)
        self.fail("Panel request timed out")

    def test_controls_and_ack_follow_controller_and_close_is_independent(self):
        adapter = FakeAdapter()
        panel = Panel(adapter)
        panel.show()
        self.wait(lambda: panel.buttons["START"].isEnabled())
        self.assertFalse(panel.buttons["STOP"].isEnabled())
        self.assertIn("TEST", panel.lcd.text())
        panel.buttons["START"].click()
        self.wait(lambda: len(adapter.commands) == 1 and panel.pending is None)
        self.assertEqual(adapter.commands[0]["name"], "START")
        self.assertIn("Acknowledged", panel.ack.text())
        panel.close()
        APP.processEvents()
        self.assertEqual(len(adapter.commands), 1)

    def test_running_stop_and_reset_confirmation(self):
        from unittest.mock import patch

        adapter = FakeAdapter()
        adapter.state.update(
            state="RUNNING",
            current={"loan_id": "X", "engine": "packet"},
            controls={"start": False, "stop": True, "reset": True},
        )
        panel = Panel(adapter)
        panel.show()
        self.wait(lambda: panel.buttons["STOP"].isEnabled())
        with patch.object(
            QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.No
        ):
            panel.command("RESET")
        self.assertEqual(adapter.commands, [])
        with patch.object(
            QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes
        ):
            panel.command("RESET")
        self.wait(lambda: len(adapter.commands) == 1)
        self.assertTrue(adapter.commands[0]["confirmed"])
        self.wait(lambda: panel.pending is None)
        panel.close()
        APP.processEvents()
