import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from PyQt5 import QtWidgets
from app.main_window import MainWindow
from app.process_manager import ProcessManager

APPLICATION = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.managers = []
        self.command_patch = patch('app.process_manager.command', side_effect=self.command)
        self.command_patch.start()
        self.env_patch = patch('app.process_manager.engine_environment', return_value=os.environ.copy())
        self.env_patch.start()

    def tearDown(self):
        for manager in self.managers:
            self.wait(manager)
        self.command_patch.stop()
        self.env_patch.stop()
        self.directory.cleanup()

    def command(self, workflow):
        return [sys.executable, '-c', 'import time; time.sleep(.2)'], self.root

    def manager(self):
        manager = ProcessManager()
        manager.lock_path = self.root / 'test.lock'
        self.managers.append(manager)
        return manager

    def wait(self, manager):
        end = time.monotonic() + 8
        while manager.active and time.monotonic() < end:
            manager.poll()
            APPLICATION.processEvents()
            time.sleep(.01)
        self.assertFalse(manager.active, 'Test engine did not shut down')

    def test_exclusive_across_managers_and_switch_after_exit(self):
        first, second = self.manager(), self.manager()
        self.assertTrue(first.start('purity'))
        self.assertFalse(first.start('packet'))
        self.assertFalse(second.start('packet'))
        self.wait(first)
        self.assertTrue(second.start('packet'))
        self.wait(second)
        self.assertTrue(first.start('purity'))

    def test_failed_start_releases_lock(self):
        first, second = self.manager(), self.manager()
        with patch('app.process_manager.command', return_value=(['/missing/embsys-python'], self.root)):
            self.assertFalse(first.start('purity'))
        self.assertTrue(second.start('packet'))

    def test_error_exit_restores_controls(self):
        manager = self.manager()
        statuses = []
        manager.changed.connect(lambda status, active: statuses.append((status, active)))
        with patch('app.process_manager.command', return_value=([sys.executable, '-c', 'raise SystemExit(7)'], self.root)):
            self.assertTrue(manager.start('packet'))
        self.wait(manager)
        self.assertTrue(statuses[-1][0].startswith('Error'))
        self.assertFalse(statuses[-1][1])

    def test_buttons_and_close_wait_for_engine(self):
        window = MainWindow()
        window.manager.lock_path = self.root / 'test.lock'
        self.managers.append(window.manager)
        window.show()
        window.purity.click()
        self.assertFalse(window.purity.isEnabled())
        self.assertFalse(window.packet.isEnabled())
        self.assertFalse(window.close())
        self.wait(window.manager)
        self.assertTrue(window.purity.isEnabled())
        self.assertTrue(window.packet.isEnabled())
        self.assertEqual(window.status.text(), 'Finished')
        self.assertTrue(window.close())

    def test_child_inherits_ownership_if_launcher_disappears(self):
        manager, other = self.manager(), self.manager()
        self.assertTrue(manager.start('purity'))
        os.close(manager._lock_fd)
        manager._lock_fd = None
        manager.timer.stop()
        self.assertFalse(other.start('packet'))
        manager.process.wait(timeout=5)
        manager.process = None
        manager._active = False
        self.assertTrue(other.start('packet'))
