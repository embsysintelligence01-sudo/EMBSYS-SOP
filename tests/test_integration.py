import json
from pathlib import Path
import unittest
from app.config import ROOT, JEWEL, PACKET, command, engine_environment


class IntegrationTests(unittest.TestCase):
    def test_launch_paths_and_environment(self):
        for mode, directory, entry in [('purity', JEWEL, 'jewel_tracker_5.py'), ('packet', PACKET, 'packet_app.py')]:
            argv, cwd = command(mode)
            self.assertEqual(cwd, directory)
            self.assertEqual(Path(argv[1]), directory / entry)
            self.assertTrue(Path(argv[0]).is_file())
            env = engine_environment(mode)
            self.assertEqual(env['JEWEL_TRACKER_DIR'], str(JEWEL))
            self.assertEqual(env['JEWEL_PACKET_DIR'], str(PACKET))
            self.assertEqual(env['YOLO_AUTOINSTALL'], 'False')
            self.assertTrue(Path(env['PACKET_OCR_PYTHON']).is_file())

    def test_preserved_working_source_and_reviewed_migration(self):
        from services.source_integrity import verify_sources
        self.assertEqual(verify_sources(),[])

    def test_no_nested_repositories_or_environments(self):
        for folder in (ROOT / 'engines').rglob('*'):
            if folder.is_dir():
                self.assertNotEqual(folder.name, '.git')
                if folder.name.startswith('.venv'):
                    self.assertTrue(folder.is_symlink(),str(folder))
                    self.assertTrue(folder.resolve().is_relative_to(ROOT/'runtime'),str(folder))

    def test_camera_and_roi_configuration_remain_engine_local_and_valid(self):
        # Current bench calibration is authoritative; the original backup may be older.
        camera=json.loads((JEWEL/'camera.json').read_text())
        zones=json.loads((JEWEL/'zones.json').read_text())
        self.assertIn(camera.get('mode'),('usb','custom','dahua_main','dahua_sub','ip','rtsp','dahua'))
        self.assertIsInstance(zones.get('roi'),list)
        self.assertGreaterEqual(len(zones['roi']),3)
        self.assertTrue((JEWEL/'sop_config.json').is_file())


class EngineSwitchTests(unittest.TestCase):
    def test_real_camera_free_engines_switch_after_clean_exit(self):
        import time
        from unittest.mock import patch
        from PyQt5 import QtWidgets
        from app.process_manager import ProcessManager
        application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        manager = ProcessManager()
        statuses = []
        manager.changed.connect(lambda text, active: statuses.append(text))
        for workflow in ('purity', 'packet'):
            argv, cwd = command(workflow)
            if workflow == 'purity':
                argv = [argv[0], str(ROOT/'scripts/engine_check.py'), 'purity-ui']
            else:
                argv += ['--smoke-ui']
            with patch('app.process_manager.command', return_value=(argv, cwd)):
                self.assertTrue(manager.start(workflow))
            self.assertFalse(manager.start('packet' if workflow == 'purity' else 'purity'))
            deadline = time.monotonic() + 90
            while manager.active and time.monotonic() < deadline:
                application.processEvents()
                manager.poll()
                time.sleep(.02)
            self.assertFalse(manager.active, 'Engine smoke did not finish')
            self.assertEqual(statuses[-1], 'Finished')
