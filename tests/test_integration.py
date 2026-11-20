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

    def test_uncommitted_packet_sources_copied(self):
        snapshot = json.loads((ROOT / 'docs/SOURCE_SNAPSHOT.json').read_text())['Packetsealing_001']
        for line in snapshot['status'].splitlines():
            path = line[3:]
            if path.endswith('.py'):
                self.assertEqual((Path(snapshot['source']) / path).read_bytes(), (PACKET / path).read_bytes(), path)

    def test_no_nested_repositories_or_environments(self):
        for folder in (ROOT / 'engines').rglob('*'):
            if folder.is_dir():
                self.assertNotEqual(folder.name, '.git')
                self.assertFalse(folder.name.startswith('.venv'))

    def test_camera_settings_and_roi_preserved(self):
        source = ROOT.parent / 'jewel_tracker/jewel_tracker'
        for name in ('camera.json', 'zones.json', 'sop_config.json'):
            self.assertEqual(json.loads((source / name).read_text()), json.loads((JEWEL / name).read_text()))


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
