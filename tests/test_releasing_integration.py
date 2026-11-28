"""Preparation boundaries, copied-source integrity and camera ownership."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from app.config import ROOT, RELEASING, command, engine_environment, interpreter
from scripts.prepare_runtime import maintenance_lock
from development import run_engine


class ReleasingIntegrationTests(unittest.TestCase):
    def test_releasing_is_development_only(self):
        with self.assertRaisesRegex(ValueError, 'development-only'):
            command('releasing')
        argv, cwd = command('releasing', development=True)
        self.assertEqual(cwd, RELEASING)
        self.assertEqual(Path(argv[1]), RELEASING/'automated_app/app.py')
        self.assertTrue(Path(argv[0]).is_file())

    def test_separate_environment_and_paths(self):
        env = engine_environment('releasing')
        self.assertNotIn('PYTHONPATH', env)
        self.assertNotIn('QT_QPA_PLATFORM_PLUGIN_PATH', env)
        self.assertEqual(Path(env['PACKET_CUT_MODELS_DIR']).resolve(),(RELEASING/'models').resolve())
        self.assertEqual(env['VIRTUAL_ENV'], str(interpreter('releasing').parent.parent))
        self.assertEqual(env['WORKFLOW_DEVICE'], 'cpu')

    def test_production_manager_refuses_releasing(self):
        from app.process_manager import ProcessManager
        manager = ProcessManager()
        self.assertFalse(manager.start('releasing'))
        self.assertFalse(manager.active)

    def test_development_and_production_share_camera_lock(self):
        from app.process_manager import ProcessManager
        with maintenance_lock():
            manager = ProcessManager()
            self.assertFalse(manager.start('purity'))
            self.assertFalse(manager.start('packet'))
            self.assertFalse(manager.start('releasing'))
            self.assertFalse(manager.active)
            # A separate development process also refuses all engines before spawn.
            for engine in ('purity', 'packet', 'releasing'):
                result = subprocess.run([sys.executable, str(ROOT/'development/run_engine.py'), engine],
                                        capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('owns the camera', result.stderr)

    def test_unmanaged_open_camera_refuses_launch(self):
        with patch.object(run_engine, 'camera_owners', return_value=[123]), \
                patch.object(run_engine.subprocess, 'Popen') as spawn:
            self.assertEqual(run_engine.main(['releasing']), 1)
            spawn.assert_not_called()

    def test_child_inherits_lock_and_uses_separate_process_session(self):
        child = unittest.mock.Mock()
        child.wait.return_value = 0
        def spawn(*args, **kwargs):
            fd, = kwargs['pass_fds']
            self.assertGreaterEqual(fd, 0)
            self.assertTrue(kwargs['start_new_session'])
            return child
        with patch.object(run_engine, 'camera_owners', return_value=[]), \
                patch.object(run_engine.subprocess, 'Popen', side_effect=spawn):
            self.assertEqual(run_engine.main(['releasing']), 0)

    def test_production_refuses_unmanaged_camera_owner(self):
        from app.process_manager import ProcessManager
        with patch('app.process_manager.camera_owners', return_value=[123]), \
                patch('app.process_manager.subprocess.Popen') as spawn:
            manager = ProcessManager()
            self.assertFalse(manager.start('purity'))
            self.assertFalse(manager.active)
            spawn.assert_not_called()

    def test_engine_help_can_be_forwarded_without_bypassing_ownership(self):
        child = unittest.mock.Mock()
        child.wait.return_value = 0
        with patch.object(run_engine, 'camera_owners', return_value=[]), \
                patch.object(run_engine.subprocess, 'Popen', return_value=child) as spawn:
            self.assertEqual(run_engine.main(['releasing', '--', '--help']), 0)
            self.assertEqual(spawn.call_args.args[0][-1], '--help')
            self.assertNotIn('--', spawn.call_args.args[0])

    def test_source_preserved(self):
        from services.source_integrity import verify_sources
        self.assertEqual(verify_sources(),[])
