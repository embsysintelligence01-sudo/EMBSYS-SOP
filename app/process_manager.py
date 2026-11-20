"""Qt-polled subprocesses with an inherited Linux camera-ownership lock."""
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from PyQt5 import QtCore
from .config import ROOT, command, engine_environment


class ProcessManager(QtCore.QObject):
    changed = QtCore.pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self._lock_fd = None
        self._exit_code = None
        self._active = False
        token = hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
        self.lock_path = Path(tempfile.gettempdir()) / f'embsys-sop-{os.getuid()}-{token}.lock'
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)

    @property
    def active(self):
        return self._active

    def _acquire(self):
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def start(self, workflow):
        if self.active:
            return False
        try:
            self._lock_fd = self._acquire()
        except BlockingIOError:
            self.changed.emit('Running — another launcher owns the workflow. Close its engine first.', False)
            return False
        except OSError:
            self.changed.emit('Error — cannot acquire workflow lock. Check local permissions.', False)
            return False
        try:
            argv, cwd = command(workflow)
            self.process = subprocess.Popen(argv, cwd=cwd, env=engine_environment(workflow),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True, pass_fds=(self._lock_fd,))
        except Exception:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            self.changed.emit('Error — workflow could not start. Check installation and local configuration.', False)
            return False
        self._active = True
        self._exit_code = None
        self.changed.emit('Running — ' + ('Purity Testing' if workflow == 'purity' else 'Packet Sealing'), True)
        self.timer.start()
        return True

    def poll(self):
        if not self.active:
            return
        if self.process is not None:
            code = self.process.poll()
            if code is None:
                return
            self._exit_code = code
            self.process = None
            # Close rather than LOCK_UN: inherited descriptors must keep ownership
            # until engine descendants finish. Never signal/kill an engine to switch.
            os.close(self._lock_fd)
            self._lock_fd = None
        try:
            fd = self._acquire()
        except BlockingIOError:
            self.changed.emit('Running — waiting for remaining engine resources to close.', True)
            return
        os.close(fd)
        self.timer.stop()
        self._active = False
        self.changed.emit('Finished' if self._exit_code == 0 else
            'Error — engine exited unexpectedly. Review evidence before starting another session.', False)
