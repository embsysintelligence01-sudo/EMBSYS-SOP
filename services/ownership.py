"""Actual process/device ownership. Locks survive a browser or panel closing."""

import fcntl
import hashlib
import os
from pathlib import Path
import tempfile
from app.config import ROOT
from app.ownership import camera_owners


class ExclusiveLock:
    def __init__(self, path):
        self.path = Path(path)
        self.fd = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise RuntimeError(
                "Another controller or engine owns this resource"
            ) from None
        self.fd = fd
        return self

    def release(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.release()


class CameraOwnership(ExclusiveLock):
    def __init__(self):
        token = hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
        super().__init__(
            Path(tempfile.gettempdir()) / f"embsys-sop-{os.getuid()}-{token}.lock"
        )

    def acquire(self):
        super().acquire()
        if camera_owners():
            self.release()
            raise RuntimeError("Camera is open outside the managed controller")
        return self
