#!/usr/bin/env python3
"""Preserve engine-relative paths while keeping writable data outside source."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = {
    'engines/Jeweltracker/diagnostics': 'runtime/Jeweltracker/diagnostics',
    'engines/Jeweltracker/jewel_guard.log': 'runtime/Jeweltracker/jewel_guard.log',
    'engines/Packetsealing_001/outputs': 'runtime/Packetsealing_001/outputs',
    'engines/Packetsealing_001/packet_mouth_lab/sample_images': 'development/fixtures/Packetsealing_001/sample_images',
}


@contextmanager
def maintenance_lock():
    token = hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f'embsys-sop-{os.getuid()}-{token}.lock'
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    except BlockingIOError:
        raise RuntimeError('An SOP engine owns the camera. Finish the active session before preparing storage.') from None
    finally:
        os.close(fd)


def prepare():
    with maintenance_lock():
        for original, relocated in LAYOUT.items():
            source, target = ROOT/original, ROOT/relocated
            if source.is_symlink():
                if source.resolve() != target.resolve() or not target.exists():
                    raise RuntimeError('Unexpected or unavailable runtime link: '+original)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                if target.exists():
                    raise RuntimeError('Storage already exists in both locations; refusing to merge evidence: '+original)
                source.rename(target)
            elif original.endswith('.log'):
                target.touch(exist_ok=True)
            else:
                target.mkdir(parents=True, exist_ok=True)
            source.symlink_to(os.path.relpath(target, source.parent), target_is_directory=target.is_dir())


if __name__ == '__main__':
    prepare()
    print('Engine paths preserved; writable data and local sample images stored outside runtime source.')
