"""Linux device ownership checks; no engine, inference or Qt imports."""
from pathlib import Path
import os


def camera_owners():
    """Best-effort detection of already-open Linux cameras outside our launcher."""
    owners = set()
    for process in Path('/proc').glob('[0-9]*'):
        try:
            for descriptor in (process/'fd').iterdir():
                try:
                    if os.readlink(descriptor).startswith('/dev/video'):
                        owners.add(int(process.name))
                except OSError:
                    pass
        except (OSError, PermissionError):
            pass
    return sorted(owners)

