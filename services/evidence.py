"""Atomic audit publication. Existing engine archives and retention stay authoritative."""

import json
import os
from pathlib import Path
import tempfile


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def attempt_directory(root, activity, attempt_id):
    return (
        Path(root)
        / "sessions"
        / activity["loan_id"]
        / activity["session_id"]
        / activity["id"]
        / attempt_id
    )
