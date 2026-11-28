"""Non-executable audit snapshot; retains arrays without asserting a new identity match."""

import math
from collections import deque


def plain(value, depth=0):
    if depth > 16:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple, deque, set)):
        return [plain(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): plain(v, depth + 1) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return plain(value.tolist(), depth + 1)
    if hasattr(value, "__dict__"):
        return {
            k: plain(v, depth + 1)
            for k, v in vars(value).items()
            if not k.startswith("_")
            and k
            not in ("model", "extractor", "backend", "dispatcher", "lock", "session")
        }
    return None


def identity_snapshot(worker):
    with getattr(worker, "lock", __import__("contextlib").nullcontext()):
        return dict(
            tracks=plain(getattr(worker, "tracks", [])),
            gallery=plain(getattr(worker, "id_model", None)),
            mode=getattr(worker, "mode", "UNKNOWN"),
            frame_wh=plain(getattr(worker, "frame_wh", None)),
        )
