"""One bounded local diagnostic snapshot; no web stream and no second camera."""

import os
from pathlib import Path
import subprocess
import time
from app.config import ROOT, interpreter, engine_environment
from services.evidence import publish


class DebugStream:
    def __init__(self, message, family):
        self.attempt_id = message["attempt_id"]
        self.directory = Path(message["directory"]) / "debug"
        self.directory.mkdir(exist_ok=True)
        self.last = 0
        self.viewer = None
        env = engine_environment("releasing" if family == "releasing" else "packet")
        if (
            os.environ.get("DISPLAY")
            or os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        ):
            self.viewer = subprocess.Popen(
                [
                    str(
                        interpreter("releasing" if family == "releasing" else "packet")
                    ),
                    str(ROOT / "ui/engine_viewer.py"),
                    "--engine",
                    message["activity"]["engine"],
                    "--directory",
                    str(self.directory),
                ],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def update(self, adapter):
        if time.monotonic() - self.last < 0.2:
            return
        self.last = time.monotonic()
        import cv2

        frame = adapter.debug_frame()
        if frame is not None:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                temporary = self.directory / "frame.tmp"
                temporary.write_bytes(encoded.tobytes())
                temporary.replace(self.directory / "frame.jpg")
        from services.identity import plain
        from services.diagnostics import image_payload

        rows = []
        for original in getattr(adapter, "table", []):
            row = list(original)
            for index in (1, 3, 8):
                if len(row) > index:
                    row[index] = image_payload(row[index], maximum=(128, 96))
            if len(row) > 6 and isinstance(row[6], dict):
                row[6] = {
                    k: v
                    for k, v in row[6].items()
                    if k
                    in (
                        "prediction",
                        "record_path",
                        "close_reason",
                        "rubbing_status",
                        "recording_assignment",
                        "recording_candidate_id",
                    )
                }
            rows.append(plain(row))
        sealing = dict(getattr(adapter, "sealing_state", {}))
        if "jewels" in sealing:
            sealing["jewels"] = [dict(jewel, thumb=None) for jewel in sealing["jewels"]]
        publish(
            self.directory / "status.json",
            dict(
                status=adapter.status,
                table=rows,
                sealing=plain(sealing),
                live=getattr(adapter, "live", {}),
                system=getattr(adapter, "system", {}),
                compliance=getattr(adapter, "compliance", {}),
                purity=(
                    adapter.diagnostics.snapshot(adapter.worker)
                    if getattr(adapter, "diagnostics", None) is not None
                    else {}
                ),
            ),
        )

    def close(self):
        publish(self.directory / "closed.json", {"closed": True})
