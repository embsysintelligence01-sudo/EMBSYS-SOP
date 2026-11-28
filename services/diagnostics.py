"""Bounded display snapshots of existing results; never runs inference."""

import base64
from collections import deque
from services.identity import plain


def image_payload(image, maximum=(640, 360)):
    if image is None or not getattr(image, "size", 0):
        return None
    import cv2

    height, width = image.shape[:2]
    scale = min(1.0, maximum[0] / width, maximum[1] / height)
    if scale < 1:
        image = cv2.resize(
            image, (max(1, round(width * scale)), max(1, round(height * scale)))
        )
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(data).decode("ascii") if ok else None


def decode_image(payload):
    if not payload:
        return None
    import cv2
    import numpy as np

    return cv2.imdecode(
        np.frombuffer(base64.b64decode(payload), np.uint8), cv2.IMREAD_COLOR
    )


class PurityDiagnostics:
    KINDS = frozenset(
        (
            "status",
            "report",
            "residue",
            "storage",
            "hand_model",
            "phase",
            "reference",
            "baseline",
            "begin",
            "contact",
            "mask",
            "mask_decision",
            "acid_area",
            "acid_detection",
            "captured",
            "result",
        )
    )

    def __init__(self):
        self.sequence = 0
        self.workflow = {}
        self.logs = deque(maxlen=40)

    def event(self, payload):
        kind = payload.get("kind")
        if kind not in self.KINDS:
            return
        self.sequence += 1
        if kind == "begin":
            for old in (
                "contact",
                "mask",
                "baseline",
                "mask_decision",
                "acid_area",
                "acid_detection",
                "captured",
                "result",
                "phase",
            ):
                self.workflow.pop(old, None)
        # Retain small display fields only, never arbitrary masks/full video arrays.
        value = {
            k: plain(payload[k])
            for k in (
                "kind",
                "message",
                "detail",
                "status",
                "path",
                "timestamp",
                "completed",
                "phase",
                "jewel_id",
                "remaining",
                "recording_assignment",
                "recording_candidate_id",
                "confirmed",
                "pending",
                "detected",
                "provisional",
                "waiting_for_baseline",
                "contamination_possible",
                "capture_complete",
                "residue_warning",
            )
            if k in payload
        }
        if kind == "result":
            record = payload["record"]
            value["record"] = {
                k: plain(record.get(k))
                for k in (
                    "prediction",
                    "jewel_id",
                    "cycle_id",
                    "capture_complete",
                    "close_reason",
                    "recording_assignment",
                    "recording_candidate_id",
                    "acid_onset",
                    "acid_target_status",
                    "rub_previous_acid_overlap",
                )
            }
        photo = payload.get("photo")
        if photo is not None:
            if kind == "mask" and payload.get("stone") is not None:
                x1, y1, x2, y2 = payload["stone"]
                photo = photo[max(0, y1 - 12) : y2 + 12, max(0, x1 - 12) : x2 + 12]
            value["image"] = image_payload(photo)
        self.workflow[kind] = dict(sequence=self.sequence, payload=value)

    def log(self, message, level):
        self.sequence += 1
        self.logs.append(
            dict(sequence=self.sequence, message=str(message), level=str(level))
        )

    def snapshot(self, worker):
        return dict(
            workflow=sorted(self.workflow.values(), key=lambda e: e["sequence"]),
            logs=list(self.logs),
            cycle_id=getattr(worker, "continuous_cycle_id", None),
            latest_token=getattr(worker, "continuous_latest_token", None),
            continuous_stage=plain(getattr(worker, "continuous_stage", {})),
            directory=str(
                getattr(getattr(worker, "continuous_monitor", None), "directory", "")
            ),
        )
