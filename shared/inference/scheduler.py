"""Bounded live inference scheduling and a preview with no frame queue."""
import threading
import time
from concurrent.futures import Future

import cv2


class StaleInference(RuntimeError):
    """A frame expired before its prediction could start."""


class InferenceDispatcher:
    """One model owner, two bounded pending slots, and priority for guarding.

    Callers wait for a result before submitting another frame. Mask decoding
    and tracking can overlap with the next prediction without sharing the
    mutable Ultralytics predictor across threads.
    """

    def __init__(self, predict):
        self.predict = predict
        self.condition = threading.Condition()
        self.pending = []
        self.closed = False
        self.thread = threading.Thread(target=self._run, name='gold-inference', daemon=True)
        self.thread.start()

    def submit(self, kind, source, kwargs, expires_at=None):
        future = Future()
        with self.condition:
            while len(self.pending) >= 2 and not self.closed:
                self.condition.wait()
            if self.closed:
                raise RuntimeError('Gold inference is stopped')
            self.pending.append((kind, source, kwargs, expires_at, future))
            self.condition.notify_all()
        return future

    def _run(self):
        while True:
            with self.condition:
                while not self.pending and not self.closed:
                    self.condition.wait()
                if self.closed:
                    return
                index = next((i for i, item in enumerate(self.pending) if item[0] == 'bench'), 0)
                _, source, kwargs, expires_at, future = self.pending.pop(index)
                self.condition.notify_all()
            if not future.set_running_or_notify_cancel():
                continue
            if expires_at is not None and time.monotonic() > expires_at:
                future.set_exception(StaleInference('Camera frame expired before inference'))
                continue
            try:
                future.set_result(self.predict(source, **kwargs))
            except Exception as exc:
                future.set_exception(exc)

    def close(self):
        with self.condition:
            self.closed = True
            pending, self.pending = self.pending, []
            self.condition.notify_all()
        for *_, future in pending:
            if not future.done():
                future.set_exception(RuntimeError('Gold inference is stopped'))
        self.thread.join()


class LatestPreview:
    """Share only the latest immutable overlay, never captured video frames.

    The GUI applies it to the newest camera frame without acquiring the
    tracking/model lock. Expired overlays cannot present old boxes as live.
    """

    def __init__(self, max_age=0.3):
        self.max_age = max_age
        self.lock = threading.Lock()
        self.overlay = None
        self.scaled_overlay = None

    def publish(self, image, captured_at):
        mask = cv2.bitwise_not(cv2.inRange(image, (0, 0, 0), (0, 0, 0)))
        with self.lock:
            self.overlay = (image, mask, captured_at)
            self.scaled_overlay = None

    def clear(self):
        with self.lock:
            self.overlay = None
            self.scaled_overlay = None

    def render(self, frame, captured_at, source_shape=None):
        """Compose at display size when the original camera shape is supplied.

        Cache resized overlays per observation; camera pixels and evidence are
        never resized here. A changed source shape still invalidates old boxes.
        """
        with self.lock:
            overlay = self.overlay
            scaled = self.scaled_overlay
        if overlay is not None:
            image, mask, ts = overlay
            if not 0 <= captured_at - ts <= self.max_age:
                return frame
            if source_shape is not None and tuple(image.shape[:2]) != tuple(source_shape[:2]):
                return frame
            if image.shape != frame.shape:
                if source_shape is None:
                    return frame
                size = (frame.shape[1], frame.shape[0])
                if scaled is None or scaled[0] is not overlay or scaled[1] != size:
                    # Area sampling preserves thin outlines that nearest-neighbor
                    # sampling could drop entirely as a box moves between pixels.
                    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
                    scaled = (overlay, size, image, mask)
                    with self.lock:
                        if self.overlay is overlay:
                            self.scaled_overlay = scaled
                image, mask = scaled[2:]
            cv2.copyTo(image, mask, frame)
        return frame
