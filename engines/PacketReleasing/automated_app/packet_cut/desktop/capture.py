"""Live camera acquisition independent of model inference, with a one-frame slot."""

import threading
import time


class LatestFrameCapture(threading.Thread):
    def __init__(self, camera, preview_callback=None, preview_fps=15):
        super().__init__(name='packet-live-camera', daemon=True)
        self.camera = camera
        self.preview_callback = preview_callback
        self.preview_interval = 1 / max(1, preview_fps)
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._latest = None
        self._sequence = 0
        self.closed = False
        self.error = None
        self.preview_rate = 0.0

    def run(self):
        next_preview = None
        rate_start = time.monotonic()
        preview_count = 0
        failures = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = self.camera.read()
                if self._stop_event.is_set():
                    break
                if not ok or frame is None:
                    failures += 1
                    if failures >= 30:
                        self.error = 'Camera stream lost after 30 consecutive read failures.'
                        break
                    self._stop_event.wait(.05)
                    continue
                failures = 0
                now = time.monotonic()
                with self._condition:
                    self._sequence += 1
                    self._latest = (self._sequence, frame, now)
                    self._condition.notify_all()
                if self.preview_callback is not None and (next_preview is None or now >= next_preview - .001):
                    self.preview_callback(frame)
                    next_preview = now + self.preview_interval if next_preview is None else next_preview + self.preview_interval
                    if next_preview <= now:
                        next_preview = now + self.preview_interval
                    preview_count += 1
                if now - rate_start >= 1:
                    self.preview_rate = preview_count / (now - rate_start)
                    preview_count = 0
                    rate_start = now
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            try:
                self.camera.release()
            finally:
                with self._condition:
                    self.closed = True
                    self._condition.notify_all()

    def next_frame(self, after_sequence, timeout=.3):
        with self._condition:
            self._condition.wait_for(lambda: self._sequence > after_sequence or self.closed or self._stop_event.is_set(), timeout)
            if self._latest is not None and self._latest[0] > after_sequence:
                return self._latest
            return None

    def stop(self):
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
