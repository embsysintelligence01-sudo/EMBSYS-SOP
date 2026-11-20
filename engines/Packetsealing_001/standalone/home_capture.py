"""Short observation window for initial homes, independent of mask stillness."""
import math

import numpy as np


def same_region(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    overlap = np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(overlap))
    smaller = min(float(np.prod(a[2:]-a[:2])), float(np.prod(b[2:]-b[:2])))
    return smaller > 0 and intersection / smaller >= .5


class HomeCaptureTrackerMixin:
    """Use current masks after one second; keep the original tracker otherwise."""
    def try_lock(self, dets, t, frame=None):
        if not getattr(self, 'fast_home_capture', False):
            return super().try_lock(dets, t, frame)
        current = [d for d in dets if self.roi_dist(d.c) > 0 and not self.on_stone(d.c, margin=0)]
        self.arm_wait_stone = False
        if self.arm_start is None:
            if not current:
                self.arm_count = 0
                return
            self.arm_start = t
        self.arm_frames.append((t, current))
        # Keep the timer across box jitter, changing counts and missed detections.
        # Retain two samples even when CPU inference takes over one second.
        while len(self.arm_frames) > 2 and t-self.arm_frames[0][0] > 1.:
            self.arm_frames.popleft()
        required = max(2, math.ceil(len(self.arm_frames)*.5))
        supported = [d for d in current if sum(
            any(same_region(d.box, old.box) for old in observations)
            for _, observations in self.arm_frames) >= required]
        self.arm_count = len(current)
        self.stable = min(1., max(0., t-self.arm_start))  # UI progress, not stability.
        if frame is None or t-self.arm_start < 1. or not supported:
            return
        # Current detections match the capture frame; do not freeze an older mask.
        jewels = self.capture(frame, supported)
        if jewels:
            self.lock_jewels(jewels, t, frame)
            self.event.emit('Initial jewel IDs/crops ready; one-second home observation complete.', 'info')
