"""Bounded, event-based video evidence from frames already used by inspection."""

from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


class SessionMediaRecorder:
    """Keep four seconds before and three seconds after each reported event.

    Samples are small JPEGs shared between the ring and pending clips. There is
    no second camera stream and no full-session recording. Video is 5 fps WebM;
    missing capture intervals hold the last observed frame, not invented motion.
    """

    def __init__(self, directory: Path, session_id: str, started_at: str,
                 started_mono: float, enabled: bool = True, fps: int = 5,
                 pre_seconds: float = 4, post_seconds: float = 3):
        self.directory = Path(directory)
        self.session_id = session_id
        self.started_at = datetime.fromisoformat(started_at)
        self.started_mono = started_mono
        self.enabled = enabled
        self.fps = fps
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.buffer = deque(maxlen=int((pre_seconds + 1) * fps) + 1)
        self.pending: Dict[str, Dict[str, Any]] = {}
        self.completed: Dict[str, Dict[str, Any]] = {}
        self.events: Dict[str, Dict[str, Any]] = {}
        self.errors = deque(maxlen=20)
        self.last_sample = float('-inf')
        self.now = started_mono

    def _utc(self, timestamp):
        return (self.started_at + timedelta(seconds=timestamp - self.started_mono)).isoformat()

    def record_frame(self, frame: np.ndarray, timestamp: float):
        if not self.enabled or frame is None or frame.size == 0:
            return
        self.now = timestamp
        if timestamp - self.last_sample < 1 / self.fps - 1e-6:
            return
        try:
            height, width = frame.shape[:2]
            scale = min(1.0, 640 / width, 480 / height)
            size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
            small = cv2.resize(frame, size)
            ok, encoded = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                raise RuntimeError('Could not buffer video frame')
            sample = (timestamp, encoded.tobytes())
            self.buffer.append(sample)
            self.last_sample = timestamp
            for key in list(self.pending):
                clip = self.pending[key]
                if timestamp <= clip['end']:
                    clip['samples'].append(sample)
                if timestamp >= clip['end']:
                    # Freeze a bounded JPEG window; encode only when STOP saves
                    # the session, so VP8 cannot stall live inspection.
                    self._close_window(key, clip['end'])
        except Exception as exc:
            self.errors.append(f'Video capture: {type(exc).__name__}: {exc}')

    def event(self, key: str, image_path: str, metadata: Dict[str, Any]):
        if not self.enabled:
            return
        record = {
            'key': key, 'session_id': self.session_id, 'timestamp': self._utc(self.now),
            'image_path': image_path, 'video_path': None, 'video_status': 'PENDING',
            **metadata,
        }
        self.events[key] = record
        self.completed.pop(key, None)
        samples = [s for s in self.buffer if s[0] >= self.now - self.pre_seconds]
        # Replacing a count event retains only evidence matching the latest count.
        self.pending[key] = {'samples': samples, 'end': self.now + self.post_seconds}

    def clear_event(self, key: str):
        self.pending.pop(key, None)
        self.completed.pop(key, None)
        self.events.pop(key, None)

    def _close_window(self, key, end):
        clip = self.pending.pop(key)
        clip['finished_end'] = end
        self.completed[key] = clip
        self.events[key]['video_status'] = 'READY_TO_SAVE'

    def _finish(self, key: str, end: float):
        clip = self.completed.pop(key)
        record = self.events[key]
        samples = clip['samples']
        if not samples:
            record['video_status'] = 'NO_FRAMES'
            return
        start = samples[0][0]
        end = max(start, min(end, clip['end']))
        writer = None
        path = self.directory / f'{self.session_id}_{key}.webm'
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            first = cv2.imdecode(np.frombuffer(samples[0][1], np.uint8), cv2.IMREAD_COLOR)
            height, width = first.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'VP80'), self.fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError('WebM/VP8 encoder unavailable')
            # Resample with real frame times so slow inference does not speed up
            # the recorded action. Repeated frames are identified in metadata.
            frame_count = max(1, int((end - start) * self.fps) + 1)
            sample_index = 0
            decoded_index = -1
            frame = first
            for index in range(frame_count):
                target = start + index / self.fps
                while sample_index + 1 < len(samples) and samples[sample_index + 1][0] <= target + 1e-6:
                    sample_index += 1
                if decoded_index != sample_index:
                    frame = cv2.imdecode(np.frombuffer(samples[sample_index][1], np.uint8), cv2.IMREAD_COLOR)
                    decoded_index = sample_index
                annotated = frame.copy()
                lines = [f"Packet {record.get('packet_id', 'UNRESOLVED')} | {key.replace('_', ' ')}",
                         f"Event result: {record.get('result', 'NOT_VERIFIED')}" + (f" | Count at event {record['count']}" if 'count' in record else ''),
                         self._utc(target)]
                cv2.rectangle(annotated, (0, 0), (width, min(height, 54)), (20, 25, 30), -1)
                for line_index, line in enumerate(lines):
                    text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)[0][0]
                    font_scale = .4 * min(1, (width - 10) / max(text_width, 1))
                    cv2.putText(annotated, line, (5, 14 + 16 * line_index), cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, (245, 245, 245), 1, cv2.LINE_AA)
                writer.write(annotated)
            writer.release()
            writer = None
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError('Video file was not written')
            record.update(video_path=str(path), video_status='SAVED',
                          clip_started_at=self._utc(start), clip_ended_at=self._utc(end),
                          duration_seconds=round(frame_count / self.fps, 2), fps=self.fps,
                          source_frames=len(samples), encoded_frames=frame_count,
                          timing_note='Last observed frame held between captured frames; no audio.')
        except Exception as exc:
            record['video_status'] = 'UNAVAILABLE'
            record['video_error'] = f'{type(exc).__name__}: {exc}'
            self.errors.append(f'{key}: {record["video_error"]}')
        finally:
            if writer is not None:
                writer.release()

    def finish(self, timestamp: float):
        for key in list(self.pending):
            self._close_window(key, min(timestamp, self.now))
        for key in list(self.completed):
            self._finish(key, self.completed[key]['finished_end'])
        return list(self.events.values())
