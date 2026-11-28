"""Helpers for tracking and counting gold instances after a confirmed cut."""


class JewelTracker:
    def __init__(self, model, device="cpu", confidence=0.25):
        self.model = model
        self.device = device
        self.confidence = confidence
        self.seen_track_ids = set()

    @property
    def count(self):
        return len(self.seen_track_ids)

    def update(self, frame):
        """Return an annotated frame, current visible count, and cumulative count."""
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        visible_ids = set()
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.id is not None:
            visible_ids = {
                int(track_id)
                for track_id in boxes.id.detach().cpu().tolist()
            }
            self.seen_track_ids.update(visible_ids)

        annotated = result.plot()
        return annotated, len(visible_ids), self.count
