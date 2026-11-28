"""Original shared/inference/gold_dispatcher.py definitions.

Loaded into tracker globals by the explicit temporary compatibility binder.
The source bodies are unchanged; see docs/IMPORT_PATH_MIGRATION.md.
"""

class SharedGoldModel:
    """One predictor for full-frame live observations and still captures.

    A live result supplies both tracking and stone-region masking; stone
    masking never submits a second prediction.
    """
    def __init__(self, path=MODEL_PATH):
        self.backend = OpenVINOYolo(path, task="segment", imgsz=320)
        self.lock = threading.Lock()
        self.dispatcher = InferenceDispatcher(self._predict)

    def _predict(self, source, **kwargs):
        with self.lock:
            return self.backend.predict(source, **kwargs)

    def predict(self, source, **kwargs):
        kind = kwargs.pop('_live_job', 'bench')
        captured_at = kwargs.pop('_captured_at', None)
        expires_at = (time.monotonic() + LIVE_RESULT_MAX_AGE_S - (time.time() - captured_at)
                      if captured_at is not None else None)
        kwargs['imgsz'] = 320
        kwargs['stream'] = False
        return self.dispatcher.submit(kind, source, kwargs, expires_at).result()

    def close(self):
        self.dispatcher.close()

