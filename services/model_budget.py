"""Per-host scheduling policy only; model input/output and evidence rules are untouched."""

import os


def apply_cpu_budget(threads):
    import cv2
    import torch
    import onnxruntime as ort
    import ultralytics.utils.torch_utils as torch_utils

    threads = max(1, int(threads))
    torch.set_num_threads(threads)
    torch_utils.NUM_THREADS = threads
    cv2.setNumThreads(threads)
    base = ort.InferenceSession

    class BudgetSession(base):
        def __init__(self, path_or_bytes, sess_options=None, *args, **kwargs):
            options = sess_options or ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            for key in (
                "session.intra_op.allow_spinning",
                "session.inter_op.allow_spinning",
            ):
                try:
                    current = options.get_session_config_entry(key)
                except RuntimeError:
                    current = None
                if current != "0":
                    options.add_session_config_entry(key, "0")
            super().__init__(path_or_bytes, options, *args, **kwargs)

    ort.InferenceSession = BudgetSession
