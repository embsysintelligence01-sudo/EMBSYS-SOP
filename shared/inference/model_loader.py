"""Resolve registered assets without altering runtime options/preprocessing."""
from .model_registry import model_path, resolve_model, validate


def onnx_session(path, *args, **kwargs):
    import onnxruntime
    return onnxruntime.InferenceSession(str(resolve_model(path)), *args, **kwargs)


def load_checkpoint(path, *args, **kwargs):
    import torch
    return torch.load(resolve_model(path), *args, **kwargs)


def require_models(stage):
    errors=validate(stage)
    if errors:
        raise RuntimeError('Model validation failed: '+ '; '.join(e['model']+': '+e['error'] for e in errors))
