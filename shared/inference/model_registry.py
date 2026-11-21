"""Logical names and checksum-verified central asset paths; never downloads."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def registry():
    return json.loads((ROOT/'configs/model_registry.json').read_text())


def model_path(name, verify=False):
    entry = registry()['models'].get(name)
    if entry is None:
        raise KeyError('Unknown logical model: '+name)
    path = ROOT/entry['path']
    if not path.is_file():
        raise FileNotFoundError(f'Missing model {name}: {entry["path"]}')
    if verify and hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Model checksum mismatch: '+name)
    return path


def asset_dir(name):
    return ROOT/registry()['directories'][name]


def resolve_model(path):
    """Resolve registered historical references; explicit external inputs stay explicit."""
    if not isinstance(path, (str, Path)):
        return path  # ONNX bytes are a supported explicit input.
    value = str(path).replace('\\', '/')
    if value in registry()['models']:
        return model_path(value)
    candidate = Path(path)
    absolute = str(candidate.absolute())
    key = registry()['aliases'].get(absolute) or registry()['aliases'].get(value)
    if key:
        return model_path(key)
    # Canonical absolute paths are checked by logical name too.
    for name, entry in registry()['models'].items():
        if candidate.absolute() == ROOT/entry['path']:
            return model_path(name)
    return candidate


def validate(stage=None):
    errors=[]
    for name,entry in registry()['models'].items():
        if stage and stage not in entry.get('stages',[]):
            continue
        try:
            path=model_path(name, verify=True)
            if entry['format']=='xml' and not path.with_suffix('.bin').is_file():
                raise ValueError('OpenVINO BIN pair missing')
        except (OSError,ValueError) as exc:
            errors.append({'model':name,'error':str(exc)})
    return errors
