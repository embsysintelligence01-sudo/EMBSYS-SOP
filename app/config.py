"""Location-based configuration; no engine or model imports here."""
from pathlib import Path
import os
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        from pip._vendor import tomli as tomllib  # Existing Python 3.10 environments.

ROOT = Path(__file__).resolve().parents[1]
JEWEL = ROOT / 'engines/Jeweltracker'
PACKET = ROOT / 'engines/Packetsealing_001'
RELEASING = ROOT / 'engines/PacketReleasing'


def settings():
    result = tomllib.loads((ROOT / 'configs/app.toml').read_text())
    local = ROOT / 'configs/local.toml'
    if local.exists():
        result.update(tomllib.loads(local.read_text()))
    return result


def resolve_path(value):
    path = Path(value).expanduser()
    return Path(os.path.abspath(ROOT / path)) if not path.is_absolute() else path


def interpreter(workflow):
    data = settings()
    variable = {'purity': 'PURITY_PYTHON', 'packet': 'PACKET_PYTHON', 'releasing': 'RELEASING_PYTHON'}[workflow]
    path = resolve_path(os.environ.get(variable) or data[workflow + '_python'])
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f'{variable} interpreter is unavailable. Run scripts/install.sh or configure configs/local.toml.')
    return path


def engine_environment(workflow):
    env = os.environ.copy()
    python = interpreter(workflow)
    env.update(JEWEL_TRACKER_DIR=str(JEWEL), JEWEL_PACKET_DIR=str(PACKET),
               VIRTUAL_ENV=str(python.parent.parent),
               PATH=str(python.parent) + os.pathsep + env.get('PATH', ''),
               PYTHONNOUSERSITE='1', YOLO_AUTOINSTALL='False',
               OMP_WAIT_POLICY='PASSIVE', PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True')
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    for key in ('ocr_python', 'reviewer_dir'):
        value = settings().get(key)
        if value:
            env['PACKET_OCR_PYTHON' if key == 'ocr_python' else 'EMBSYS_REVIEWER_DIR'] = str(resolve_path(value))
    for key in ('QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        env.pop(key, None)
    if workflow == 'releasing':
        # Match the preserved engine's working Linux launcher, without importing Qt6.
        env.update(PACKET_CUT_MODELS_DIR=str(ROOT/'models/releasing'), WORKFLOW_DEVICE='cpu',
                   OMP_NUM_THREADS=env.get('OMP_NUM_THREADS', '4'),
                   MKL_NUM_THREADS=env.get('MKL_NUM_THREADS', '4'),
                   YOLO_CONFIG_DIR=str(ROOT/'runtime/PacketReleasing/ultralytics'),
                   PACKET_OCR_DET_MODEL=env.get('PACKET_OCR_DET_MODEL', 'PP-OCRv5_mobile_det'),
                   PACKET_OCR_REC_MODEL=env.get('PACKET_OCR_REC_MODEL', 'en_PP-OCRv5_mobile_rec'))
    if workflow=='releasing':
        from shared.inference.model_registry import asset_dir
        env['PACKET_OCR_DET_DIR']=str(asset_dir('PP-OCRv5_mobile_det'))
        env['PACKET_OCR_REC_DIR']=str(asset_dir('en_PP-OCRv5_mobile_rec'))
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    cache=ROOT/'runtime/model-cache/paddlex'
    if cache.is_dir():env['PADDLE_PDX_CACHE_HOME']=str(cache)
    return env


def command(workflow, *, development=False):
    if workflow == 'releasing':
        if not development:
            raise ValueError('Packet Releasing is development-only; production integration is not enabled')
        return [str(interpreter(workflow)), str(RELEASING/'automated_app/app.py'), '--cpu'], RELEASING
    if workflow not in ('purity', 'packet'):
        raise ValueError('Unknown workflow')
    directory = JEWEL if workflow == 'purity' else PACKET
    return [str(interpreter(workflow)), str(directory / ('jewel_tracker_5.py' if workflow == 'purity' else 'packet_app.py'))], directory
