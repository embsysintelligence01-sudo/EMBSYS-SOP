"""Explicit camera-free checks; never changes runtime detection behavior."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import JEWEL, PACKET
sys.path[:0] = [str(JEWEL), str(PACKET)]
os.environ.setdefault('YOLO_AUTOINSTALL', 'False')

mode = sys.argv[1]
if mode in ('imports', 'imports-purity'):
    import jewel_tracker_5
    assert Path(jewel_tracker_5.__file__).resolve().parent == JEWEL
    import packet_app
    assert Path(packet_app.__file__).resolve().parent == PACKET
    if mode != 'imports-purity':
        import standalone.app
    import acid_test.integration
    import backup_recovery, audit_report, evidence_archive
    print('PASS: main engines, camera, evidence, recovery and reporting imports')
elif mode == 'purity-ui':
    import jewel_tracker_5 as tracker
    from unittest.mock import patch
    application = tracker.QtWidgets.QApplication([])
    # MainWindow normally auto-connects via a zero-delay timer. Suppress that
    # connection only in this test so no camera is opened.
    with patch.object(tracker.MainWindow, 'toggle_connect', lambda self: None):
        window = tracker.MainWindow()
        window.show()
        application.processEvents()
        assert window.close(), 'Window did not shut down cleanly'
        application.processEvents()
    print('PASS: purity UI constructed and closed without camera')
elif mode == 'purity-models':
    import numpy as np
    import jewel_tracker_5 as tracker
    from acid_test.integration import AcidModelCache
    from int8_models import OpenVINOYolo, RUBBED_MODEL
    blank = np.full((640, 640, 3), 230, np.uint8)
    tracker.StoneOnnxModel(tracker.STONE_MODEL_PATH).predict(blank)
    OpenVINOYolo(RUBBED_MODEL, task='detect', imgsz=640).predict(blank)
    cache = AcidModelCache()
    cache.get()
    cache.get_ph()
    cache.close()
    from jewel_classifier import JewelClassifier
    classifier = JewelClassifier(tracker.SIGLIP_MODEL)
    print('PASS: stone, rubbed mask, temporal acid detector/classifier, pH and SigLIP models loaded')
else:
    raise SystemExit('Unknown check')
