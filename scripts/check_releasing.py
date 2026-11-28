#!/usr/bin/env python3
"""Camera-free checks, run with the Packet Releasing interpreter."""
from pathlib import Path
import argparse
import importlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'engines/PacketReleasing'
sys.path[:0] = [str(ENGINE/'automated_app'), str(ENGINE)]
os.chdir(ENGINE)
os.environ.setdefault('PACKET_CUT_MODELS_DIR', str(ENGINE/'models'))
os.environ.setdefault('WORKFLOW_DEVICE', 'cpu')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('PACKET_OCR_REC_MODEL', 'en_PP-OCRv5_mobile_rec')
os.environ.setdefault('PACKET_OCR_DET_MODEL', 'PP-OCRv5_mobile_det')
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'runtime/PacketReleasing/ultralytics'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('imports', 'ui', 'models'))
    args = parser.parse_args()
    if args.mode == 'imports':
        modules = ['packet_cut.desktop.app', 'packet_cut.desktop.automation',
                   'packet_cut.desktop.release_observer', 'packet_cut.desktop.ocr',
                   'packet_cut.desktop.capture', 'packet_cut.cli.ocr',
                   'inspection.post_cut_inspection', 'workflow_monitor.automated_engine',
                   'workflow_monitor.pdf_report', 'workflow_monitor.visual_report',
                   'packet_mouth_detection', 'packet_cut_core']
        for name in modules:
            module = importlib.import_module(name)
            assert Path(module.__file__).resolve().is_relative_to(ENGINE), (name, module.__file__)
        from workflow_monitor.config import WorkflowMonitorConfig
        config = WorkflowMonitorConfig.load()
        for key in ('packet', 'hand', 'scissor', 'topstrip', 'number', 'cut', 'jewel'):
            path = config.models.path_for(key)
            assert path.is_file() and path.resolve().is_relative_to((ENGINE/'models').resolve()), (key, path)
        print('PASS: imports resolve inside the copy; seven local model paths verified')
    elif args.mode == 'ui':
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QLibraryInfo
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
        from packet_cut.desktop.widgets import MainWindow
        application = QApplication([])
        window = MainWindow()
        window.show()
        application.processEvents()
        assert window.worker is None, 'UI started acquisition unexpectedly'
        assert window.close()
        application.processEvents()
        print('PASS: original PyQt6 UI constructed and closed; no camera opened')
    else:
        import cv2
        import numpy as np
        from workflow_monitor.models import UnifiedModelManager
        from workflow_monitor.config import WorkflowMonitorConfig
        from packet_cut.desktop.ocr import run_number_ocr
        manager = UnifiedModelManager(WorkflowMonitorConfig.load())
        manager.load_all_models()
        required = ('packet', 'hand', 'scissor', 'topstrip', 'number', 'cut', 'jewel', 'ocr')
        assert all(manager.has(key) for key in required), manager.missing
        blank = np.full((480, 640, 3), 230, np.uint8)
        for key in required[:-1]:
            manager.get(key).predict(source=blank, imgsz=640 if key == 'number' else 320,
                                     device='cpu', verbose=False)
        crop = np.full((65, 240, 3), 255, np.uint8)
        cv2.putText(crop, '324871', (8, 47), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        text, confidence, _ = run_number_ocr(crop)
        assert text == '324871' and confidence >= .65, (text, confidence)
        print('PASS: seven vision models loaded and inferred; OCR read', text, confidence)


if __name__ == '__main__':
    main()
