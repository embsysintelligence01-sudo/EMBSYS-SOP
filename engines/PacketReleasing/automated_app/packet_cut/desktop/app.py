"""App module."""

from ._imports import *

from .config import *

from .widgets import MainWindow
from PyQt6.QtCore import QLibraryInfo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default=0, help="Camera index or video file path")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    camera_src = int(args.camera) if str(args.camera).isdigit() else args.camera

    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = int(args.device) if str(args.device).isdigit() else args.device

    # OpenCV's Linux wheel sets a Qt 5 plugin path; this UI uses Qt 6.
    if sys.platform.startswith("linux"):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.path(
            QLibraryInfo.LibraryPath.PluginsPath
        )
        if "cv2" in os.environ.get("QT_QPA_FONTDIR", ""):
            os.environ.pop("QT_QPA_FONTDIR", None)
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow(camera_src, device)
    window.show()
    sys.exit(app.exec())
