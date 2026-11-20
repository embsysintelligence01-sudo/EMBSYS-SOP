#!/usr/bin/env python3
import sys
from PyQt5 import QtWidgets
from app.main_window import MainWindow


def main():
    application = QtWidgets.QApplication(sys.argv[:1])
    application.setApplicationName('EMBSYS SOP')
    window = MainWindow()
    window.show()
    if '--smoke-ui' in sys.argv:
        application.processEvents()
        window.close()
        return 0
    return application.exec_()


if __name__ == '__main__':
    sys.exit(main())
