from PyQt5 import QtCore, QtWidgets
from .process_manager import ProcessManager


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('EMBSYS SOP')
        self.setMinimumSize(660, 420)
        self.manager = ProcessManager(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(40, 36, 40, 32)
        title = QtWidgets.QLabel('EMBSYS SOP')
        title.setObjectName('title')
        layout.addWidget(title)
        layout.addWidget(QtWidgets.QLabel('Select a standard operating procedure'))
        layout.addSpacing(24)
        self.purity = QtWidgets.QPushButton('PURITY TESTING')
        self.packet = QtWidgets.QPushButton('PACKET SEALING')
        for button in (self.purity, self.packet):
            button.setMinimumHeight(80)
            layout.addWidget(button)
        layout.addSpacing(18)
        self.status = QtWidgets.QLabel('Ready')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addWidget(QtWidgets.QLabel('Close the workflow window after evidence export to return here.'))
        self.purity.clicked.connect(lambda: self.manager.start('purity'))
        self.packet.clicked.connect(lambda: self.manager.start('packet'))
        self.manager.changed.connect(self.update_status)
        self.setStyleSheet('''QWidget {background:#f1f5f9; color:#172b4d; font-size:15px;}
            QLabel#title {font-size:30px; font-weight:700;}
            QPushButton {background:#174a73;color:white;border:0;border-radius:8px;font-weight:700;font-size:19px;}
            QPushButton:hover {background:#216391;}
            QPushButton:disabled {background:#a3b4c4;color:#edf2f7;}''')

    def update_status(self, text, active):
        self.status.setText(text)
        self.purity.setEnabled(not active)
        self.packet.setEnabled(not active)

    def closeEvent(self, event):
        if self.manager.active:
            self.status.setText('Running — close the workflow window and let evidence export finish first.')
            event.ignore()
        else:
            event.accept()
