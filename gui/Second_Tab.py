from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QFile
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QPixmap, QColor
from PyQt6 import uic
import os
import numpy as np

# from ..hand_tracking import *


class Second_Tab(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        uic.loadUi(os.path.join(os.path.dirname(__file__), "ui-files/untitled.ui"), self)
        self.initUI()
  

    def handle_button_1(self):
        self.label_1.setText("Hello")
        # self.label_2 = QLabel("penis")
        # self.test_layout.addWidget(self.label_2, 0, 0)

        for i in range(0,10):
            label = QLabel(f'{str(i)} penis')
            self.test_layout.addWidget(label, i, 0)

    def initUI(self):
        self.label_1.setText("pee pee")
        self.button_1.clicked.connect(self.handle_button_1)
        
    