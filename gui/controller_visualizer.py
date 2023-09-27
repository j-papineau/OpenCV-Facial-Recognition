from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QPointF, QRectF, QLineF
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QPixmap, QColor, QImage, QPainter
from PyQt6 import uic
import os
import cv2
import numpy as np
from enum import Enum

class Controller_Visualizer(QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.input = QVBoxLayout()
        self.label = QLabel("Controller Visualizer")
        self.layout.addWidget(self.label, 0 , 0)
        

class Direction(Enum):
    Left = 0
    Right = 1
    Up = 2
    Down = 3
