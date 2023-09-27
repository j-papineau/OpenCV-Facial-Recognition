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
        
        self.center = QPushButton("C")
        self.center.setStyleSheet("background-color: green;")
        self.center.resize(50,50)
        self.center.setMaximumWidth(50)
        
        self.up = QPushButton("U")
        self.up.resize(50,50)
        self.up.setMaximumWidth(50)
        
        self.down = QPushButton("D")
        self.down.resize(50,50)
        self.down.setMaximumWidth(50)
        
        self.right = QPushButton("R")
        self.right.resize(50,50)
        self.right.setMaximumWidth(50)
        
        self.left = QPushButton("L")
        self.left.resize(50,50)
        self.left.setMaximumWidth(50)
        
        self.clear_colors()
        
        
        
        
        self.layout.addWidget(self.up, 0, 1)
        self.layout.addWidget(self.down, 2, 1)
        self.layout.addWidget(self.right, 1, 2)
        self.layout.addWidget(self.left, 1, 0)
        self.layout.addWidget(self.center, 1, 1)
        
        # self.layout.addWidget()
    
    def clear_colors(self):
        self.center.setStyleSheet("background-color: gray;")
        self.up.setStyleSheet("background-color: gray;")
        self.down.setStyleSheet("background-color: gray;")
        self.left.setStyleSheet("background-color: gray;")
        self.right.setStyleSheet("background-color: gray;")
        
    def set_up(self):
        self.clear_colors()
        self.up.setStyleSheet("background-color: green;")
    
    def set_down(self):
        self.clear_colors()
        self.down.setStyleSheet("background-color: green;")
        
    def set_left(self):
        self.clear_colors()
        self.left.setStyleSheet("background-color: green;")
        
    def set_right(self):
        self.clear_colors()
        self.right.setStyleSheet("background-color: green;")
        
    def set_center(self):
        self.clear_colors()
        self.center.setStyleSheet("background-color: green;")
        

class Direction(Enum):
    Left = 0
    Right = 1
    Up = 2
    Down = 3
