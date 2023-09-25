from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QPixmap, QColor, QImage
from PyQt6 import uic
import os
import cv2
import numpy as np
from hand_tracking import *
import mediapipe as mp
from keras.models import load_model


# from ..hand_tracking import *


class Main_Tab(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layout = QGridLayout()
        

        self.FeedLabel = QLabel()
        self.layout.addWidget(self.FeedLabel, 0, 0)

        # buttons
        self.do_hand_tracking = False
        self.do_gesture_tracking = False

        self.cancel_btn = QPushButton("Stop Camera")
        self.cancel_btn.clicked.connect(self.cancel_feed)

        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_feed)

        # sidebar for options

        self.sidebar_layout = QFormLayout()

        self.sidebar_label = QLabel("Options")
        self.sidebar_layout.addWidget(self.sidebar_label)

        
        self.hand_tracking_check = QCheckBox("Hand Tracking")
        self.hand_tracking_check.setChecked(True)
        self.hand_tracking_check.stateChanged.connect(self.handle_hand_check)
        self.sidebar_layout.addWidget(self.hand_tracking_check)

        self.gesture_check = QCheckBox("Gesture Recognition")
        self.gesture_check.setChecked(False)
        self.gesture_check.stateChanged.connect(self.handle_gesture_check)
        self.sidebar_layout.addWidget(self.gesture_check)

        # bottom buttons

        self.layout.addWidget(self.start_btn, 1, 0)
        self.layout.addWidget(self.cancel_btn, 1, 1)
        self.layout.addLayout(self.sidebar_layout, 0, 1)

        self.camera = Camera_Thread_Hands()
        self.camera.ImageUpdate.connect(self.ImageUpdateSlot)
        self.camera.start()

        # bottom layout for output
        self.bottom_layout = QGridLayout()
        self.bottom_label = QLabel("Output (For Gestures)")
        self.bottom_out_label = QLabel("None")
        self.bottom_out_label.setStyleSheet("font-size:24pt")
        self.bottom_layout.addWidget(self.bottom_label)
        self.bottom_layout.addWidget(self.bottom_out_label)

        self.layout.addLayout(self.bottom_layout, 2, 0)

        self.setLayout(self.layout)

    def ImageUpdateSlot(self, img):
        self.FeedLabel.setPixmap(QPixmap.fromImage(img))

    def GestureUpdateSlot(self, gestures):
        print(gestures)
        self.bottom_out_label.setText(gestures[0])
    
    def hand_pos_update_slot(self, hand_pos):
        print(hand_pos)

    def handle_hand_check(self):
        self.do_hand_tracking = self.hand_tracking_check.isChecked()

        if self.do_hand_tracking:
            self.camera.stop()
            self.camera = Camera_Thread_Hands()
            self.camera.ImageUpdate.connect(self.ImageUpdateSlot)
            self.camera.start()
        else:
            self.camera.stop()
            self.camera = Camera_Thread()
            self.camera.ImageUpdate.connect(self.ImageUpdateSlot)
            self.camera.start()

    def handle_gesture_check(self):
        self.do_gesture_tracking = self.gesture_check.isChecked()

        if self.do_gesture_tracking:
            self.camera.stop()
            self.camera = Camera_Thread_Gesture()
            self.camera.ImageUpdate.connect(self.ImageUpdateSlot)
            self.camera.GestureUpdate.connect(self.GestureUpdateSlot)
            self.camera.start()
        else:
            self.camera.stop()
            self.camera = Camera_Thread()
            self.camera.ImageUpdate.connect(self.ImageUpdateSlot)
            self.camera.start()

    def cancel_feed(self):
        self.camera.stop()

    def start_feed(self):
        self.camera.start()

class Camera_Thread(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        cap = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.flip(image, 1)
                qt_format = QImage(img.data, img.shape[1], img.shape[0], QImage.Format.Format_RGB888)
                pic = qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate.emit(pic)

    def stop(self):
        self.ThreadActive = False
        self.quit()

class Camera_Thread_Hands(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        cap = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_2 = cv2.flip(image, 1)
                img = recognize_hands(image_2)
                qt_format = QImage(img.data, img.shape[1], img.shape[0], QImage.Format.Format_RGB888)
                pic = qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate.emit(pic)
                

    def stop(self):
        self.ThreadActive = False
        self.quit()

class Camera_Thread_Gesture(QThread):
    ImageUpdate = pyqtSignal(QImage)
    GestureUpdate = pyqtSignal(list)
    def run(self):
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
            # print('gesture recognition result: {}'.format(result))
            for gesture in result.gestures:
                # print([category.category_name for category in gesture])
                self.GestureUpdate.emit([category.category_name for category in gesture])

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path='./gui/gesture_recognizer.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=print_result)
        with GestureRecognizer.create_from_options(options) as recognizer:

            camera = cv2.VideoCapture(0)
            pTime = 0
            framerate = camera.get(cv2.CAP_PROP_FPS)
            timestamp = 0
            while True:
                ret, frame = camera.read()
                # timestamp = mp.Timestamp.from_seconds(time.time())
                pTime = time.time()
                if ret:
                    timestamp += 1
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    recognizer.recognize_async(mp_image, timestamp)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = cv2.flip(image, 1)
                    qt_format = QImage(img.data, img.shape[1], img.shape[0], QImage.Format.Format_RGB888)
                    pic = qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                    self.ImageUpdate.emit(pic)
        
    def stop(self):
        self.ThreadActive = False
        self.quit()

        
    