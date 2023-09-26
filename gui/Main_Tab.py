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
# from keras.models import load_model
import pyautogui


# from ..hand_tracking import *


class Main_Tab(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layout = QGridLayout()

        self.screenWidth, self.screenHeight = pyautogui.size()
        

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

        self.sidebar_label = QLabel("Hand Pos:")
        self.sidebar_layout.addWidget(self.sidebar_label)

        self.sidebar_layout.addWidget(QLabel("X"))
        self.x_pos = QLineEdit()
        self.sidebar_layout.addWidget(self.x_pos)

        self.sidebar_layout.addWidget(QLabel("Y"))
        self.y_pos = QLineEdit()
        self.sidebar_layout.addWidget(self.y_pos)

        self.sidebar_layout.addWidget(QLabel("Z"))
        self.z_pos = QLineEdit()
        self.sidebar_layout.addWidget(self.z_pos)

        self.claw_indicator = QCheckBox("Close Activation")
        self.sidebar_layout.addWidget(self.claw_indicator)

        self.sidebar_layout.addWidget(QLabel("Command"))
        self.command_label = QLabel("None")
        self.command_label.setStyleSheet("font-size: 24pt;")
        self.sidebar_layout.addWidget(self.command_label)


        # bottom buttons

        self.layout.addWidget(self.start_btn, 1, 0)
        self.layout.addWidget(self.cancel_btn, 1, 1)
        self.layout.addLayout(self.sidebar_layout, 0, 1)

        self.camera = Camera_Thread_Gesture()
        self.camera.ImageUpdate.connect(self.ImageUpdateSlot)
        self.camera.GestureUpdate.connect(self.GestureUpdateSlot)
        self.camera.HandPosUpdate.connect(self.HandPosUpdateSlot)
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
        # print(gestures)
        # none for no hands {}, for no gesture, or gesture[0] most likely
        self.bottom_out_label.setText(gestures[0])

        if gestures[0] == "Closed_Fist":
            self.claw_indicator.setChecked(True)
        else:
            self.claw_indicator.setChecked(False)

    def HandPosUpdateSlot(self, landmarks):
        # print(landmarks)
        # calc is being done in thread for mouse pos
        # print(landmarks)
        if landmarks:
            command_string = ""
            self.x_pos.setText(str(round(landmarks["x"], 2)))
            self.y_pos.setText(str(round(landmarks["y"], 2)))
            self.z_pos.setText(str(round(landmarks["z"], 2)))

            # check x first
            # can calculate "rightness-factor" here

            if landmarks["x"] > .60:
                command_string = "Right"
            elif landmarks["x"] < .40:
                command_string = "Left"
            else:
                command_string = "None"

            if landmarks["y"] > .70:
                command_string += "-Down"
            elif landmarks["y"] < .30:
                command_string += "-Up"
            else:
                command_string += "-None"

            self.command_label.setText(command_string)
            self.command_label.adjustSize()
            # pyautogui.moveTo(mouseX, mouseY)
        
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
    HandPosUpdate = pyqtSignal(dict)
    
    def run(self):
        self.ThreadActive = True
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
            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands


            camera = cv2.VideoCapture(0)
            with mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as hands:
                
                pTime = 0
                framerate = camera.get(cv2.CAP_PROP_FPS)
                timestamp = 0
                while self.ThreadActive:
                    ret, frame = camera.read()
                    # timestamp = mp.Timestamp.from_seconds(time.time())
                    pTime = time.time()
                    if ret:
                        timestamp += 1
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                        recognizer.recognize_async(mp_image, timestamp)
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = cv2.flip(image, 1)
                        # draws hand ladmarks
                        results = hands.process(img)

                        result = results.multi_hand_landmarks
                        frameHeight, frameWidth, _ = frame.shape

                        # looping through hands causes slowdown

                        if result:
                            for hand in result:
                                mp_drawing.draw_landmarks(img, hand)
                                landmarks = hand.landmark
                                for id, landmark in enumerate(landmarks):
                                    # index 8 is index fingie
                                    if id == 8:
                                        # x = int(landmark.x * frameWidth)
                                        # y = int(landmark.y * frameHeight)
                                        cv2.circle(img=img, center=(int(landmark.x * frameWidth), int(landmark.y * frameHeight)), radius=30, color=(0,255,255))
                                        
                                        coords = {
                                            'x': landmark.x,
                                            'y': landmark.y,
                                            'z': landmark.z
                                        }
                                        self.HandPosUpdate.emit(coords)
                                        # z

                        qt_format = QImage(img.data, img.shape[1], img.shape[0], QImage.Format.Format_RGB888)
                        pic = qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                        self.ImageUpdate.emit(pic)
        
    def stop(self):
        self.ThreadActive = False
        self.quit()

        
    