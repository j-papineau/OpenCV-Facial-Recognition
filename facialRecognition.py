import argparse
import string
import threading
from pathlib import Path

import cv2
import face_recognition
import pickle
from collections import Counter
import time
from PIL import Image, ImageDraw
import numpy as np
import os
from sys import platform
from recognizeChad import *

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "red"
TEXT_COLOR = "white"

parser = argparse.ArgumentParser(description="Facial Recognition Project")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument("--train_emotion", action="store_true", help="Train on input data for emotions")
parser.add_argument(
    "--validate", action="store_true", help="Run Validation Script (many windows will open)"
)
parser.add_argument(
    "--test", action="store_true", help="Test with unknown image (png or jpg)"
)
parser.add_argument(
    "--doiwork", action="store_true", help="makes sure project is able to be run"
)
parser.add_argument(
    "--live", action="store_true", help="Runs Live Facial Recognition"
)
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: HOG(cpu intensive) preferred"
)
parser.add_argument(
    "-f", action="store", help="Image Path"
)
args = parser.parse_args()

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)


# encodes ALL face from training data
def encode_known_faces(
        model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

# takes in unknown encoding and loaded encodings and compares
# returns likely result?
def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    # TODO: possibly could implement "Could also be..." here
    if votes:
        # for item in votes:
        #     print(item)
        return votes.most_common(1)[0][0]


def validate(model: str = "hog"):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )

# optimize to take loaded_encodings as a param
def recognize_multiple_faces_live():
    with DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
    
    print("attempting to start video captue")

    # OS camera statement
    if platform == "darwin":
        camera = cv2.VideoCapture(0)
    elif platform == "win32":
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise IOError("Cannot open video source")
    
    # camera.set(cv2.CAP_PROP_FPS, 20)
    
    pTime = 0
    counter = 0

    # load known encodings from pickle

    if loaded_encodings:
        print("facial encodings loaded")

    face_data = []
    
    while True:
        
        ret, frame = camera.read()

        frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        
        if counter % 30 == 0:
            face_data = recognize_multiple_faces(frame, loaded_encodings)
        else:
            pass
        counter += 1

        if counter > 100000:
            counter = 0

        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)

        if face_data:
            for face in face_data:
                name = face[0]
                box = face[1]
                top, right, bottom, left = box
                cv2.rectangle(frame, (left * 2,top * 2), (right * 2, bottom * 2), (0, 50, 255), 2)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(frame, name, (left * 2, bottom * 2), font, 0.5, (255,255, 255), 1, cv2.LINE_AA)
        
        # fps calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
    
        cv2.putText(frame, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("video", frame)
        c = cv2.waitKey(1)
        if c == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
    

def recognize_multiple_faces(
        frame,
        encodings,
        model: str = "hog",
) -> None:
    
    face_data = []

    input_face_locations = face_recognition.face_locations(
        frame, model=model
    )   
    input_face_encodings = face_recognition.face_encodings(
        frame, input_face_locations
    )

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, encodings)
        if not name:
            name = "Unknown"
        if not bounding_box:
            bounding_box = ""
        face_data.append([name, bounding_box])
    return face_data


def only_facial_recognition():
    print("attempting to start video captue")

    # OS camera statement
    if platform == "darwin":
        camera = cv2.VideoCapture(0)
    elif platform == "win32":
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise IOError("Cannot open video source")
    
    # camera.set(cv2.CAP_PROP_FPS, 20)
    
    pTime = 0
    counter = 0

    # load known encodings from pickle


    face_data = []
    

    while True:
        
        ret, frame = camera.read()

        frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        
        if counter % 2 == 0:
            face_data = only_recognize_face(frame)
        else:
            pass
        counter += 1
        
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
        
        if face_data:
            for bounding_box in face_data:
                top, right, bottom, left = bounding_box
                cv2.rectangle(frame, (left * 2,top * 2), (right * 2, bottom * 2), (0, 50, 255), 2)
                
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
    
        cv2.putText(frame, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("video", frame)
        c = cv2.waitKey(1)
        if c == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
            

def only_recognize_face(frame, model="hog") -> None:
    input_face_locations = face_recognition.face_locations(
        frame, model=model
    )
    
    return input_face_locations
    

if __name__ == "__main__":
    
    recognize_multiple_faces_live()
