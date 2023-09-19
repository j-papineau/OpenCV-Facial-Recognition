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

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "red"
TEXT_COLOR = "white"

parser = argparse.ArgumentParser(description="Facial Recognition Project")
parser.add_argument("--train", action="store_true", help="Train on input data")
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


# recognizes all faces
def recognize_faces(
        image_location: str,
        model: str = "hog",
        encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    # gets locations of faces in input image
    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    # gets encodings of faces in input image
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    # load image into pillow and create draw object
    # pillow_image = Image.fromarray(input_image)
    # draw = ImageDraw.Draw(pillow_image)

    # load image in cv2 img obj
    img = cv2.imread(image_location, cv2.IMREAD_COLOR)

    # compare new encodings to known encodings from training data
    # bounding box is perceived area around face detected
    for bounding_box, unknown_encoding in zip(
            input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        # prints output (duh)
        # print(name, bounding_box)
        _display_face(img, bounding_box, name)
    # del draw
    # pillow_image.show()


def recognize_face_live(
        frame,
        model: str = "hog",
        encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_face_locations = face_recognition.face_locations(
        frame, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        frame, input_face_locations
    )

    for bounding_box, unknown_encoding in zip(
            input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        display_face_live(frame, bounding_box, name)


def display_face_live(img, bounding_box, name):
    top, right, bottom, left = bounding_box
    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 4)
    # add name text
    font = cv2.FONT_HERSHEY_SIMPLEX
    uppercase_name = name.upper()
    # img = cv2.flip(img, 1)
    cv2.putText(img, uppercase_name, (left, bottom), font, 0.5, (200, 255, 155), 1, cv2.LINE_AA)
    # cv2.imshow("video", img)


def _display_face(img, bounding_box, name):
    # draw rectangle on face and stuff
    top, right, bottom, left = bounding_box
    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 6)

    # add name text
    font = cv2.FONT_HERSHEY_SIMPLEX
    uppercase_name = name.upper()
    # img = cv2.flip(img, 1)
    cv2.putText(img, uppercase_name, (left, bottom), font, 1, (200, 255, 155), 2, cv2.LINE_AA)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def live_facial_recognition():
    print("attempting to open video capture")
    
    # change camera operation based on OS
    
    if platform == "darwin":
        camera = cv2.VideoCapture(0)
    elif platform == "win32":
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    

    if not camera.isOpened():
        raise IOError("Cannot open video source")

    do_facial_recognition = True

    print("camera opened successfully")

    pTime = 0

    counter = 0

    while True:
        ret, frame = camera.read()

        frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        # use boolean to only do every other frame (cpu will die)
        if counter % 30 == 0:
            recognize_face_live(frame)
        else:
            if counter > 1230:
                counter = 0
            else:
                counter += 1

        # recognize_face_live(frame)

        # instantiate thread to speed up
        # threading.Thread(target=recognize_face_live, args=(frame.copy(),)).start()
        # frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
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


if __name__ == "__main__":

    global face_match
    global bounding_box

    if args.train:
        encode_known_faces(model=args.m)
    elif args.validate:
        validate(model=args.m)
    elif args.test:
        recognize_faces(image_location=args.f, model=args.m)
    elif args.doiwork:
        print("yes, i am working")
    elif args.live:
        live_facial_recognition()
    else:
        # general run format (no args)
        user_input = int(input("Mode (1 for img, 2 for live): "))
        if user_input == 1:
            user_input = input("Enter Image Path: ")
            recognize_faces(image_location=user_input)
        elif user_input == 2:
            live_facial_recognition()
        else:
            print("Invalid input")
