import cv2
import mediapipe as mp
from keras.models import load_model
import time

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # print('gesture recognition result: {}'.format(result))
    for gesture in result.gestures:
        print([category.category_name for category in gesture])

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

        cv2.imshow("video", frame)
        c = cv2.waitKey(1)
        if c == 27:
             break
    camera.release()
    cv2.destroyAllWindows()
        
