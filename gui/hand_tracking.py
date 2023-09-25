import mediapipe as mp
import cv2 
import time

def recognize_hands(frame):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands = 2,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    results = hands.process(frame)
        # if hands are detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                # if id == 0:
                cv2.circle(frame, (cx,cy), 3, (255,0,0), cv2.FILLED)
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    
    

    cv2.putText(frame, "Hand Tracking On (two hands)", (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    return frame