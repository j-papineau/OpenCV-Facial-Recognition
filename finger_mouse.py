import cv2
import mediapipe as mp
import pyautogui

camera = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands()
drawingTools = mp.solutions.drawing_utils

screenWidth, screenHeight = pyautogui.size()

while True:
    ret, frame = camera.read()
    if ret:
        frame = cv2.flip(frame, 1)
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
        output = hands.process(frame)
        result = output.multi_hand_landmarks
        
        frameHeight, frameWidth, _ = frame.shape

        if result:
            for hand in result:
                drawingTools.draw_landmarks(frame, hand)
                landmarks = hand.landmark
                for id, landmark in enumerate(landmarks):
                    if id == 8:
                        # print(landmark)
                        
                        x = int(landmark.x * frameWidth)
                        y = int(landmark.y * frameHeight)
                        cv2.circle(img=frame, center=(x,y), radius=30, color=(0,255,255))
                        mousePositionX = screenWidth/frameWidth*x
                        mousePositionY = screenHeight/frameHeight*y
                        # pyautogui.moveTo(mousePositionX, mousePositionY)

    

    cv2.imshow("video", frame)
    c = cv2.waitKey(1)
    if c == 27:
             break
camera.release()
cv2.destroyAllWindows()