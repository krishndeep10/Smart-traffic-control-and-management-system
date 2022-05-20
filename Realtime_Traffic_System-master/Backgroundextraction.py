import cv2
import numpy as np

cap=cv2.VideoCapture("night_traffic.mp4")

subtractor=cv2.createBackgroundSubtractorMOG2(varThreshold=50)

while True:
    _,frame=cap.read()

    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    frame=subtractor.apply(frame)
    cv2.imshow("frame",frame)

    key=cv2.waitKey(15)

    if key==27:
        break
cap.release()
cv2.destroyAllWindows()
