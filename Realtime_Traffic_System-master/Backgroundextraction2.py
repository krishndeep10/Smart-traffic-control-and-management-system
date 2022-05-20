import cv2
import numpy as np

cap=cv2.VideoCapture("night_traffic.mp4")

while True:
    _,frame=cap.read()
    
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame=cv2.GaussianBlur(frame,(5,5),0)
    _,frame=cv2.threshold(frame,210,255,cv2.THRESH_BINARY)
    
    cv2.imshow("frame",frame)

    key=cv2.waitKey(50)

    if key==27:
       break
cap.release()
cv2.destroyAllWindows()
