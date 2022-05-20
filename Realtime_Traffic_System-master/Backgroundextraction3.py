import cv2
import numpy as np

cap=cv2.VideoCapture("night_traffic.mp4")
cap2=cv2.VideoCapture("night_traffic.mp4")

while True:
    _,frame=cap.read()
    _,frame2=cap2.read()
    
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    frame=cv2.GaussianBlur(opening,(5,5),0)
    _,frame=cv2.threshold(frame,170,255,cv2.THRESH_BINARY)
    
    cv2.imshow("frame",frame)
    cv2.imshow("frame2",frame2)

    key=cv2.waitKey(300)

    if key==27:
       break
cap2.release()
cap.release()
cv2.destroyAllWindows()
