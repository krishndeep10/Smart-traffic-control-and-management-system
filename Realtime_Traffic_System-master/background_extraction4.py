import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("night_traffic.jpeg")
img2=img
frame=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
frame=cv2.GaussianBlur(opening,(5,5),0)
_,frame=cv2.threshold(frame,140,255,cv2.THRESH_BINARY)
cv2.imshow("i",frame)
cv2.imshow("i2",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
