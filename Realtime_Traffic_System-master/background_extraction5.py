import cv2
import numpy as np

cap=cv2.VideoCapture("night_traffic.mp4")


_,frame2=cap.read()
cv2.imshow("color",frame2)

#applying filter to image
frame2=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(frame2, cv2.MORPH_OPEN, kernel)
frame2=cv2.GaussianBlur(opening,(5,5),0)
_,frame2=cv2.threshold(frame2,170,255,cv2.THRESH_BINARY)
#converting image to array of 336,596
fm=np.array(frame2)
x=fm.reshape(-1)
white_pixels=0
black_pixels=0

for i in x:
        if i>=170:
            white_pixels+=1
        else:
            black_pixels+=1
print(fm.shape)
print("Number of white pixels in image= ",white_pixels)
print("Number of black pixels in image= ",black_pixels)
print("Area of white= ",(white_pixels/black_pixels))
font = cv2.FONT_HERSHEY_PLAIN


while True:
    _,frame=cap.read()
    _,frame2=cap.read()
    fp=np.array(frame)
    
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    frame=cv2.GaussianBlur(opening,(5,5),0)
    _,frame=cv2.threshold(frame,170,255,cv2.THRESH_BINARY)

    x=fp.reshape(-1)
    white_pixels=0
    black_pixels=0
    for i in x:
        if i>=170:
            white_pixels+=1
        else:
            black_pixels+=1
            
    cv2.putText(frame,"Density of white pixels = "+str(white_pixels/black_pixels),(0,15),font,1,(255,255,255),1)
    
    cv2.imshow("frame",frame)
    cv2.imshow("frame2",frame2)

    key=cv2.waitKey(1)

    if key==27:
       break
cap.release()
cv2.destroyAllWindows()
