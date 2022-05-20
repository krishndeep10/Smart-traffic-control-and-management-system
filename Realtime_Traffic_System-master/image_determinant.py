import numpy as np
import cv2

img=cv2.imread("traffic.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.resize(img,(10,10))
numpy=np.array(img)
print(numpy)
cv2.imshow('j',img)
print(numpy.shape)
numpy=numpy.astype('float32')
numpy/=140
det=np.linalg.det(numpy)
print(det)

