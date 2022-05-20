import cv2
import numpy as np
import time


#Load YOLO
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg") # Original yolov3
#net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") #Tiny Yolo
classes = []
interested_classes=['car','motorbike','bus','truck']
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

print(classes)

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(classes),3))

cap=cv2.imread('traffic.jpg')
height,width,channels=cap.shape

frame=cap
frame=frame.astype('float32')
frame /=255
#detecting objects
blob = cv2.dnn.blobFromImage(frame,1,(608,608),(0,0,0),True,crop=False)
net.setInput(blob)
outs = net.forward(outputlayers)
class_ids=[]
confidences=[]
boxes=[]
for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3 and classes[class_id] in interested_classes:
                #object detected
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected

indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence= confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            cv2.putText(frame,"NOV:"+str(len(boxes)),(10,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),1)

cv2.imshow("Image",frame)
    

