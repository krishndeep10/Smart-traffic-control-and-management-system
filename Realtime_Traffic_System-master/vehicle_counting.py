import cv2
import numpy as np
import time


#Load YOLO
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg") 
#net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") #Tiny Yolo
classes = []
interested_classes=['car','motorbike','bus','truck']
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

print(classes)

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(classes),3))

#loading image
cap=cv2.VideoCapture("TrafficVideo.mp4") 
font = cv2.FONT_HERSHEY_PLAIN
starting_time= time.time()
frame_id = 0

while True:
    _,frame= cap.read() 
    frame_id+=1
    
    height,width,channels = frame.shape

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
                
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                

                boxes.append([x,y,w,h]) 
                confidences.append(float(confidence)) 
                class_ids.append(class_id) 
    
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

    car=0
    motorbike=0
    truck=0
    bus=0
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            if(label=="car"):
                car=car+1
            if(label=="bus"):
                bus+=1
            if(label=="truck"):
                truck+=1
            if(label=="motorbike"):
                motorbike+=1
            confidence= confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
            

    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2))+" "+"NOV:"+str(car+bus+motorbike+truck),(10,60),font,2,(0,0,0),1)
    cv2.putText(frame,interested_classes[0]+" "+str(car),(10,90),font,2,(0,0,1),1)
    cv2.putText(frame,interested_classes[1]+" "+str(motorbike),(10,120),font,2,(0,0,1),1)
    cv2.putText(frame,interested_classes[2]+" "+str(bus),(10,150),font,2,(0,0,1),1)
    cv2.putText(frame,interested_classes[3]+" "+str(truck),(10,180),font,2,(0,0,1),1)
    print("car: "+str(car)+" "+"truck: "+str(truck)+" "+"bus: "+str(bus) +" "+"motorbike: "+str(motorbike))
    
    
    cv2.imshow("Image",frame)
    key = cv2.waitKey(1) 
    
    if key == 27: 
        break;
    
cap.release()    
cv2.destroyAllWindows()
