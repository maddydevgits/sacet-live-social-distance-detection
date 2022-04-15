import cv2
import numpy as np
import imutils
import time
import os
import math

from constants import *
from itertools import chain

# get label details from default path
LABELS=open(YOLOV4_LABELS_PATH).read().strip().split('\n')

# generate random colors for different classes
COLORS=np.random.randint(0,255,size=(len(LABELS),3))

# load the neural net from the trained weights
neural_net=cv2.dnn.readNetFromDarknet(YOLOV4_CFG_PATH,YOLOV4_WEIGHTS_PATH)

# extract the layer names
layer_names=neural_net.getLayerNames()

# to store the layer names in a variable
layer_names=[layer_names[i-1] for i in neural_net.getUnconnectedOutLayers()]

# start the camera
cam=cv2.VideoCapture(0)

# store output in output folder
writer=None

# fix the width and height of frame
(W,H) = (None,None)

# start the infinite loop
while True:
    (res,frame)=cam.read() # frame will be read from the camera

    if not res: # try to check whether cam is working or not
        print('cam is not working')
        break

    # cam is working now
    if W is None or H is None:
        H,W=(frame.shape[0],frame.shape[1]) # loading the height and width of frame
    
    # pre-process the frame
    # 1/255.0 -> normalising the pixel values (0 to 1)
    # 255 - 0xFF - 1 Byte
    # 1 - 1 Bit 
    # fixed input - (416, 416)
    blob=cv2.dnn.blobFromImage(frame, 1/255.0, (416,416),swapRB=True,crop=False)

    # pass this blob to neural net
    neural_net.setInput(blob)

    # start the timer
    start_time=time.time()

    # load the layer outputs
    layer_outputs=neural_net.forward(layer_names)

    # end the timer
    end_time=time.time()

    # create some boxes to represent the objects
    boxes = []

    # when object is scanned, it will be having some confidence
    confidences=[]

    # every class will be having some class id
    classIDs=[]

    # identify the distance between the person
    lines=[]

    # identify the center of the box
    box_centers=[]

    # iterate the layer outputs
    for output in layer_outputs:
        # identify the detection in the output
        for detection in output:
            # to identify the scores of the object
            scores=detection[5:]
            # maximum confidence
            classID=np.argmax(scores)
            # store the confidence
            confidence=scores[classID]
            # take decision
            if confidence>0.5 and classID==0: # (>50% confidence and person)
                box=detection[0:4]*np.array([W,H,W,H]) # extracting the rectangle dimension
                (centerX,centerY,width,height)=box.astype('int') # calculating the center
                # calculated the x, y coordinates of the center
                x=int(centerX-(width/2))
                y=int(centerY-(height/2))
                # store the box centers
                box_centers=[centerX,centerY]
                # store x-cord, y-cord
                boxes.append([x,y,int(width),int(height)])
                # store confidences
                confidences.append(float(confidence))
                # store class ids
                classIDs.append(classID)
    
    # check how many boxes are there
    idxs= cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.3)

    # if no-person is available, len(idxs)=0
    # if 1 person is available, len(idxs)=1
    # if 2 persons are available, len(idxs)=2

    if(len(idxs)>0): # if persons are available
        unsafe=[]
        count=0

        for i in idxs.flatten():
            
            # load the x-cord, y-cord
            (x,y)=(boxes[i][0],boxes[i][1])

            # load width, height
            (w,h)=(boxes[i][2],boxes[i][3])

            # calculate the centers
            centeriX=boxes[i][0] + (boxes[i][2]//2)
            centeriY=boxes[i][1] + (boxes[i][3]//2)

            # extract some color
            color=[int(c) for c in COLORS[classIDs[i]]]

            # display text
            text = '{}: {}'.format(LABELS[classIDs[i]],confidences[i])

            # copy the idxs
            idxs_copy=list(idxs.flatten())
            idxs_copy.remove(i)

            # trying to calculate the second person coordinates

            for j in np.array(idxs_copy):
                centerjX=boxes[j][0] + (boxes[j][2]//2)
                centerjY=boxes[j][1] + (boxes[j][3]//2)

                distance=math.sqrt(math.pow(centerjX-centeriX,2) + math.pow(centerjY-centeriY),2)
                print(distance)

                if distance<=SAFE_DISTANCE:
                    cv2.line(frame,(boxes[i][0] + (boxes[i][2]//2),boxes[i][1] + (boxes[i][3]//2)),(boxes[j][0]+(boxes[j][2]//2),boxes[j][1]+(boxes[j][3]//2)),(0,0,255),2)
                    unsafe.append([centerjX,centerjY])
                    unsafe.append([centeriX,centeriY])
            
            # to check whether this person is in unsafe or not
            if centeriX in chain(*unsafe) and centeriY in chain(*unsafe):
                count+=1
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            
            cv2.putText(frame,text,(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.5,color,2)
            cv2.rectangle(frame,(50,50),(450,90),(0,0,0),-1)
            cv2.putText(frame,'No of people unsafe: {}'.format(count),(70,70),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),3)
    
    if writer is None:
        fourcc=cv2.VideoWriter_fourcc(*'MJPG')
        writer=cv2.VideoWriter(OUTPUT_PATH,fourcc,30,(frame.shape[1],frame.shape[0]),True)
    
    cv2.imshow('output',frame)
    writer.write(frame)
    cv2.waitKey(1)

print('Cleaning Up')
writer.release()
cam.release()