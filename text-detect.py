
#cmd to run in cwd->
#py text-detect.py -i C:\python\OCR\OCR\images\testdata\example_01.jpg -east C:/python/OCR/opencv-text-recognition/frozen_east_text_detection.pb

from imutils.object_detection import non_max_suppression
import numpy as np
import argparse 
import time
import cv2

ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',type=str,
                help='path to input image')
ap.add_argument('-east','--east',type=str,
                help='path to input EAST TEST DETECTOR')
ap.add_argument('-c','--min-confidence',type=float,default=0.5,
                help='minimum probabilty required to inspect a region')
ap.add_argument('-w','--width',type=int,default=320,
                help='resized image width(should be multiple of 32)')
ap.add_argument('-e','--height',type=int,default=320,
                help='resized image height(should be multiple of 32)')
args=vars(ap.parse_args())

img=cv2.imread(args['image'])
orig=img.copy()
(H,W)=img.shape[:2]
(newW,newH)=(args['width'],args['height'])
rH=H/float(newH)
rW=W/float(newW)
#rH is the ratio of height of image to heigth in cmd line args

img=cv2.resize(img,(newW,newH))
(H,W)=img.shape[:2]
# resize the image and grab the new image dimensions


# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text

layerNames=[
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
        ]
#The first layer is our output sigmoid activation which gives us the probability 
#of a region containing text or not.
#The second layer is the output feature map that represents the “geometry” of the 
#image — we’ll be able to use this geometry to derive the bounding box coordinates
#of the text in the input image

print('[INFO]East text detector loading')

net=cv2.dnn.readNet(args['east'])
#dnn is deep neural network model
#readNet is read network

blob=cv2.dnn.blobFromImage(img,1.0,(W,H),(123.68,116.78,103.94),swapRB=True,crop=False)

start=time.time()
net.setInput(blob)
(scores,geometry)=net.forward(layerNames)
end=time.time()

print("[INFO] text detection took{:.6f} seconds",format(end-start))
#print(scores)

#To predict text we can simply set the blob  as input and call net.forward
(numRows,numCols)=scores.shape[2:4]
#cv2.imshow('img',scores)
#print(geometry)
#print(numRows,numCols)
rects=[]
confidences=[]

#By supplying layerNames  as a parameter to net.forward , we are instructing
#OpenCV to return the two feature maps that we are interested in:

#The output geometry  map used to derive the bounding box coordinates of text in our input images
#And similarly, the scores  map, containing the probability of a given region containing text

for y in range(0,numRows):
    scoresData=scores[0,0,y]
    xData0=geometry[0,0,y]
    xData1=geometry[0,1,y]
    xData2=geometry[0,2,y]
    xData3=geometry[0,3,y]
    anglesData=geometry[0,4,y]

    for x in range(0,numCols):
        if scoresData[x]<args["min_confidence"]:
            continue

        (offsetX,offsetY)=(x * 4.0,y * 4.0)

        angle=anglesData[x]
        cos=np.cos(angle)
        sin=np.sin(angle)
        h=xData0[x]+xData2[x]
        w=xData1[x]+xData3[x]
        
        endX=int(offsetX+(cos*xData1[x])+(sin*xData2[x]))
        endY=int(offsetY-(sin*xData1[x])+(cos*xData2[x]))
        startX=int(endX-w)
        startY=int(endY-h)
        
        

        rects.append((startX,startY,endX,endY))
        confidences.append(scoresData[x])

boxes=non_max_suppression(np.array(rects),probs=confidences)

for(startX,startY,endX,endY) in boxes:
    startX=int(startX*rW)
    startY=int(startY*rH)
    endX=int(endX*rW)
    endY=int(endY*rH)

    cv2.rectangle(orig,(startX,startY),(endX,endY),(0,255,255),2)
cv2.imshow("text Detection",orig)
cv2.waitKey(0)
    






