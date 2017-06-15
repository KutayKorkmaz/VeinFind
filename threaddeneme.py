from __future__ import print_function
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import imutils
import argparse
import time
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="no of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

# initialize the camera and stream
# allow the camera to warmup
vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()
# capture frames from the camera
a=0
while 1:
    startt=time.time()
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    frame = vs.read()
    gray_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if a<2:
	picname="kutayel"+ str(a)  +".png"
        cv2.imwrite(picname,gray_image)
	a+=1
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray_image)    
    blur = cv2.GaussianBlur(cl1,(11,11),sigmaX=20)
    subtracted=cl1-blur
    #sub1=cl1-gray_image
    #eqhsub1= cv2.equalizeHist(sub1)
    #ret3,th3= cv2.threshold(eqhsub1,127,255,cv2.THRESH_BINARY)
    #blur1= cv2.GaussianBlur(th3,(3,3),sigmaX=1)
    #eqhblur1=cv2.equalizeHist(blur1)
    #ret4,th4=cv2.threshold(eqhblur1,127,255,cv2.THRESH_BINARY)
    ret,th1 = cv2.threshold(subtracted,126,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.dilate(th1, kernel, iterations=1)
    img_dilate = cv2.erode(img_erosion, kernel, iterations=1)
    img_erosion1 = cv2.erode(img_dilate, kernel, iterations=1)
    img_erosion = cv2.erode(th1, kernel, iterations=1)

    #eqhsub=cv2.equalizeHist(subtracted)
    # ret2,th2 = cv2.threshold(eqhsub,126,255,cv2.THRESH_BINARY_INV)
    # show the frame
    cv2.imshow("Frame",img_dilate)
    cv2.moveWindow("Frame",0,0)
    cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN,cv2.cv.CV_WINDOW_FULLSCREEN)
    key = cv2.waitKey(1) & 0xFF
    stopp=time.time()
    print (stopp - startt)
    # clear the stream in preparation for the next frame    
    # if the `q` key was pressed, break from the loop
    
