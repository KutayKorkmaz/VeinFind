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
while 1:
    startt=time.time()
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    frame = vs.read()
    gray_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray_image)
    blur = cv2.GaussianBlur(cl1,(91,91),sigmaX=12)
    subtracted=cl1-blur
    ret,th1 = cv2.threshold(subtracted,126,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    img_erosion = cv2.dilate(th1, kernel, iterations=2)
    eqhsub=cv2.equalizeHist(subtracted)
    # show the frame
    cv2.imshow("Frame",th1)
    key = cv2.waitKey(1) & 0xFF
    stopp=time.time()
    print (stopp - startt)
    # clear the stream in preparation for the next frame    
    # if the `q` key was pressed, break from the loop
    
