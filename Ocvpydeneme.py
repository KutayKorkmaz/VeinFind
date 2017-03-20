import cv2
import sys
import os
import picamera
from time import sleep
camera=picamera.PiCamera()
camera.contrast=40
camera.capture('denemedamar.jpg')
img=cv2.imread('denemedamar.jpg')
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.imwrite("graydenemedamar.png",img)
equ=cv2.equalizeHist(img)
cv2.imwrite('graywhist.png',equ)

