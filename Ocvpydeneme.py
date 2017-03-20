import cv2
import sys
import os
import picamera
from time import sleep
camera=picamera.PiCamera()
camera.resolution = (1920, 1080)
camera.contrast=100
camera.capture('denemedamar.jpg')
img=cv2.imread('denemedamar.jpg')
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.imwrite("graydenemedamar.png",img)
equ=cv2.equalizeHist(img)
cv2.imwrite('graywhist.png',equ)
clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
cl1=clahe.apply(img)
cv2.imwrite('clahevein.png',cl1)
