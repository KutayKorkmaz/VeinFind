import cv2
import sys
import os
import picamera
import numpy as np
from time import sleep
camera=picamera.PiCamera()
camera.resolution = (768,576)
camera.contrast=100
camera.sharpness=100
camera.capture('denemedamar.png')
sleep(0.042)
src=cv2.imread('denemedamar.png')
dstg=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
cv2.imwrite('test.png',dstg)
img=cv2.imread('test.png')
equ=cv2.calcHist([img],[0],None,[256],[30,256])
cv2.imwrite('graywhist.png',equ)
img2=cv2.imread('graywhist.png',cv2.IMREAD_GRAYSCALE)
clahe=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
cl1=clahe.apply(img2)
cv2.imwrite('clahevein.png',cl1)
bnw=cv2.imread('clahevein.png',cv2.IMREAD_GRAYSCALE)
thresh=cv2.adaptiveThreshold(bnw,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imwrite('bwdeneme.png',thresh)
