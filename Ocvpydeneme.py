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
camera.exposure_mode='auto'
camera.capture('denemedamar.png')
sleep(0.042)
src=cv2.imread('denemedamar.png')
dstg=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
cv2.imwrite('test.png',dstg)
img=cv2.imread('test.png',0)
equ=cv2.equalizeHist(img)
cv2.imwrite('graywhist.png',equ)
img2=cv2.imread('graywhist.png',cv2.IMREAD_GRAYSCALE)
clahe=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
cl1=clahe.apply(img2)
cv2.imwrite('clahevein.png',cl1)
bnw=cv2.imread('clahevein.png',cv2.IMREAD_GRAYSCALE)
thresh=cv2.adaptiveThreshold(equ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imwrite('bwdenemegaussian.png',thresh)
thresh1=cv2.adaptiveThreshold(equ,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
cv2.imwrite('bwdenememean.png',thresh1)
ret,thresh2 = cv2.threshold(equ,70,255,cv2.THRESH_BINARY)
cv2.imwrite('bwdenemebinary.png',thresh2)
