import cv2
from skimage import data,img_as_float
from skimage import exposure
import sys
import os
import picamera
import numpy as np
from time import sleep
camera=picamera.PiCamera()
camera.resolution = (1920,1080)
camera.contrast=100
camera.sharpness=100
camera.exposure_mode='auto'
camera.capture('denemedamar.png')
sleep(0.042)
src=cv2.imread('denemedamar.png')
sleep(0.042)
dstg=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
sleep(0.042)
cv2.imwrite('test.png',dstg)
sleep(0.042)
img=cv2.imread('test.png',0)
p2 = np.percentile(img, 2)
p98 = np.percentile(img, 98)
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
sleep(0.042)
equ=cv2.equalizeHist(img_rescale)
sleep(0.042)
cv2.imwrite('graywhist.png',equ)
sleep(0.042)
img2=cv2.imread('graywhist.png',cv2.IMREAD_GRAYSCALE)
sleep(0.042)
clahe=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
sleep(0.042)
cl1=clahe.apply(img2)
sleep(0.042)
cv2.imwrite('clahevein.png',cl1)
sleep(0.042)
bnw=cv2.imread('clahevein.png',cv2.IMREAD_GRAYSCALE)
sleep(0.042)
thresh=cv2.adaptiveThreshold(equ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
sleep(0.042)
cv2.imwrite('bwdenemegaussian.png',thresh)
sleep(0.042)
thresh1=cv2.adaptiveThreshold(equ,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
sleep(0.042)
cv2.imwrite('bwdenememean.png',thresh1)
sleep(0.042)
ret,thresh2 = cv2.threshold(equ,70,255,cv2.THRESH_BINARY)
sleep(0.042)
cv2.imwrite('bwdenemebinary.png',thresh2)
