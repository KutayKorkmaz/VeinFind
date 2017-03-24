import cv2
import sys
import os
import picamera
from time import sleep
camera=picamera.PiCamera()
camera.resolution = (768,576)
camera.contrast=100
camera.sharpness=100
camera.capture('denemedamar.png')
sleep(0.042)
src=cv2.imread('denemedamar.png')
tmp=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
_,alpha=cv2.threshold(tmp,50,255,cv2.THRESH_BINARY)
b,g,r=cv2.split(src)
rgba=[b,g,r,alpha]
dst=cv2.merge(rgba,4)
dstg=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
cv2.imwrite('test.png',dst)
img=cv2.imread('test.png')
imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite('testgray.png',imgray)
equ=cv2.equalizeHist(dstg)
cv2.imwrite('graywhist.png',equ)
clahe=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
cl1=clahe.apply(dstg)
cv2.imwrite('clahevein.png',dstg)
bnw=cv2.imread('clahevein.png',cv2.IMREAD_GRAYSCALE)
(thresh,imbw)=cv2.threshold(bnw,45,255,cv2.THRESH_BINARY_INV)
cv2.imwrite('bwdeneme.png',imbw)
