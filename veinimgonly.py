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
import RPi.GPIO as GPIO

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="no of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())
#GPIO.setmode(GPIO.BCM)
#butPin=21
#GPIO.setup(butPin,GPIO.IN,pull_up_down=GPIO.PUD_UP)
#GPIO.add_event_detect(butPin, GPIO.RISING)
# initialize the camera and stream
# allow the camera to warmup
vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()
fix=0
train=0
# capture frames from the camera
#load train images and calculate biometrics
#trainpic1=cv2.imread("trainimg1.png",0)
#trainpic2=cv2.imread("trainimg2.png",0)
#trainpic3=cv2.imread("trainimg3.png",0)
#trainpic4=cv2.imread("trainimg4.png",0)
#orb = cv2.ORB()
#kptr1, destr1 = orb.detectAndCompute(trainpic1,None)
#kptr2, destr2 = orb.detectAndCompute(trainpic2,None)
#kptr3, destr3 = orb.detectAndCompute(trainpic3,None)
#kptr4, destr4 = orb.detectAndCompute(trainpic4,None)

#print (len(kptr1),len(kptr2),len(kptr3),len(kptr4))
#bf = cv2.BFMatcher(cv2.NORM_HAMMING2,crossCheck=True)
#matches12 = bf.match(destr1,destr2)
#matches13 = bf.match(destr1,destr3)
#matches14 = bf.match(destr1,destr4)
#matches23 = bf.match(destr2,destr3)
#matches24 = bf.match(destr2,destr4)
#matches34 = bf.match(destr3,destr4)
#bf1 = cv2.BFMatcher(cv2.NORM_HAMMING)
#matchesknn12 = bf1.knnMatch(destr1,destr2,k=2)
#matchesknn13 = bf1.knnMatch(destr1,destr3,k=2)
#matchesknn14 = bf1.knnMatch(destr1,destr4,k=2)
#matchesknn23 = bf1.knnMatch(destr2,destr3,k=2)
#matchesknn24 = bf1.knnMatch(destr2,destr4,k=2)
#matchesknn34 = bf1.knnMatch(destr3,destr4,k=2)
#FLANN_INDEX_LSH = 6
#index_params= dict(algorithm = FLANN_INDEX_LSH,
#                   table_number = 4, 
#                   key_size = 9,     
#                   multi_probe_level = 1) 
#search_params = dict(checks=50)
#flann = cv2.FlannBasedMatcher(index_params,search_params)
#matchesflnn12 = flann.knnMatch(destr1,destr2,k=2)
#matchesflnn13 = flann.knnMatch(destr1,destr3,k=2)
#matchesflnn14 = flann.knnMatch(destr1,destr4,k=2)
#matchesflnn23 = flann.knnMatch(destr2,destr3,k=2)
#matchesflnn24 = flann.knnMatch(destr2,destr4,k=2)
#matchesflnn34 = flann.knnMatch(destr3,destr4,k=2)
#
#matches12 = sorted(matches12, key = lambda x:x.distance)
#
#goodflann=[]

#for m,n in matchesflnn34:
#    if m.distance < 0.75*n.distance:
#        goodflann.append(m)

#goodmatches=[]


#for j in matches34:
#    if j.distance < 64:
#    	goodmatches.append(j)

#goodknnmatches=[]

#for l,o in matchesknn34:
#    if l.distance < 0.75*o.distance:
#	goodknnmatches.append([l])

#print ("no of good bf matches", len(goodmatches),"no of good knn matches", len(goodknnmatches), "no of good flann matches", len(goodflann)) 

#good13=[]
#for m,n in matches13:
#	if m.distance < 0.75*n.distance:
#		good13.append([m])
#print (len(good13))
#del good13[:]

while 1:
    startt=time.time()
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    frame = vs.read()
    gray_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eqhi=cv2.equalizeHist(gray_image)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(21,21))
    cl1 = clahe.apply(eqhi)
    eqhia=cv2.equalizeHist(cl1)    
    blur = cv2.GaussianBlur(eqhia,(5,5),sigmaX=5)
    blur2 = cv2.GaussianBlur(eqhia,(15,15),sigmaX=25)
    blurinv = cv2.bitwise_not(blur)
    subtracted= cv2.add(blur2,blurinv)
    #subtracted=cv2.absdiff(blur3,blur)
    #subtracted2=cv2.absdiff(blur2,blur4)
    eqhsub= cv2.equalizeHist(subtracted)
    cl2=clahe.apply(eqhsub)
    blur3=cv2.medianBlur(cl2,3)
    eqhsub2=cv2.equalizeHist(blur3)
    ret,th1 = cv2.threshold(eqhsub2,145,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.dilate(th1, kernel, iterations=1)
    img_dilate = cv2.erode(img_erosion, kernel, iterations=1)
    #kpin,destrin = orb.detectAndCompute(cl1,None)
	
    #if GPIO.event_detected(butPin):
	#inimage=cl1
	#if fix==0:
		#if train==0:
		#	cv2.imwrite("trainimg1.png",inimage)
		#elif train==1:
		#	cv2.imwrite("trainimg2.png",inimage)
		#elif train==2:
		#	cv2.imwrite("trainimg3.png",inimage)
		#elif train==3:
		#	cv2.imwrite("trainimg4.png",inimage)
		#else:
		#	train=2
	#else :
	#	fix=-1
	#	train+=1
	#fix+=1
		
	#if train==0:
	#        cv2.imwrite("trainimg1.png",inimage)
	#elif train==1:
	#	cv2.imwrite("trainimg2.png",inimage)
	#elif train==2:
	#	cv2.imwrite("trainimg3.png",inimage)
	#else :
	#	cv2.imwrite("trainimg4.png",inimage)
	
	#match1=bf.match(destrin,destr1)
	#match2=bf.match(destrin,destr2)
  	#match3=bf.match(destrin,destr3)
	#match4=bf.match(destrin,destr4)
	#matchlist=[]
	#matchlist.append(match1)
	#matchlist.append(match2)
	#matchlist.append(match3)
	#matchlist.append(match4)

	#matchknn1 = bf1.knnMatch(destrin,destr1,k=2)
	#matchknn2 = bf1.knnMatch(destrin,destr2,k=2)
	#matchknn3 = bf1.knnMatch(destrin,destr3,k=2)
	#matchknn4 = bf1.knnMatch(destrin,destr4,k=2)
	#matchlistknn=[]
	#matchlistknn.append(matchknn1)
        #matchlistknn.append(matchknn2)
        #matchlistknn.append(matchknn3)
        #matchlistknn.append(matchknn4)

	#matchflnn1 = flann.knnMatch(destr1,destr2,k=2)
	#matchflnn2 = flann.knnMatch(destr1,destr3,k=2)
	#matchflnn3 = flann.knnMatch(destr1,destr4,k=2)
	#matchflnn4 = flann.knnMatch(destr2,destr3,k=2)
	#matchlistflnn=[]
	#matchlistflnn.append(matchflnn1)
        #matchlistflnn.append(matchflnn2)
        #matchlistflnn.append(matchflnn3)
        #matchlistflnn.append(matchflnn4)

	#goodinputm=[]
	#goodinputmknn=[]
	#goodinputmflnn=[]
	#for a in matchlist:
	#	for m in a:
	#		if m.distance < 64:
	#			goodinputm.append(m)
	#for b in matchlistknn:
	#	for n,p in b:
	#		if n.distance < 0.75*p.distance:
	#			goodinputmknn.append([n])		
	#for c in matchlistflnn:
	#	for t,r in c:
	#		if t.distance < 0.75*r.distance:
	#			goodinputmflnn.append([t])
	#print (len(goodinputm)/4,len(goodinputmknn)/4,len(goodinputmflnn)/4)
	#train+=1
	#del match1[:]
        #del match2[:]
	#del match3[:]
    	#del match4[:]
       #img_erosion1 = cv2.erode(img_dilate, kernel, iterations=1)
    #img_erosion = cv2.erode(th1, kernel, iterations=1)
	
    #eqhsub=cv2.equalizeHist(subtracted)
    # ret2,th2 = cv2.threshold(eqhsub,126,255,cv2.THRESH_BINARY_INV)
    # show the frame
    #outimg = cv2.drawKeypoints(cl1, kpin, None, color=(0,255,0), flags=0)
    cv2.imshow("Frame",img_dilate)
    cv2.moveWindow("Frame",0,0)
    cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN,cv2.cv.CV_WINDOW_FULLSCREEN)
    key = cv2.waitKey(1) & 0xFF
    stopp=time.time()
    secpframe=stopp-startt
    framepsec=1/secpframe
    print (framepsec)
    # clear the stream in preparation for the next frame    
    # if the `q` key was pressed, break from the loop
    

