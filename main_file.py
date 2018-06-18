import keyboard
import cv2
import numpy as np
import math
import os
import time
import pyautogui
from point import p_crop
from hand import h_crop
from fin import f_crop
from fist import fs_crop
from thumbdown import t_crop
from okay import ok_crop

os.chdir("location of scripts,xml files")
h1_cascade=cv2.CascadeClassifier('hand.xml')
okay_cascade = cv2.CascadeClassifier('ok.xml')
point_cascade = cv2.CascadeClassifier('point1.xml')
fin_cascade=cv2.CascadeClassifier('fin_2.xml')
fist_cascade=cv2.CascadeClassifier('fist.xml')
thumbdown_cascade = cv2.CascadeClassifier('thumbdown.xml')
cap = cv2.VideoCapture(0)
ca=0

while(1):    
        _,img=cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        point=point_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,flags=0, minSize=(100,80))
        fin=fin_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,flags=0, minSize=(100,80))
        hand=h1_cascade.detectMultiScale(gray,1.1, 5)
        fist=fist_cascade.detectMultiScale(gray,1.3, 5)
        okay=okay_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,flags=0, minSize=(100,150))
        thumbdown=thumbdown_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,flags=0, minSize=(100,80))
        for (x,y,w,h) in okay:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y:y+h,x:x+w]
                crop_img=img[y:y+h,x:x+w]
                ok_crop(crop_img,img)
                #time.sleep(0.5)
                
                
        for (x,y,w,h) in point:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y:y+h,x:x+w]
                crop_img=img[y:y+h,x:x+w]
                ha=2
                p_crop(crop_img,img)
                #time.sleep(0.5)

        for (x,y,w,h) in fin:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y:y+h,x:x+w]
                crop_img=img[y:y+h,x:x+w]
                f_crop(crop_img,img)
                #time.sleep(0.5)


        for (x,y,w,h) in thumbdown:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y:y+h,x:x+w]
                crop_img=img[y:y+h,x:x+w]
                ha=5
                t_crop(crop_img,img)
                #time.sleep(0.5)

        for (x,y,w,h) in hand:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y:y+h,x:x+w]
                crop_img=img[y:y+h,x:x+w]
                h_crop(crop_img,img)


                
        cv2.imshow('Feed',img)
        
        k=cv2.waitKey(10)
        if k==27:
                break
cv2.destroyAllWindows()

    
