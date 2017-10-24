# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:23:43 2017

@author: Andrei
"""

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imutils

#parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

camera = cv2.VideoCapture(args["video"])
camera.set(0 , 100000)

img_array = []
orig_array = []

while True:
    (grabbed, frame) = camera.read()
    
    if(grabbed == False):
        break
    
    frame = imutils.resize(frame, width=300)
    
    orig_array.append(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11,11), 0)
    
    img_array.append(blur)
    
    #cv2.imshow("Original", frame)
    #cv2.imshow("Blurred", blur)

   # key = cv2.waitKey(25) & 0xFF
                     
    #if key == ord("q"):
        #break
    
    
X = np.array(img_array)
O = np.array(orig_array)
print(X.size)
print(X.shape)

#3D fast fourier transform on the whole sequence of frames
q = np.fft.fftn(X)

#compute the phase angle
angle = np.arctan2(q.imag, q.real)

#compute phase spectrum array from q
phase_spectrum_array = np.exp(1j*angle)

#apply 3d inverse fast fourier transform on phase spectrum array
reconstructed_array = np.fft.ifftn(phase_spectrum_array)

#reconstruct the frames of the video
q= 0
for i in range(0,X.shape[0]):
    #smooth the frame using the averaging filter
    frame = abs(reconstructed_array[i])
    
    filteredFrame = cv2.GaussianBlur(frame, (15,15), 0)
    
    #convert the frame into binary image using mean value as threshold
    mean_value = np.mean(filteredFrame)
    ret, binary_frame = cv2.threshold(filteredFrame, 2*mean_value, 255, cv2.THRESH_BINARY) #previous alpha of mean = 1.6
    
    #perform morphological operations
    
    kernel = np.ones((5,5), np.uint8)
    
    closing = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)
    
    if q==0:
        print(frame)
        q = 1
    #cv2.imshow("abs", frame)
    
    cv2.imshow("Binary", binary_frame)
    cv2.imshow("Opening", opening)
    cv2.imshow("Blurred", X[i])
    cv2.imshow("Original", O[i])
    cv2.imshow("Closing", closing)
    
    key = cv2.waitKey(10) & 0xFF
                     
    if key == ord("q"):
        break

    
    #superimpose segmented masks on its respective frames to obtain moving objects
    
    


                     
                     