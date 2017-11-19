# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:23:43 2017

@author: Andrei
"""

import cv2
import argparse
import numpy as np
import imutils
import pickle
import os

#parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

camera = cv2.VideoCapture(args["video"])
camera.set(0, 150000)

img_array = []
orig_array = []

#take each frame of the video and build two arrays,
#one with original frames and the other with blurred
#gray frames

while True:
    (grabbed, frame) = camera.read()
    
    if(grabbed == False):
        break
    
    frame = imutils.resize(frame, width=300)
    
    orig_array.append(frame)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11,11), 0)
    img_array.append(blur)
    
X = np.array(img_array)

#we need this only for displaying the result
O = np.array(orig_array)

#3D fast fourier transform on the whole sequence of frames
q = np.fft.fftn(X)

#compute the phase angle
angle = np.arctan2(q.imag, q.real)

#compute phase spectrum array from q
phase_spectrum_array = np.exp(1j*angle)

# Serialize the fourier object
if (os.access('stream_fft.pickle', os.R_OK)):
    pickle_in = open('stream_fft.pickle', 'rb')
    reconstructed_array = pickle.load(pickle_in)
    pickle_in.close()
else:
    #apply 3d inverse fast fourier transform on phase spectrum array
    reconstructed_array = np.fft.ifftn(phase_spectrum_array)
    with open('stream_fft.pickle', 'wb') as f:
        pickle.dump(reconstructed_array, f)

#reconstruct the frames of the video
q= 0
for i in range(0,O.shape[0]):
    #smooth the frame using the averaging filter
    frame = abs(reconstructed_array[i])
    
    filteredFrame = cv2.GaussianBlur(frame, (5,5), 0)
    
    #convert the frame into binary image using mean value as threshold
    mean_value = np.mean(filteredFrame)
    
    #median_value = np.median(filteredFrame)
    ret, binary_frame = cv2.threshold(filteredFrame, 1.8*mean_value, 255, cv2.THRESH_BINARY) #previous alpha of mean = 1.6
    
    #denoise the binary_frame
    npbinary = np.array(binary_frame, dtype = np.uint8)
    denoised = cv2.fastNlMeansDenoising(src=npbinary, h=200, templateWindowSize=7, searchWindowSize=21)
    
    #perform morphological operations
    kernel = np.ones((13,13), np.uint8)
    
    closing = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)
                
    (_, cnts, _) = cv2.findContours(denoised, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

	 #loop over the contours
    for c in cnts:
		 #if the contour is too small, ignore it
        if cv2.contourArea(c) < 500:
            continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(O[i], (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(denoised, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
    cv2.imshow("Denoised", denoised)
    cv2.imshow("Binary", binary_frame)
    cv2.imshow("Opening", opening)
    cv2.imshow("Original", O[i])
    cv2.imshow("Closing", closing)
    
    key = cv2.waitKey(30) & 0xFF
                     
    if key == ord("q"):
        break

    #superimpose segmented masks on its respective frames to obtain moving objects - search the link
    
cv2.destroyAllWindows()


                     
                     