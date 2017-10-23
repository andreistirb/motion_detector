# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:23:43 2017

@author: Andrei
"""

import cv2
import argparse

#parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (25,25), 0)
    
    cv2.imshow("Original", frame)
    cv2.imshow("Blurred", blur)

    key = cv2.waitKey(1) & 0xFF
                     
    if key == ord("q"):
        break
                
                     
                     