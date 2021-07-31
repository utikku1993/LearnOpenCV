import numpy as np
import cv2

# Load video from file
cap = cv2.VideoCapture('./data/Chantaje.mp4')

# Load video from webcam
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Can not open video stream')

while True:
    ret, frame = cap.read()

    if ret:
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) and 0xFF == 27:
            break
    else:
        break