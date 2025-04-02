import cv2 as cv
import numpy as np
import os


os.chdir("C:/Users/serha/PycharmProjects/YOLO")


cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Farbbereich für Grün definieren
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()