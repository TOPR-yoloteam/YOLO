import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #erkennt farben in bestimmten ranges
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    final_mask = mask_red + mask_blue

    res = cv2.bitwise_and(frame,frame, mask= final_mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',final_mask)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()