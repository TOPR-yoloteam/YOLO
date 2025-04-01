import cv2 
import numpy as np
import os

os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")

file = "img3.png"

img = cv2.imread("src/img/"+file)
img = cv2.resize(img, (640,640))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
cv2.imshow("Original", img)
cv2.imshow("gray", gray)

cv2.imshow("Thresh", thresh)
cv2.imshow("Adaptive", adaptive_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
