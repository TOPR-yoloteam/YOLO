import numpy as np
import easyocr
import cv2
import os


os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")


image = cv2.imread("src/img/img.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imshow("thresh", thresh)




cv2.waitKey(0)