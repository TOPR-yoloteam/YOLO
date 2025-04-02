import numpy as np
import cv2
import os

print(os.getcwd())
os.chdir("C:/Users/serha/PycharmProjects/YOLO/")
print(os.getcwd())
img = cv2.imread("ImagePreprocessing/Images/licanceplateger.jpg")


res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
cv2.imwrite("ImagePreprocessing/GeometricTransformation/ScaledImage/test.png",res)