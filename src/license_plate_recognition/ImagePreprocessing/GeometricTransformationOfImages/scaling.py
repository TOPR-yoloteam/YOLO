import numpy as np
import cv2
import os

os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")


img = cv2.imread("src/img/img6.jpg")
res = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imwrite("src/img/img6_scaled.jpg",res)