import os
import cv2
import numpy as np

os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")

file = "img3.png"

img = cv2.imread("src/img/"+file)
img = cv2.resize(img, (640,640))

blur = cv2.GaussianBlur(img,(5,5),0)

cv2.imshow("Original", img)
cv2.imshow("Gaussian Blur", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()