import os
import cv2
import numpy as np

os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")

file = "img3.png"

img = cv2.imread("src/img/"+file)
img = cv2.resize(img, (640,640))

blur = cv2.bilateralFilter(img,9,75,75)

cv2.imshow("Original", img)
cv2.imshow("Bilateral Filtering", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()