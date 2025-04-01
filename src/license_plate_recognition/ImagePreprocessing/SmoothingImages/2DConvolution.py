import os
import cv2
import numpy as np

os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")

file = "img3.png"

img = cv2.imread("src/img/"+file)
img = cv2.resize(img, (640,640))

kernel = np.ones((5,5),np.float32)/20
dst = cv2.filter2D(img,-1,kernel)

cv2.imshow("Original", img)
cv2.imshow("2D Convolution", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()