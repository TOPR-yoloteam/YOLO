# Translation is the shifting of an object's location.

import numpy
import cv2
import os

os.chdir("C:/Users/serha/PycharmProjects/YOLO/")

img = cv2.imread('ImagePreprocessing/Images/plates_tur.jpg', cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
rows,cols = img.shape

M = numpy.float32([[1,0,50],[0,1,50]])
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()