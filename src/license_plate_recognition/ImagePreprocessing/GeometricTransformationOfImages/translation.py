import cv2
import os
import numpy as np

os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")
img = cv2.imread("src/img/img6.jpg")
rows, cols, ch = img.shape

M = np.float32([[1, 0, 100], [0, 1, 100]])
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('img', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()