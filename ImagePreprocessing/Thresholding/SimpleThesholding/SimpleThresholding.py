# For every pixel, the same threshold value is applied.
# If the pixel value is smaller than or equal to the threshold,
# it is set to 0, otherwise it is set to a maximum value.

import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


os.chdir("C:/Users/serha/PycharmProjects/YOLO/")

img = cv.imread('ImagePreprocessing/Images/licenceplate_Louisiana.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

cv.imwrite("ImagePreprocessing/Thresholding/SimpleThesholding/Example/thresh1_binary.jpg", thresh1)
cv.imwrite("ImagePreprocessing/Thresholding/SimpleThesholding/Example/thresh2_binary_inv.jpg", thresh2)
cv.imwrite("ImagePreprocessing/Thresholding/SimpleThesholding/Example/thresh3_trunc.jpg", thresh3)
cv.imwrite("ImagePreprocessing/Thresholding/SimpleThesholding/Example/thresh4_tozero.jpg", thresh4)
cv.imwrite("ImagePreprocessing/Thresholding/SimpleThesholding/Example/thresh5_tozero_inv.jpg", thresh5)

plt.savefig("ImagePreprocessing/Thresholding/SimpleThesholding/Example/plots.png")
plt.show()
