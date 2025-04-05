import os
import cv2


os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")

image = cv2.imread("src/license_plate_recognition/data/img/img.png")

cv2.imwrite("src/license_plate_recognition/data/img/ohne.png", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("src/license_plate_recognition/data/img/gray.png", gray)

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imwrite("src/license_plate_recognition/data/img/thresh.png", thresh)

