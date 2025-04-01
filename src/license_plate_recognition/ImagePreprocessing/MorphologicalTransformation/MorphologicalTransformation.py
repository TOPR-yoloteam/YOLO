import os
import cv2
import numpy as np
import easyocr

os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")

file = "img_1.png"

img = cv2.imread("src/img/"+file)
img = cv2.resize(img, (640,640))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(gray,kernel,iterations = 1)
dilation = cv2.dilate(gray,kernel,iterations = 1)
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


cv2.imshow("Original", img)
cv2.imshow("Gray", gray)
#cv2.imshow("Erosion", erosion)
cv2.imshow("Dilation", dilation)
cv2.imshow("Opening", opening)
cv2.imshow("Closing", closing)
#cv2.imshow("Gradient", gradient)
#cv2.imshow("Tophat", tophat)
#cv2.imshow("Blackhat", blackhat)


images = {
    "Original": img,
    "Gray": gray,
    "Dilation": dilation, #beste1
    "Opening": opening, #beste2
    "Closing": closing,

}




read = easyocr.Reader(['en'])

for name, img in images.items():
    result = read.readtext(img)

    for (bbox, text, prob) in result:
        print(f'Image: {name} Text: {text}, Probability: {prob} \n\n')


cv2.waitKey(0)
cv2.destroyAllWindows()