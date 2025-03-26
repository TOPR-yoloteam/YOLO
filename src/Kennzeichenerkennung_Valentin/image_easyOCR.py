import easyocr
import cv2
import os

os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")

image = cv2.imread('C:/Users/Valentin.Talmon/PycharmProjects/YOLO/src/Kennzeichenerkennung_Valentin/Schrifterkennung/img/img.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imshow('thresh', thresh)


reader = easyocr.Reader(['en']) # specify the language
result = reader.readtext('C:/Users/Valentin.Talmon/PycharmProjects/YOLO/src/Kennzeichenerkennung_Valentin/Schrifterkennung/img/licence_plate.png')


for (bbox, text, prob) in result:
    print(f'Text: {text}, Probability: {prob}')