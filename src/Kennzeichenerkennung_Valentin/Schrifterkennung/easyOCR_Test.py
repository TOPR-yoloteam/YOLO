import easyocr
import cv2

reader = easyocr.Reader(['en']) # specify the language
result = reader.readtext('src/Kennzeichenerkennung_Valentin/Schrifterkennung/img/img.png')

for (bbox, text, prob) in result:
    print(f'Text: {text}, Probability: {prob}')