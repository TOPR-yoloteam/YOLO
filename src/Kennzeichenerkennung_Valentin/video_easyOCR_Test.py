import re

import easyocr
import cv2
from ultralytics import YOLO
import os


def filter_uppercase_and_numbers(input_string):
    # Verwende einen regulären Ausdruck, um nur Großbuchstaben (A-Z) und Zahlen (0-9) zuzulassen
    result = re.sub(r"[^A-Z0-9\s]", "", input_string)
    return result




os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO")
print(os.getcwd())

model = YOLO("src/Kennzeichenerkennung_Sero/test/license_plate_detector.pt")

cap = cv2.VideoCapture(0)  # 0 für die Standard-Webcam
cap.set(cv2.CAP_PROP_FPS, 5)

reader = easyocr.Reader(['en'])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Ende des Videos


    results = model(frame)

    for result in results:
        #score = result
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if cls == 0:  # Klasse "0" Kennzeichen
                if conf > 0.4:
                    license_plate = frame[y1-50:y2+50, x1-50:x2+50]
                # Preprocessing
                    gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                # OCR mit Tesseract
                    # OCR mit Tesseract verbessern
                    result = reader.readtext(gray)

                    for (bbox, text, prob) in result:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if prob > 0.9:
                            text_gesamt = " ".join([text for (_, text, _) in result])

                            print(f'Text: {text_gesamt}, Probability: {prob}')
                            prob_text = filter_uppercase_and_numbers(text_gesamt)
                        # Kennzeichen-Box zeichnen

                            cv2.putText(frame, prob_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Zeige das Video mit erkannten Kennzeichen
    cv2.imshow("Kennzeichen-Erkennung", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


##load image via openCV
#image = cv2.imread('C:/Users/Valentin.Talmon/PycharmProjects/YOLO/src/Kennzeichenerkennung_Valentin/Schrifterkennung/img/img3.png')
##grayfilter on image
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('gray.png', gray)
#
#
##gray tones with openCV
#thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#cv2.imwrite('thresh.png', thresh)
#
##OCR reader
# # specify the language

