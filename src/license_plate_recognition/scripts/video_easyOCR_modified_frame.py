import re
import easyocr
import cv2
from ultralytics import YOLO
import os

def filter_uppercase_and_numbers(input_string):
    result = re.sub(r"[^A-Z0-9\s]", "", input_string)
    return result

os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")
print(os.getcwd())

#lade Model
model = YOLO("src/Kennzeichenerkennung_Sero/test/license_plate_detector.pt")
#Kamera init
cap = cv2.VideoCapture(0)
#easyOCR reader mit englischer sprache/Buchstaben
reader = easyocr.Reader(['en'])

while cap.isOpened():
    ret, frame = cap.read()
    temp = frame
    if not ret:
        break
    #benutze Modell um Kennzeichen zu erkennen
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if cls == 0:  # Klasse "0" Kennzeichen
                if conf > 0.4:
                    #frame für die ausgabe anpassen
                    license_plate = frame[y1-50:y2+50, x1-50:x2+50]
                    #Image Preprocessing
                    temp = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                    temp = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                    #erkenne Text in Licence Plate frame
                    result = reader.readtext(temp)

                    for (bbox, text, prob) in result:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if prob > 0.9:
                            text_gesamt = " ".join([text for (_, text, _) in result])
                            print(f'Text: {text_gesamt}, Probability: {prob}')
                            prob_text = filter_uppercase_and_numbers(text_gesamt)

                            cv2.putText(frame, prob_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Kennzeichen-Erkennung", temp)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


