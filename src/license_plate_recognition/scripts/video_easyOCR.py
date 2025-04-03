import easyocr
import cv2
from ultralytics import YOLO
import os
import re


def filter_uppercase_and_numbers(input_string):
    result = re.sub(r"[^A-Z0-9\s]", "", input_string)
    return result

#os.chdir("C:/Users/serha/PycharmProjects/YOLO/")
os.chdir("/home/talmva/workspace/YOLO/")
#os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")

#lade Model
model = YOLO("src/license_plate_recognition/models/license_plate_detector_ncnn_model")
cap = cv2.VideoCapture(0)  # 0 für die Standard-Webcam
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
                    license_plate = frame[y1:y2, x1:x2]

                # Preprocessing
                    gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                    # OCR mit Easyocr
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
