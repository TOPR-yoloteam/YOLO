from ultralytics import YOLO
import cv2
import pytesseract

# YOLOv8-Modell laden
model = YOLO("../models/license_plate_detector.pt")

cap = cv2.VideoCapture(0)  # 0 für die Standard-Webcam


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

                # OCR mit Tesseract
                    # OCR mit Tesseract verbessern
                    text = pytesseract.image_to_string(gray,config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789").strip()
                    print(f"Erkanntes Kennzeichen: {text}")

                # Kennzeichen-Box zeichnen
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Zeige das Video mit erkannten Kennzeichen
    cv2.imshow("Kennzeichen-Erkennung", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()