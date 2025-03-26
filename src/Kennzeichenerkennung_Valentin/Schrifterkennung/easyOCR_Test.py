import easyocr
import cv2
from ultralytics import YOLO
import os

os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")
print(os.getcwd())

model = YOLO("src/Kennzeichenerkennung_Sero/test/license_plate_detector.pt")

cap = cv2.VideoCapture(0)  # 0 für die Standard-Webcam

reader = easyocr.Reader(['en'])

# Initialize frame counter
frame_counter = 0
frame_interval = 30  # Process every 30th frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Ende des Videos

    # Increment frame counter
    frame_counter += 1

    # Only process every 30th frame
    if frame_counter % frame_interval == 0:
        results = model(frame)

        for result in results:
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

                        # OCR mit EasyOCR
                        result = reader.readtext(gray)

                        for (bbox, text, prob) in result:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            if prob > 0.9:
                                text_gesamt = " ".join([text for (_, text, _) in result])
                                print(f'Text: {text_gesamt}, Probability: {prob}')
                                prob_text = text_gesamt
                                # Kennzeichen-Box zeichnen
                                cv2.putText(frame, prob_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                                            2)

    # Always show the video feed, even when not processing
    cv2.imshow("Kennzeichen-Erkennung", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()