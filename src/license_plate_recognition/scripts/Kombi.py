from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/test/license_plate_detector.pt")


image = cv2.imread("img/auto.jpeg")


results = model(image)


for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding Box-Koordinaten
        conf = box.conf[0].item()  # Konfidenz
        cls = int(box.cls[0].item())  # Klasse


        if cls == 0:  # "0" Klasse für Kennzeichen 
            license_plate = image[y1:y2, x1:x2]

            # Preprocessing für Tesseract
            gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # OCR mit Tesseract
            text = pytesseract.image_to_string(gray, config="--psm 7")

            print(f"Erkanntes Kennzeichen: {text.strip()}")

            # Optional: Gefundenes Kennzeichen anzeigen
            cv2.imshow("Kennzeichen", gray)
            cv2.waitKey(0)