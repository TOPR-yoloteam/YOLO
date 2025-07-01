from ultralytics import YOLO
import cv2
import os

# Lade ein vortrainiertes YOLO Modell für Gesichter
base_dir = os.path.dirname(__file__)
yolo_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'models', 'yolov8n-face.pt'))

model = YOLO(yolo_path)

# Öffne Webcam (0 = Standardkamera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferenz auf aktuellem Frame
    results = model(frame)

    # Bounding Boxes und Labels auf Frame zeichnen
    annotated_frame = results[0].plot()

    # Zeige das Ergebnis
    cv2.imshow('YOLOv8 Gesichtserkennung', annotated_frame)

    # Breche mit Taste 'q' ab
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera und Fenster freigeben
cap.release()
cv2.destroyAllWindows()