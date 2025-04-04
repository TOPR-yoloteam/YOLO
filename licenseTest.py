import cv2
from ultralytics import YOLO

# Modelle laden
model1 = YOLO("license_plate_detector.pt")
model2 = YOLO("yolov8n.pt")

# Klassenlisten abrufen
class_names1 = model1.names
class_names2 = model2.names

# Bild laden
image_path = "test_license1.jpg"
image = cv2.imread(image_path)
video = cv2.VideoCapture("https://www.youtube.com/watch?v=EUUT1CW_9cg")

# Erkennung durchführen
results1 = model1(video)
results2 = model2(video)

# Ergebnisse kombinieren
detections = []
if results1[0].boxes is not None:
    for box in results1[0].boxes.data.tolist():
        detections.append((box, class_names1))  # Speichert Box + passende Klassenliste
if results2[0].boxes is not None:
    for box in results2[0].boxes.data.tolist():
        detections.append((box, class_names2))  # Speichert Box + passende Klassenliste

# Bounding-Boxen auf das Bild zeichnen
for detection, class_names in detections:
    x1, y1, x2, y2, score, class_id = detection
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Integer-Werte für OpenCV

    # Klassenname abrufen (falls vorhanden, sonst "Unbekannt")
    class_label = class_names.get(int(class_id), "Unbekannt")

    # Rechteck zeichnen
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Beschriftung hinzufügen (Klassenname + Score)
    label = f"{class_label} ({score:.2f})"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Bild anzeigen
cv2.imshow("YOLOv8 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
