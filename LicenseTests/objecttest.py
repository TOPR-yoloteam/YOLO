import cv2
from ultralytics import YOLO

# Modell laden
model = YOLO("yolov8n.pt")

# Bild einlesen
image_path = "test3.jpg"
image = cv2.imread(image_path)

# Objekterkennung ausführen
results = model(image)

# Ergebnisse auf das Bild zeichnen
annotated_image = results[0].plot()

# Bild anzeigen (bleibt offen, bis eine Taste gedrückt wird)
cv2.imshow("YOLOv8 Detection", annotated_image)
cv2.waitKey(0)  # Warten auf eine beliebige Taste
cv2.destroyAllWindows()  # Fenster schließen

