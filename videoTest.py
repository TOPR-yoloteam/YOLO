import cv2
from ultralytics import YOLO

model1 = YOLO("yolov8n.pt") #-> object detection
model2 = YOLO("license_plate_detector.pt") #https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8 → auch nicht so der Brüller
model3 = YOLO("LP-detection.pt") #https://huggingface.co/MKgoud/License-Plate-Recognizer -> schlecht

video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Fehler beim Öffnen des Videos!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Verkleinere das Frame auf eine kleinere Größe
    width = int(frame.shape[1] * 0.3)
    height = int(frame.shape[0] * 0.3)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim)


 #   results1 = model1(resized_frame)
  #  annotated_frame = results1[0].plot()
    # Führe die Objekterkennung auf dem Frame durch
    results2 = model2(resized_frame)

    # Ergebnisse auf das Frame zeichnen
    annotated_frame = results2[0].plot(annotated_frame)

    # Zeige das annotierte Frame an
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Wenn die 'q'-Taste gedrückt wird, beende die Schleife
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Schließe das Video und alle Fenster
cap.release()
cv2.destroyAllWindows()
