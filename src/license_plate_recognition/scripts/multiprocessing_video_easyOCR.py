import multiprocessing
import cv2
from ultralytics import YOLO  # YOLO-Modell importieren (stellen Sie sicher, dass ultralytics installiert ist)
import os

def read_video(frame_queue):
    """Liest Frames von der Kamera und fügt sie in die Queue ein."""
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Fügt jedes gelesene Frame in die Queue ein
        if not frame_queue.full():
            frame_queue.put(frame)

        cv2.imshow("Original Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_with_yolo(frame_queue):
    """Liest Frames aus der Queue, verarbeitet sie mit YOLO und zeigt die Ergebnisse."""
    model = YOLO("src/license_plate_recognition/models/license_plate_detector.pt")  # YOLOv8-Modell laden (geben Sie den korrekten Pfad zur PT-Datei ein)

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Verarbeite das Frame mit YOLO
            results = model(frame)

            # Darstellen der Ergebnisse von YOLO auf dem Frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    # Nur Ergebnisse mit höherer Konfidenz zeichnen
                    if conf > 0.4:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Class {cls} ({conf:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Zeige die Ergebnisse im Fenster an
            cv2.imshow("YOLO Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    print(os.getcwd())
    #os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")
    os.chdir("/home/talmva/workspace/YOLO")
    # Queue erstellen, damit Prozesse Daten austauschen können
    frame_queue = multiprocessing.Queue(maxsize=10)

    # Prozesse definieren
    video_process = multiprocessing.Process(target=read_video, args=(frame_queue,))
    yolo_process = multiprocessing.Process(target=process_with_yolo, args=(frame_queue,))

    # Prozesse starten
    video_process.start()
    yolo_process.start()

    # Auf Beendigung der Prozesse warten
    video_process.join()
    yolo_process.join()
