import easyocr
import cv2
from ultralytics import YOLO
from multiprocessing import Process, Queue
import os
import time

# Verzeichniswechsel
os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO")

# YOLO- und EasyOCR-Modelle laden
model = YOLO("src/license_plate_recognition/models/license_plate_detector.pt")
reader = easyocr.Reader(['en'])


# Funktion für die Frame-Erfassung (Producer)
def capture_frames(frame_queue):
    cap = cv2.VideoCapture(0)  # Webcam öffnen
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            time.sleep(0.01)  # Warte kurz, um Frame-Überlauf zu vermeiden
            continue
        frame_queue.put(frame)  # Speichere Frame in der Warteschlange

    cap.release()
    frame_queue.put(None)  # Kennzeichnet das Ende der Frame-Erfassung


# Funktion für die Frame-Verarbeitung (Consumer)
def process_frames(frame_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break  # Beendet den Prozess, wenn keine Frames mehr verfügbar sind

        # YOLO-Modell ausführen
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                if cls == 0:  # Klasse "0" für Kennzeichen
                    if conf > 0.4:
                        license_plate = frame[y1:y2, x1:x2]

                        # Preprocessing
                        gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                        # OCR mit EasyOCR
                        ocr_results = reader.readtext(gray)

                        for (bbox, text, prob) in ocr_results:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            if prob > 0.9:
                                text_gesamt = " ".join([text for (_, text, _) in ocr_results])
                                print(f'Text: {text_gesamt}, Probability: {prob}')
                                prob_text = text_gesamt

                                # Text auf das Frame schreiben
                                cv2.putText(frame, prob_text, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Verarbeitetes Frame anzeigen
        cv2.imshow("Kennzeichen-Erkennung", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    frame_queue = Queue(maxsize=5)  # Warteschlange für Frames (max. 10 Frames speichern)

    # Prozesse erstellen
    capture_process = Process(target=capture_frames, args=(frame_queue,))
    process_process = Process(target=process_frames, args=(frame_queue,))

    # Prozesse starten
    capture_process.start()
    process_process.start()

    # Warten, bis beide Prozesse beendet sind
    capture_process.join()
    process_process.join()


if __name__ == "__main__":
    main()
