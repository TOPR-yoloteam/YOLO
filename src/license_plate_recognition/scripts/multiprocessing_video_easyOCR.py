import easyocr
import cv2
from ultralytics import YOLO
import os
import re
import time
import threading
from queue import Queue

# Konfigurationsparameter
RESIZE_FACTOR = 0.5  # Faktor für Frame-Verkleinerung
PROCESS_EVERY_N_FRAMES = 5  # Nur jeden N-ten Frame vollständig verarbeiten
MAX_QUEUE_SIZE = 5  # Maximale Größe der Frame-Queue

# Globale Variablen
#os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")
os.chdir("/home/talmva/workspace/YOLO/")
model = YOLO("src/license_plate_recognition/models/license_plate_detector_ncnn_model")
reader = easyocr.Reader(['en'])

frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
output_queue = Queue(maxsize=MAX_QUEUE_SIZE)
processed_frame = None
running = True
frame_count = 0


def filter_uppercase_and_numbers(input_string):
    """Filtert nur Großbuchstaben und Zahlen"""
    result = re.sub(r"[^A-Z0-9\s]", "", input_string)
    return result


def capture_video():
    """Thread-Funktion zum Erfassen von Frames von der Kamera"""
    global running

    cap = cv2.VideoCapture(0)  # 0 für die Standard-Webcam

    while running:
        ret, frame = cap.read()
        if not ret:
            running = False
            break

        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            time.sleep(0.01)  # CPU-Entlastung bei voller Queue

    cap.release()


def process_frames():
    """Thread-Funktion zur Verarbeitung von Frames"""
    global processed_frame, frame_count, running

    while running:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        frame_count += 1

        # Frame verkleinern für schnellere Verarbeitung
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

        # Analyse nur bei jedem N-ten Frame
        full_processing = (frame_count % PROCESS_EVERY_N_FRAMES == 0)

        if full_processing:
            # YOLO-Verarbeitung
            results = model(small_frame)
            output_frame = frame.copy()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0] / RESIZE_FACTOR)
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    if cls == 0 and conf > 0.4:  # Kennzeichen
                        license_plate = frame[y1:y2, x1:x2]

                        # Preprocessing
                        gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                        # OCR mit EasyOCR
                        result = reader.readtext(gray)

                        for (_, text, prob) in result:
                            if prob > 0.9:
                                text_gesamt = " ".join([text for (_, text, _) in result])
                                prob_text = filter_uppercase_and_numbers(text_gesamt)

                                # Zeichnen auf das Frame
                                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(output_frame, prob_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                            (0, 255, 0), 2)

            processed_frame = output_frame

        # Zwischenspeichern des Frames
        if not output_queue.full():
            output_queue.put(processed_frame)


def display_frames():
    """Thread-Funktion zur Anzeige der Frames"""
    global running

    while running:
        if not output_queue.empty():
            frame = output_queue.get()
            cv2.imshow("Kennzeichen-Erkennung", frame)

        # Beenden mit 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    cv2.destroyAllWindows()


# Threads initialisieren
capture_thread = threading.Thread(target=capture_video)
process_thread = threading.Thread(target=process_frames)
display_thread = threading.Thread(target=display_frames)

# Threads starten
capture_thread.start()
process_thread.start()
display_thread.start()

# Warten auf Abschluss der Threads
capture_thread.join()
process_thread.join()
display_thread.join()
