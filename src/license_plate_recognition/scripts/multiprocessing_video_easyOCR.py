import cv2
from ultralytics import YOLO
import os
import re
import time
import threading
import easyocr
from queue import Queue

# Konfigurationsparameter für Performance-Optimierung
RESIZE_FACTOR = 0.5  # Faktor für Frame-Verkleinerung
PROCESS_EVERY_N_FRAMES = 5  # Nur jeden N-ten Frame vollständig verarbeiten
MAX_QUEUE_SIZE = 5  # Maximale Anzahl von Frames in der Queue

# Globale Variablen
#os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")
os.chdir("/home/talmva/workspace/YOLO/")
#os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")
model = YOLO("src/license_plate_recognition/models/license_plate_detector_ncnn_model")
easyocr_reader = easyocr.Reader(['en'])

frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
processed_frame = None
running = True
frame_count = 0


def filter_uppercase_and_numbers(input_string):
    result = re.sub(r"[^A-Z0-9\s]", "", input_string)
    return result


def capture_video():
    """Thread-Funktion zum Erfassen von Video-Frames"""
    global running
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            running = False
            break

        # Überspringen, wenn Queue voll
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            time.sleep(0.01)  # Kurze Pause, um CPU zu entlasten

    cap.release()


def process_frames():
    """Thread-Funktion zur Verarbeitung von Frames"""
    global processed_frame, frame_count, running

    while running:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        frame_count +=1

        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

        full_processing = (frame_count % PROCESS_EVERY_N_FRAMES == 0)

        if full_processing:
            results = model(small_frame)

            output_frame = frame.copy()

            for result in results:
                for box in result.boxes:
                    if box.cls[0].item() == 0 and box.conf[0].item() > 0.4:
                        x1, y1, x2, y2 = map(int, box.xyxy[0] / RESIZE_FACTOR)
                        license_plate = frame[y1:y2, x1:x2]

                        gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                        result = easyocr_reader.readtext(gray)

                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        for (bbox, text, prob) in result:
                            if prob > 0.9:
                                if prob > 0.9:
                                    text_gesamt = " ".join([text for (_, text, _) in result])
                                    prob_text = filter_uppercase_and_numbers(text_gesamt)
                                    cv2.putText(output_frame, prob_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            processed_frame = output_frame
        else:
            if processed_frame is None:
                processed_frame = frame

capture_thread = threading.Thread(target=capture_video)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()

# Hauptschleife für die Anzeige
while running:
    if processed_frame is not None:
        cv2.imshow('Object Detection', processed_frame)

    # Beenden mit 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# Aufräumen
capture_thread.join()
process_thread.join()
cv2.destroyAllWindows()