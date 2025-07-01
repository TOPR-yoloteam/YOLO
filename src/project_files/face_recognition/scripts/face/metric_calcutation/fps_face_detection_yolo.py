from ultralytics import YOLO
import cv2
import os
import time
import statistics
from collections import deque
import numpy as np

# Lade ein vortrainiertes YOLO Modell für Gesichter
base_dir = os.path.dirname(__file__)
yolo_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'models', 'yolov8n-face.pt'))

model = YOLO(yolo_path)

# Webcam
cap = cv2.VideoCapture(0)

# Optional: Auflösung reduzieren für bessere Performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Performance-Variablen
frame_times = deque(maxlen=30)
process_times = deque(maxlen=30)
faces_detected = deque(maxlen=30)
fps_values = []
instant_fps_values = deque(maxlen=30)

# Zeitvariablen für 1-minütige Aufzeichnung
start_time = time.time()
last_record_time = start_time
total_duration = 70  # 1 Minute
record_interval = 5  # Alle 5 Sekunden speichern

print(
    f"Performance-Aufzeichnung über {total_duration} Sekunden gestartet. Werte werden alle {record_interval} Sekunden gespeichert.")

try:
    while cap.isOpened():
        current_time = time.time()
        elapsed_time = current_time - start_time
        frame_start_time = time.time()

        if elapsed_time >= total_duration:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Zeit zwischen Frames für FPS-Berechnung
        if frame_times:
            frame_time = frame_start_time - frame_times[-1]
            if frame_time > 0:
                frame_times.append(frame_start_time)
                instant_fps = 1.0 / frame_time
                instant_fps_values.append(instant_fps)
        else:
            frame_times.append(frame_start_time)
            frame_time = 0

        # YOLO Inferenz mit Zeitmessung
        process_start = time.time()
        results = model(frame)
        process_end = time.time()
        process_time = process_end - process_start
        process_times.append(process_time)

        # Anzahl erkannter Gesichter zählen
        face_count = len(results[0].boxes)
        faces_detected.append(face_count)

        # FPS berechnen
        if len(frame_times) >= 2:
            avg_frame_time = (frame_times[-1] - frame_times[0]) / (len(frame_times) - 1)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            fps = 0

        # Stability Score berechnen
        avg_process_time = sum(process_times) / len(process_times) if process_times else 0
        avg_faces = sum(faces_detected) / len(faces_detected) if faces_detected else 0

        stability_score = 100
        face_stability = 0
        fps_stability = 0

        if len(faces_detected) > 1:
            try:
                face_stability = statistics.stdev(faces_detected)
            except statistics.StatisticsError:
                face_stability = 0

        if len(instant_fps_values) > 1:
            try:
                fps_stability = statistics.stdev(instant_fps_values)
            except statistics.StatisticsError:
                fps_stability = 0

        if fps > 0:
            stability_score = round(100 - min(100, (10 * fps_stability / max(fps, 1)) + (20 * face_stability)))
            stability_score = max(0, min(100, stability_score))

        # Performance-Metriken speichern
        if current_time - last_record_time >= record_interval:
            performance_data = {
                "time": int(elapsed_time),
                "fps": round(fps, 2),
                "avg_process_time": round(avg_process_time * 1000, 2),
                "avg_faces": round(avg_faces, 2),
                "stability": stability_score
            }
            fps_values.append(performance_data)
            print(f"Zeit: {performance_data['time']}s, FPS: {performance_data['fps']}, "
                  f"Verarbeitungszeit: {performance_data['avg_process_time']}ms, "
                  f"Erkannte Gesichter: {performance_data['avg_faces']}, "
                  f"Stabilität: {performance_data['stability']}%")
            last_record_time = current_time

        # Performance-Infos anzeigen
        annotated_frame = results[0].plot()

        cv2.putText(annotated_frame, f"FPS: {round(fps, 1)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Verarbeitung: {round(avg_process_time * 1000, 1)}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Gesichter: {face_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Stabilität: {stability_score}%", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Zeit: {int(elapsed_time)}s/{total_duration}s", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('YOLOv8 Gesichtserkennung', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Programm manuell beendet.")
finally:
    cap.release()
    cv2.destroyAllWindows()

    # Performance-Daten ausgeben
    print("\nPerformance-Aufzeichnung abgeschlossen!")
    print("Zeit (s) | FPS | Verarbeitung (ms) | Gesichter | Stabilität (%)")
    print("-" * 65)
    for data in fps_values:
        print(
            f"{data['time']:8} | {data['fps']:>4} | {data['avg_process_time']:>15} | {data['avg_faces']:>9} | {data['stability']:>13}")

    # Durchschnittswerte berechnen
    if fps_values:
        avg_fps = statistics.mean([data['fps'] for data in fps_values])
        avg_process = statistics.mean([data['avg_process_time'] for data in fps_values])
        avg_stability = statistics.mean([data['stability'] for data in fps_values])
        print(f"\nDurchschnittliche Werte:")
        print(f"FPS: {round(avg_fps, 2)}")
        print(f"Verarbeitungszeit: {round(avg_process, 2)}ms")
        print(f"Stabilität: {round(avg_stability, 2)}%")