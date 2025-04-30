import cv2
import time
import statistics
import mediapipe as mp
import numpy as np
from collections import deque

# Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Webcam
cap = cv2.VideoCapture(0)

# Verbesserte FPS- und Performance-Variablen
frame_times = deque(maxlen=30)  # Speichert die letzten 30 Frame-Zeiten für stabilere FPS
process_times = deque(maxlen=30)  # Speichert die Verarbeitungszeiten der Gesichtserkennung
faces_detected = deque(maxlen=30)  # Anzahl der erkannten Gesichter je Frame
fps_values = []  # Gespeicherte Performance-Metriken
instant_fps_values = deque(maxlen=30)  # Speichert die aktuellen FPS-Werte für Stabilitätsberechnung

# Zeitvariablen für 1-minütige Aufzeichnung
start_time = time.time()
last_record_time = start_time
total_duration = 60  # 1 Minute
record_interval = 5  # Alle 5 Sekunden speichern

print(f"Performance-Aufzeichnung über {total_duration} Sekunden gestartet. Werte werden alle {record_interval} Sekunden gespeichert.")

try:
    while cap.isOpened():
        # Aktuelle Zeit berechnen
        current_time = time.time()
        elapsed_time = current_time - start_time
        frame_start_time = time.time()

        # Nach 1 Minute beenden
        if elapsed_time >= total_duration:
            break

        success, frame = cap.read()
        if not success:
            print("Fehler beim Lesen des Frames.")
            break

        # Zeit zwischen Frames für FPS-Berechnung
        if frame_times:
            frame_time = frame_start_time - frame_times[-1]
            if frame_time > 0:  # Verhindert Division durch Null
                frame_times.append(frame_start_time)
                # Momentane FPS für diesen Frame berechnen und speichern
                instant_fps = 1.0 / frame_time
                instant_fps_values.append(instant_fps)
        else:
            frame_times.append(frame_start_time)
            frame_time = 0

        # Bild in RGB umwandeln
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe-Verarbeitung mit Zeitmessung
        process_start = time.time()
        results = face_mesh.process(rgb_frame)
        process_end = time.time()
        process_time = process_end - process_start
        process_times.append(process_time)

        # Anzahl erkannter Gesichter zählen
        face_count = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
        faces_detected.append(face_count)

        # FPS berechnen (gleitender Durchschnitt)
        if len(frame_times) >= 2:
            # Durchschnittliche Zeit zwischen den letzten Frames berechnen
            avg_frame_time = (frame_times[-1] - frame_times[0]) / (len(frame_times) - 1)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            fps = 0

        # Stability Score berechnen (Kombination aus FPS-Stabilität und Gesichtserkennungsrate)
        avg_process_time = sum(process_times) / len(process_times) if process_times else 0
        avg_faces = sum(faces_detected) / len(faces_detected) if faces_detected else 0
        
        # Berechnung der Stabilitätswerte mit sicheren Prüfungen
        stability_score = 100
        face_stability = 0
        fps_stability = 0

        # Gesichtsstabilität: Nur berechnen, wenn genug Daten vorhanden sind
        if len(faces_detected) > 1:
            try:
                face_stability = statistics.stdev(faces_detected)
            except statistics.StatisticsError:
                face_stability = 0
        
        # FPS-Stabilität: Nur berechnen, wenn genug Daten vorhanden sind
        if len(instant_fps_values) > 1:
            try:
                fps_stability = statistics.stdev(instant_fps_values)
            except statistics.StatisticsError:
                fps_stability = 0
        
        # Stabilitätsscore berechnen (höherer Wert = bessere Stabilität)
        if fps > 0:
            stability_score = round(100 - min(100, (10 * fps_stability / max(fps, 1)) + (20 * face_stability)))
            # Sicherstellen, dass der Wert zwischen 0 und 100 bleibt
            stability_score = max(0, min(100, stability_score))

        # Alle 5 Sekunden Performance-Metriken speichern
        if current_time - last_record_time >= record_interval:
            performance_data = {
                "time": int(elapsed_time),
                "fps": round(fps, 2),
                "avg_process_time": round(avg_process_time * 1000, 2),  # In ms
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
        cv2.putText(frame, f"FPS: {round(fps, 1)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Verarbeitung: {round(avg_process_time*1000, 1)}ms", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Gesichter: {face_count}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Stabilität: {stability_score}%", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Zeit: {int(elapsed_time)}s/{total_duration}s", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Gesichter zeichnen
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

        # Bild anzeigen
        cv2.imshow('MediaPipe Face Mesh', frame)

        # Beenden mit 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Programm manuell beendet.")
finally:
    # Ressourcen freigeben
    cap.release()
    cv2.destroyAllWindows()

    # Performance-Daten ausgeben
    print("\nPerformance-Aufzeichnung abgeschlossen!")
    print("Zeit (s) | FPS | Verarbeitung (ms) | Gesichter | Stabilität (%)")
    print("-" * 65)
    for data in fps_values:
        print(f"{data['time']:8} | {data['fps']:>4} | {data['avg_process_time']:>15} | {data['avg_faces']:>9} | {data['stability']:>13}")

    # Durchschnittswerte berechnen
    if fps_values:
        avg_fps = statistics.mean([data['fps'] for data in fps_values])
        avg_process = statistics.mean([data['avg_process_time'] for data in fps_values])
        avg_stability = statistics.mean([data['stability'] for data in fps_values])
        print(f"\nDurchschnittliche Werte:")
        print(f"FPS: {round(avg_fps, 2)}")
        print(f"Verarbeitungszeit: {round(avg_process, 2)}ms")
        print(f"Stabilität: {round(avg_stability, 2)}%")