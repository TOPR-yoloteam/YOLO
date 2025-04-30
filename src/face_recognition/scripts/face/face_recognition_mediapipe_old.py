import cv2
import time
import statistics
import mediapipe as mp

# Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Webcam
cap = cv2.VideoCapture(0)

# FPS-Variablen
prev_frame_time = 0
curr_frame_time = 0
fps_buffer = []   # Einzelne FPS-Werte sammeln
fps_values = []   # Gespeicherte Median-FPS

# Zeitvariablen für 1-minütige Aufzeichnung
start_time = time.time()
last_record_time = start_time
total_duration = 60  # 1 Minute
record_interval = 5  # Alle 5 Sekunden speichern

print(f"FPS-Aufzeichnung über {total_duration} Sekunden gestartet. Werte werden alle {record_interval} Sekunden gespeichert.")

try:
    while cap.isOpened():
        # Aktuelle Zeit berechnen
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Nach 1 Minute beenden
        if elapsed_time >= total_duration:
            break

        success, frame = cap.read()
        if not success:
            print("Fehler beim Lesen des Frames.")
            break

        # FPS berechnen
        curr_frame_time = time.time()
        fps = 1 / (curr_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = curr_frame_time

        # FPS-Wert zum Buffer hinzufügen
        if fps > 0:
            fps_buffer.append(fps)

        # Alle 5 Sekunden Median-FPS speichern
        if current_time - last_record_time >= record_interval:
            if fps_buffer:
                median_fps = statistics.median(fps_buffer)
                fps_values.append((int(elapsed_time), round(median_fps, 2)))
                print(f"Zeit: {int(elapsed_time)}s, Median-FPS: {round(median_fps, 2)}")
                fps_buffer = []  # Buffer leeren
            last_record_time = current_time

        # Bild in RGB umwandeln
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe-Verarbeitung
        results = face_mesh.process(rgb_frame)

        # FPS-Text anzeigen
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Zeit: {int(elapsed_time)}s/{total_duration}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

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

    # FPS-Liste ausgeben
    print("\nFPS-Aufzeichnung abgeschlossen!")
    print("Zeit (s) | Median-FPS")
    print("-" * 30)
    for time_point, median_fps in fps_values:
        print(f"{time_point:8} | {median_fps}")

    # Gesamtdurchschnitt berechnen (von allen Medians)
    if fps_values:
        overall_median_fps = statistics.median([fps for _, fps in fps_values])
        print(f"\nGesamt-Median der FPS: {round(overall_median_fps, 2)}")
