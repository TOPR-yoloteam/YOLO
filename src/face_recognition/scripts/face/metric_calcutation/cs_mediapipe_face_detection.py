import cv2
import mediapipe as mp
import time
import numpy as np

# Initialisieren
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Parameter
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

# Tracking variables
start_time = time.time()
frame_count = 0
detection_scores = []
mesh_scores = []

with mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence) as face_detection, \
        mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        ) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Gesichtserkennung
        detection_results = face_detection.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if detection_results.detections:
            for detection in detection_results.detections:
                detection_confidence = detection.score[0]
                detection_scores.append(detection_confidence)
                print(f"[INFO] Face Detection confidence: {detection_confidence:.2f}")

                if detection_confidence > 0.7:
                    mesh_results = face_mesh.process(frame_rgb)

                    if mesh_results.multi_face_landmarks:
                        for idx, face_landmarks in enumerate(mesh_results.multi_face_landmarks):
                            # Berechne Mesh Confidence basierend auf Landmark-Stabilität
                            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                            # Berechne durchschnittliche z-Koordinate als Approximation für Konfidenz
                            mesh_confidence = 1.0 - np.mean(np.abs(landmarks[:, 2]))
                            mesh_scores.append(mesh_confidence)
                            print(f"[INFO] Face Mesh confidence: {mesh_confidence:.2f}")
                            
                            mp_drawing.draw_landmarks(
                                frame,
                                face_landmarks,
                                mp_face_mesh.FACEMESH_CONTOURS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))
                            
                            # Display confidence scores on frame
                            cv2.putText(frame, f"Detection: {detection_confidence:.2f}", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, f"Mesh: {mesh_confidence:.2f}", (10, 70),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Face Mesh', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC drücken zum Beenden
            break

end_time = time.time()
runtime = end_time - start_time
fps = frame_count / runtime

# Calculate statistics
print("\n=== Session Statistics ===")
print(f"Runtime: {runtime:.2f} seconds")
print(f"Total Frames: {frame_count}")
print(f"Average FPS: {fps:.2f}")

if detection_scores:
    print("\nFace Detection Scores:")
    print(f"Average: {np.mean(detection_scores):.3f}")
    print(f"Median: {np.median(detection_scores):.3f}")
    print(f"Min: {np.min(detection_scores):.3f}")
    print(f"Max: {np.max(detection_scores):.3f}")

if mesh_scores:
    print("\nFace Mesh Scores:")
    print(f"Average: {np.mean(mesh_scores):.3f}")
    print(f"Median: {np.median(mesh_scores):.3f}")
    print(f"Min: {np.min(mesh_scores):.3f}")
    print(f"Max: {np.max(mesh_scores):.3f}")

cap.release()
cv2.destroyAllWindows()
