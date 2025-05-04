import cv2
import mediapipe as mp

# Initialisieren
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Parameter
min_detection_confidence = 0.5

with mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence) as face_detection, \
        mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                              min_detection_confidence=min_detection_confidence) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Gesichtserkennung
        detection_results = face_detection.process(frame_rgb)

        frame_rgb.flags.writeable = True

        if detection_results.detections:
            for detection in detection_results.detections:
                confidence_score = detection.score[0]
                print(f"[INFO] Detected face with confidence: {confidence_score:.2f}")

                if confidence_score > 0.7:
                    # Wenn sicher genug, dann Face Mesh starten
                    mesh_results = face_mesh.process(frame_rgb)

                    if mesh_results.multi_face_landmarks:
                        for face_landmarks in mesh_results.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                face_landmarks,
                                mp_face_mesh.FACEMESH_CONTOURS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

        cv2.imshow('MediaPipe Face Mesh', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC drücken zum Beenden
            break

cap.release()
cv2.destroyAllWindows()
