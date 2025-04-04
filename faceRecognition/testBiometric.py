#https://ai.google.dev/edge/mediapipe/solutions/guide
#https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

import cv2
import mediapipe as mp

#install mediapipe with
#pip install mediapipe opencv-python


# prepare Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # refine = Iris detection aktivieren!
mp_drawing = mp.solutions.drawing_utils

# Augen-Landmark-Indices (basierend auf Mediapipe-Doku)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]     # außen rund um linkes Auge
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]   # außen rund um rechtes Auge

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            # points on left eye
            for idx in LEFT_EYE_IDX:
                point = face_landmarks.landmark[idx]
                x, y = int(point.x * width), int(point.y * height)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # points on right eye
            for idx in RIGHT_EYE_IDX:
                point = face_landmarks.landmark[idx]
                x, y = int(point.x * width), int(point.y * height)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("Test", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to close
        break

cap.release()
cv2.destroyAllWindows()
