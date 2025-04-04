#https://ai.google.dev/edge/mediapipe/solutions/guide
#https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

#install mediapipe with
#pip install mediapipe opencv-python

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import text
from mediapipe.tasks.python import audio
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # refine_landmarks enables iris tracking
mp_drawing = mp.solutions.drawing_utils

# Landmark indices for left and right eyes (based on Mediapipe documentation)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Landmark indices for left and right eyebrows
LEFT_EYEBROW_IDX = [70, 63, 105, 66, 107]       # Outer left eyebrow arch
RIGHT_EYEBROW_IDX = [336, 296, 334, 293, 300]   # Outer right eyebrow arch

def draw_feature(frame, landmark_list, indices, width, height, color):
    points = []
    for idx in indices:
        point = landmark_list.landmark[idx]
        x, y = int(point.x * width), int(point.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 2, color, -1)
    if len(points) > 1:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], isClosed=False, color=color, thickness=1)

# Open webcam
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
            # Draw features
            draw_feature(frame, face_landmarks, LEFT_EYE_IDX, width, height, (0, 255, 0))       # Green
            draw_feature(frame, face_landmarks, RIGHT_EYE_IDX, width, height, (255, 0, 0))      # Blue
            draw_feature(frame, face_landmarks, LEFT_EYEBROW_IDX, width, height, (0, 255, 255)) # Yellow
            draw_feature(frame, face_landmarks, RIGHT_EYEBROW_IDX, width, height, (255, 0, 255))# Magenta

    cv2.imshow("Biometric Face Recognition", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()