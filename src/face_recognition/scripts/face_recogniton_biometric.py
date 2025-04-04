#https://ai.google.dev/edge/mediapipe/solutions/guide
#https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

#install mediapipe with
#pip install mediapipe opencv-python

import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # refine_landmarks enables iris tracking
mp_drawing = mp.solutions.drawing_utils

# Landmark indices for left and right eyes (based on Mediapipe documentation)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


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

            # Collect and draw left eye landmarks
            left_eye_points = []
            for idx in LEFT_EYE_IDX:
                point = face_landmarks.landmark[idx]
                x, y = int(point.x * width), int(point.y * height)
                left_eye_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Draw lines between left eye landmarks (closed polygon)
            cv2.polylines(frame, [np.array(left_eye_points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)

            # Collect and draw right eye landmarks
            right_eye_points = []
            for idx in RIGHT_EYE_IDX:
                point = face_landmarks.landmark[idx]
                x, y = int(point.x * width), int(point.y * height)
                right_eye_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # Draw lines between right eye landmarks (closed polygon)
            cv2.polylines(frame, [np.array(right_eye_points, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=1)

    # Show the result
    cv2.imshow("Eye Detection with Landmarks", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()