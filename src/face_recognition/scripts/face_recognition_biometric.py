#https://ai.google.dev/edge/mediapipe/solutions/guide
#https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

#install mediapipe with
#pip install mediapipe opencv-python

import cv2
import mediapipe as mp
import numpy as np

#Newest version of numpy is incompatible
#mediapipe 0.10.21 requires numpy<2, but you have numpy 2.2.4 which is incompatible.
#ultralytics 8.3.100 requires numpy<=2.1.1,>=1.23.0, but you have numpy 2.2.4 which is incompatible.
#numpy==1.26.4
#python==3.12.9


# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # refine_landmarks enables iris tracking
mp_drawing = mp.solutions.drawing_utils

# based on Mediapipe documentation -> https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
# Landmark indices for left and right eyes
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144, 33]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380, 362]

# Landmark indices for left and right eyebrows
OUTER_LEFT_EYEBROW_IDX = [70, 63, 105, 66, 107]
OUTER_RIGHT_EYEBROW_IDX = [336, 296, 334, 293, 300]
INNER_LEFT_EYEBROW_IDX = [46, 53, 52, 65, 55]
INNER_RIGHT_EYEBROW_IDX = [285, 295, 282, 283, 276]

# Landmark indices for lips
UPPER_OUTER_LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
UPPER_INNER_LIP_IDX = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_OUTER_LIP_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LOWER_INNER_LIP_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Landmark indices for face contour
FACE_CONTOUR_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                    288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                    150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]


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
            draw_feature(frame, face_landmarks, RIGHT_EYE_IDX, width, height, (0, 255, 0))      # Green
            draw_feature(frame, face_landmarks, OUTER_LEFT_EYEBROW_IDX, width, height, (0, 255, 255)) # Yellow
            draw_feature(frame, face_landmarks, OUTER_RIGHT_EYEBROW_IDX, width, height, (0, 255, 255))# Yellow
            draw_feature(frame, face_landmarks, INNER_LEFT_EYEBROW_IDX, width, height, (0, 255, 255)) # Yellow
            draw_feature(frame, face_landmarks, INNER_RIGHT_EYEBROW_IDX, width, height, (0, 255, 255)) # Yellow
            draw_feature(frame, face_landmarks, UPPER_OUTER_LIP_IDX, width, height, (128, 0, 255))  # Violett
            draw_feature(frame, face_landmarks, UPPER_INNER_LIP_IDX, width, height, (128, 0, 255))  # Violett
            draw_feature(frame, face_landmarks, LOWER_OUTER_LIP_IDX, width, height, (128, 0, 255))  # Violett
            draw_feature(frame, face_landmarks, LOWER_INNER_LIP_IDX, width, height, (128, 0, 255))  # Violett
            draw_feature(frame, face_landmarks, FACE_CONTOUR_IDX, width, height, (255, 165, 0))  # Orange


    cv2.imshow("Biometric Face Recognition", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()