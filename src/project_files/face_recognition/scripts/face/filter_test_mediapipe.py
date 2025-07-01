import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5)
mp_drawing = mp.solutions.drawing_utils

# Build absolute path to the sunglasses image relative to this script
base_dir = os.path.dirname(__file__)
glasses_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'img', 'sunglasses.png'))
glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)

# Check if the image was loaded successfully
if glasses is None:
    print(f"Error: Could not load image.\nChecked path: {glasses_path}")
    exit()
else:
    print(f"Image loaded successfully: {glasses_path}")

# Start webcam capture
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get coordinates of left eye, right eye, and nose
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose = face_landmarks.landmark[168]

            # Convert normalized coordinates to pixel values
            x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
            x2, y2 = int(right_eye.x * w), int(right_eye.y * h)
            xn, yn = int(nose.x * w), int(nose.y * h)

            # Calculate glasses width based on distance between eyes
            glasses_width = int(np.linalg.norm([x2 - x1, y2 - y1]) * 1.65)
            scale = glasses_width / glasses.shape[1]
            new_h = int(glasses.shape[0] * scale)

            # Resize the glasses image
            resized_glasses = cv2.resize(glasses, (glasses_width, new_h), interpolation=cv2.INTER_AREA)

            # Determine position to place the glasses
            x_offset = xn - glasses_width // 2
            y_offset = yn - new_h // 2

            # Overlay the glasses image onto the video frame
            for c in range(0, 3):
                alpha = resized_glasses[:, :, 3] / 255.0
                y1_clip = max(0, y_offset)
                y2_clip = min(h, y_offset + new_h)
                x1_clip = max(0, x_offset)
                x2_clip = min(w, x_offset + glasses_width)

                frame[y1_clip:y2_clip, x1_clip:x2_clip, c] = (
                    alpha[y1_clip - y_offset:y2_clip - y_offset, x1_clip - x_offset:x2_clip - x_offset] *
                    resized_glasses[y1_clip - y_offset:y2_clip - y_offset, x1_clip - x_offset:x2_clip - x_offset, c] +
                    (1 - alpha[y1_clip - y_offset:y2_clip - y_offset, x1_clip - x_offset:x2_clip - x_offset]) *
                    frame[y1_clip:y2_clip, x1_clip:x2_clip, c]
                )

    # Display the result
    cv2.imshow('Sunglasses Filter', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
