import cv2
import mediapipe as mp
import numpy as np
import time
import pygame

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indices (Mediapipe)
LEFT_EYE_EAR_IDX = {
    "top": [159, 160],
    "bottom": [145, 144],
    "left": 33,
    "right": 133
}

RIGHT_EYE_EAR_IDX = {
    "top": [386, 385],
    "bottom": [374, 380],
    "left": 362,
    "right": 263
}

# Initialize pygame mixer
pygame.mixer.init()

def play_sound(file_path):
    pygame.mixer.music.load(file_path)  # load sound file
    pygame.mixer.music.play()  # play alarm
    while pygame.mixer.music.get_busy():  # wait until alarm is finished
        pygame.time.Clock().tick(10)

# Function to calculate EAR (Eye Aspect Ratio)
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(landmarks, idxs, width, height):
    left_point = landmarks[idxs["left"]]
    right_point = landmarks[idxs["right"]]
    hor_dist = euclidean_distance((left_point.x * width, left_point.y * height),
                                  (right_point.x * width, right_point.y * height))

    top_mean = np.mean([
        [landmarks[i].x * width, landmarks[i].y * height]
        for i in idxs["top"]
    ], axis=0)

    bottom_mean = np.mean([
        [landmarks[i].x * width, landmarks[i].y * height]
        for i in idxs["bottom"]
    ], axis=0)

    ver_dist = euclidean_distance(top_mean, bottom_mean)
    ear = ver_dist / hor_dist
    return ear


# Webcam
cap = cv2.VideoCapture(0)

# Variables for blink detection and fatigue detection
blink_threshold = 0.20  # EAR threshold for detecting blink
blink_counter = 0  # Count the number of frames the EAR is below the threshold
blink_duration = 1  # Duration in seconds to consider as a blink
fatigue_threshold = 4  # Time threshold (in seconds) to consider as fatigue (both eyes closed)
last_blink_time = 0  # Last blink timestamp
fatigue_time = 0  # Time when both eyes were closed for a long duration

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Compute EAR for both eyes
            ear_left = compute_ear(face_landmarks.landmark, LEFT_EYE_EAR_IDX, width, height)
            ear_right = compute_ear(face_landmarks.landmark, RIGHT_EYE_EAR_IDX, width, height)

            # Detect eye states (OPEN or CLOSED)
            left_eye_state = "OPEN" if ear_left > blink_threshold else "CLOSED"
            right_eye_state = "OPEN" if ear_right > blink_threshold else "CLOSED"

            # Check for blinking (both eyes closed for a brief moment)
            if left_eye_state == "CLOSED" and right_eye_state == "CLOSED":
                if blink_counter == 0:
                    last_blink_time = time.time()  # Start the timer for blinking

                blink_counter += 1
                # Display blinking detected message
                cv2.putText(frame, "Blinking Detected", (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
            else:
                blink_counter = 0  # Reset blink counter if eyes are open

            # Detect fatigue (both eyes closed for too long)
            if left_eye_state == "CLOSED" and right_eye_state == "CLOSED":
                if fatigue_time == 0:
                    fatigue_time = time.time()  # Start the timer for fatigue detection

                # If both eyes have been closed for more than the fatigue threshold
                if time.time() - fatigue_time > fatigue_threshold:
                    # Play alarm sound when fatigue is detected
                    play_sound('/YOLO/src/face_recognition/data/audio/alarm_sound.wav')  # Make sure to provide the correct path to the sound file

            else:
                fatigue_time = 0  # Reset fatigue timer if eyes are open

            # Display the eye states
            cv2.putText(frame, f"Left Eye: {left_eye_state}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0) if left_eye_state == "OPEN" else (0, 0, 255), 2)
            cv2.putText(frame, f"Right Eye: {right_eye_state}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0) if right_eye_state == "OPEN" else (0, 0, 255), 2)

    # Show the result frame
    cv2.imshow("Test", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
