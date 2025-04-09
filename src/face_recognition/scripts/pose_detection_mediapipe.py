import cv2
import mediapipe as mp

# initialize Mediapipe Pose Modul
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # mirror image because webcam image is mirrored
    frame = cv2.flip(frame, 1)

    # convert picture in RGB (MediaPipe needs that)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # pose detection
    results = pose.process(rgb_frame)

    # if body landmarks have been detected, draw them on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # show results
    cv2.imshow("pose detection", frame)

    # close with 'esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
