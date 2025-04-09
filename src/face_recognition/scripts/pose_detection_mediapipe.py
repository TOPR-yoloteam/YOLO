import cv2
import mediapipe as mp

# Mediapipe Pose Modul initialisieren
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Webcam öffnen
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Bild spiegeln, da Webcam-Bild gespiegelt ist
    frame = cv2.flip(frame, 1)

    # Umwandlung in RGB, da Mediapipe RGB benötigt
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose-Erkennung
    results = pose.process(rgb_frame)

    # Wenn Körperlandmarks erkannt wurden, zeichne sie auf das Bild
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Anzeigen des Bildes
    cv2.imshow("Körperhaltungserkennung", frame)

    # Mit 'ESC' beenden
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
