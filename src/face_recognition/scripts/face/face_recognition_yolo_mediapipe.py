import cv2
import mediapipe as mp
from ultralytics import YOLO
import torch

# Load YOLOv11 model
model = YOLO("YOLO/src/face_recognition/models/yolov8n.pt")

# Setup MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(source=frame, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue

                # Convert ROI to RGB for MediaPipe
                rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                mp_result = face_mesh.process(rgb_face)

                # Draw landmarks
                if mp_result.multi_face_landmarks:
                    for landmarks in mp_result.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=face_roi,
                            landmark_list=landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec
                        )

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 0), 2)

    cv2.imshow("YOLOv8 + Mediapipe Face Mesh", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
