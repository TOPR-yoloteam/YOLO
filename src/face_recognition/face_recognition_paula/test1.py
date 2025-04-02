#Sources:
#https://medium.com/@erdibillkaan/object-detection-and-face-recognition-with-yolov8n-356b22bacc48
#https://github.com/carolinedunn/facial_recognition/tree/main
#https://github.com/akanametov/yolo-face/tree/dev


import cv2
from ultralytics import YOLO
import face_recognition

#OpenCV needs to be installed with
#conda install -c conda-forge opencv
#Ultralytics needs to be installed with
#pip install ultralytics

#face_recognition needs to be installed with
#conda install dlib -c conda-forge
#conda install face_recognition -c conda-forge

import numpy as np
import os

model = YOLO("yolov8n.pt")

known_faces_dir = '/Users/jangaschler/PycharmProjects/YOLO/src/face_recognition/data/img'
known_faces_encodings = []
known_faces_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg",".png",".jpeg",".webp")):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)

        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            known_faces_encodings.append(face_encodings[0])

            name = os.path.splitext(filename)[0]
            known_faces_names.append(name)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            if int(box.cls[0].item()) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_frame = frame[y1:y2, x1:x2]

                rgb_small_frame = np.ascontiguousarray(frame[:, :, ::-1])

                face_locations = face_recognition.face_locations(rgb_small_frame)

                if face_locations is not None and face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    face_names = []
                    for face_encoding in face_encodings:

                        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
                        name = "Unknown"

                        face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_faces_names[best_match_index]

                        face_names.append(name)

                    for (top, right, bottom, left), name in zip(face_locations, face_names):

                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = model.names[int(box.cls)]
                confidence = box.conf[0]
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()