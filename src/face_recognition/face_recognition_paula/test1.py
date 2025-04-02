#Sources:
#https://medium.com/@erdibillkaan/object-detection-and-face-recognition-with-yolov8n-356b22bacc48
#https://github.com/carolinedunn/facial_recognition/tree/main
#https://github.com/akanametov/yolo-face/tree/dev


import cv2
from ultralytics import YOLO
import face_recognition

#face_recognition needs to be installed with
#conda install dlib -c conda-forge
#conda install face_recognition -c conda-forge

import numpy as np
import os

model = YOLO("yolov8n.pt")

known_faces_dir = "YOLO/src/face_recognition/data/img/"
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
