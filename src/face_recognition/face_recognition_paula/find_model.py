from ultralytics import YOLO
import os

os.chdir("C:/Users/pauli/Programming/PycharmProjects/YOLO/YOLO/")

model = YOLO('src/license_plate_recognition/models/license_plate_detector.pt')
print(model.names)  # Shows class names