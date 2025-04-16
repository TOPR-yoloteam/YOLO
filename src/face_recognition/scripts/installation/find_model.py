hrom ultralytics import YOLO

import os

os.chdir("/YOLO/")

model = YOLO('src/license_plate_recognition/models/license_plate_detector.pt')
print(model.names)  # Shows class names