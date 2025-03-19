import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train8/weights/best.pt")

model(source=0,show= True, conf=0.65, iou=0.8)