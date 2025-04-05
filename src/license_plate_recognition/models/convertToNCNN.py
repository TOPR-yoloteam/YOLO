from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("licence_plate_yolov5.pt")

# Export the model to NCNN format
model.export(format="ncnn")