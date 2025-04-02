from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("license_plate_detector.pt")

# Export the model to NCNN format
model.export(format="ncnn")