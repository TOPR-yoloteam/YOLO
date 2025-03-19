from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="C:/Users/serha/PycharmProjects/PythonProject3/src/data.yaml",imgsz=640,batch=1, epochs=20)