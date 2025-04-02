import cv2
from ultralytics import YOLO
import os

# Set the working directory
os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO")

# Initialize the YOLO model
model = YOLO("src/license_plate_recognition/models/license_plate_detector_ncnn_model")

# Open the webcam (0 is typically the default webcam)
cap = cv2.VideoCapture(0)

# Check if the capture device is opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video source.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Ensure the frame is valid before processing
    if not ret or frame is None:
        print("Error: Failed to capture a valid frame.")
        break

    # Run inference with the YOLO model
    results = model(frame)

    # Draw bounding boxes on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame with annotations
    cv2.imshow("Kennzeichen-Erkennung", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close display windows
cap.release()
cv2.destroyAllWindows()
