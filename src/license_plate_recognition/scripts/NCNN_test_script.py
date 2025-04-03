import time
import threading
import cv2
from ultralytics import YOLO
import os

# Set the working directory
os.chdir("/home/talmva/workspace/YOLO/")

# Initialize the YOLO model
model = YOLO("src/license_plate_recognition/models/licence_plate_yolov5_ncnn_model")

# Open the webcam (0 is typically the default webcam)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)

# Check if the capture device is opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video source.")
    exit()

# Initialize FPS variables
frame_count = 0
start_time = time.time()
fps = 0


# Function to calculate and update FPS in a separate thread
def calculate_fps():
    global frame_count, start_time, fps
    while True:
        time.sleep(1)  # Update FPS once per second
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0


# Start the FPS calculation thread
fps_thread = threading.Thread(target=calculate_fps, daemon=True)
fps_thread.start()

# Main loop
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Ensure the frame is valid before processing
    if not ret or frame is None:
        print("Error: Failed to capture a valid frame.")
        break

    # Run inference with the YOLO model
    results = model(frame)

    # Increment frame count
    frame_count += 1

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
