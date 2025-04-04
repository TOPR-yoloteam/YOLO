#Sources:
#https://medium.com/@erdibillkaan/object-detection-and-face-recognition-with-yolov8n-356b22bacc48
#https://github.com/carolinedunn/facial_recognition/tree/main
#https://github.com/akanametov/yolo-face/tree/dev


import cv2
from ultralytics import YOLO
import face_recognition
import numpy as np
import os
import time
import threading
from queue import Queue
import torch  # Import PyTorch module

# Configuration parameters for performance optimization
RESIZE_FACTOR = 0.5  # Factor for frame resizing
PROCESS_EVERY_N_FRAMES = 5  # Only process every N-th frame completely
MAX_QUEUE_SIZE = 5  # Maximum number of frames in the queue

# Load YOLO model with GPU support (if available)
model = YOLO("yolov8n.pt")
model.to('cuda' if torch.cuda.is_available() else 'cpu')

known_faces_dir = '/Users/jangaschler/PycharmProjects/YOLO/src/face_recognition/data/img'
known_faces_encodings = []
known_faces_names = []

# Load known faces once
for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg",".png",".jpeg",".webp")):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)

        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            known_faces_encodings.append(face_encodings[0])

            name = os.path.splitext(filename)[0]
            known_faces_names.append(name)

# Global variables for threading
frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
processed_frame = None
running = True
frame_count = 0

def capture_video():
    """Thread function for capturing video frames"""
    global running
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            running = False
            break

        # Skip if queue is full
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            time.sleep(0.01)  # Short pause to relieve CPU

    cap.release()

def process_frames():
    """Thread function for processing frames"""
    global processed_frame, running, frame_count

    while running:
        if frame_queue.empty():
            time.sleep(0.01)  # Wait for new frames
            continue

        frame = frame_queue.get()
        frame_count += 1

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

        # Complete analysis only on every N-th frame
        full_processing = (frame_count % PROCESS_EVERY_N_FRAMES == 0)

        if full_processing:
            # Perform YOLO detection on reduced frame
            results = model(small_frame)

            # Scale to original size for display
            output_frame = frame.copy()

            for result in results:
                for box in result.boxes:
                    if int(box.cls[0].item()) == 0:  # Person detected
                        # Scale coordinates to original frame
                        x1, y1, x2, y2 = map(int, box.xyxy[0] / RESIZE_FACTOR)

                        # Extract ROI (only area with detected person)
                        person_roi = frame[y1:y2, x1:x2]

                        if person_roi.size > 0:  # Check if ROI is not empty
                            # Only search for faces in ROI area
                            rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb_roi)

                            if face_locations:
                                face_encodings = face_recognition.face_encodings(rgb_roi, face_locations)

                                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                                    # Convert coordinates from ROI to main image
                                    face_top = top + y1
                                    face_right = right + x1
                                    face_bottom = bottom + y1
                                    face_left = left + x1

                                    # Face recognition
                                    matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
                                    name = "Unknown"

                                    if len(known_faces_encodings) > 0:
                                        face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
                                        best_match_index = np.argmin(face_distances)
                                        if matches[best_match_index]:
                                            name = known_faces_names[best_match_index]

                                    # Mark face
                                    cv2.rectangle(output_frame, (face_left, face_top), (face_right, face_bottom), (0, 255, 0), 2)
                                    cv2.rectangle(output_frame, (face_left, face_bottom - 35), (face_right, face_bottom), (0, 255, 0), cv2.FILLED)
                                    cv2.putText(output_frame, name, (face_left + 6, face_bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

                        # Draw person bounding box
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = model.names[int(box.cls)]
                        confidence = box.conf[0]
                        cv2.putText(output_frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            processed_frame = output_frame
        else:
            # For skipped frames, only display the last result
            # or perform simpler operations if needed
            if processed_frame is None:
                processed_frame = frame

# Start threads
capture_thread = threading.Thread(target=capture_video)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()

# Main loop for display
while running:
    if processed_frame is not None:
        cv2.imshow('Object Detection', processed_frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# Cleanup
capture_thread.join()
process_thread.join()
cv2.destroyAllWindows()
