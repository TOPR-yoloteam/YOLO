import os
import cv2
import numpy as np
import time
from ultralytics import YOLO

class FaceRecognitionSystemTask1:
    def __init__(self):
        """
        Initializes the face recognition system for Task 1:
        - Loads YOLO model for face detection
        - Sets up camera
        """
        # Paths and directories
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "yolov8n-face.pt")

        # --- TODO Task 1.1: Load YOLO model ---
        # Load the YOLO model with the path 'self.model_path'
        # Use the YOLO class from the ultralytics library
        # Example: self.model = YOLO(...)
        self.model = None  # Placeholder - Must be replaced!
        # --- End TODO Task 1.1 ---

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Initialization complete (Task 1). Press 'q' to quit.")

    def detect_faces(self, frame):
        """
        Detects faces in the given frame using YOLO.

        Args:
            frame (np.ndarray): Current video frame.

        Returns:
            list: A list of detected bounding boxes in the format [(x1, y1, x2, y2), ...].
        """
        # --- TODO Task 1.2: Perform YOLO detection ---
        # Process the 'frame' with the YOLO model (self.model)
        # Limit detection to class 0 (person/face)
        # Example: results = self.model(...)
        results = None  # Placeholder - Must be replaced!
        # --- End TODO Task 1.2 ---

        bounding_boxes = []

        # Check if results are available
        if results and len(results) > 0:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # --- TODO Task 1.3: Extract bounding box coordinates ---
                    # Extract the coordinates (x1, y1, x2, y2) from the box object (box.xyxy[0])
                    # Convert the values to integers using int()
                    # Add the extracted coordinates to the bounding_boxes list
                    # Example: x1, y1, x2, y2 = map(int, ...)
                    # bounding_boxes.append((x1, y1, x2, y2))
                    pass  # Remove this line and add your code
                    # --- End TODO Task 1.3 ---

        return bounding_boxes

    def draw_boxes(self, frame, bounding_boxes):
        """
        Draws bounding boxes on the frame.

        Args:
            frame (np.ndarray): Input video frame.
            bounding_boxes (list): List of bounding box coordinates [(x1, y1, x2, y2), ...].

        Returns:
            np.ndarray: Frame with drawn bounding boxes.
        """
        display_frame = frame.copy()

        for x1, y1, x2, y2 in bounding_boxes:
            # --- TODO Task 1.4: Draw bounding box ---
            # Draw a rectangle on the display_frame with coordinates (x1, y1) and (x2, y2)
            # Use the OpenCV function cv2.rectangle
            # Choose a green color (0, 255, 0) and a line thickness of 2
            # Example: cv2.rectangle(...)
            pass  # Remove this line and add your code
            # --- End TODO Task 1.4 ---

        return display_frame

    def run(self):
        """
        Main loop for the face recognition system (Task 1).
        
        Continuously captures frames, detects faces, and displays the result.
        """
        window_title = 'Face Recognition with YOLO (Task 1)'
        cv2.namedWindow(window_title)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
                
            frame = cv2.flip(frame, 1)  # Horizontal flip for selfie view

            # Perform face detection
            bounding_boxes = self.detect_faces(frame)
            
            # Draw bounding boxes
            display_frame = self.draw_boxes(frame, bounding_boxes)
            
            # Show result
            cv2.imshow(window_title, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting program...")
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

# Entry point of the script
if __name__ == "__main__":
    face_system = FaceRecognitionSystemTask1()
    face_system.run()