import os
import cv2
import numpy as np
import time
import mediapipe as mp
from datetime import datetime

class FaceRecognitionSystemTask1: # Renamed class slightly for clarity
    def __init__(self):
        """
        Initializes the FaceRecognitionSystem for Task 1.

        Sets up MediaPipe components and camera.
        """
        # Initialize MediaPipe components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils # Not used in Task 1 but kept for structure

        # --- TODO Task 1.1: Initialize MediaPipe FaceMesh ---
        # Initialize the 'FaceMesh' object from 'self.mp_face_mesh' here.
        # Use appropriate parameters. Consult the MediaPipe documentation
        # for parameters like 'max_num_faces', 'min_detection_confidence', 'min_tracking_confidence'.
        # Example: self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.7)

        # --- End TODO Task 1.1 ---

        # Initialize camera
        self.cap = cv2.VideoCapture(0) # 0 is typically the default webcam
        if not self.cap.isOpened():
            raise Exception("Camera could not be opened")

        # Set camera resolution (optional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Initialization complete (Task 1). Press 'q' to exit.")

    def extract_face_locations(self, image):
        """
        Extracts face bounding box locations using MediaPipe FaceMesh.
        (Simplified for Task 1 - only returns locations)

        Args:
            image (np.ndarray): The input image frame (BGR format from OpenCV).

        Returns:
            list: A list of tuples, each representing the bounding box
                  (top, right, bottom, left) of a detected face.
        """
         # Check if FaceMesh was initialized
        if self.face_mesh is None:
             print("ERROR: FaceMesh was not initialized (see TODO Task 1.1)")
             return [] # Return empty list if not initialized

        # Convert BGR image to RGB, required by MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Mark image as not writeable to pass by reference (performance)
        image_rgb.flags.writeable = False

        # --- TODO Task 1.2: Process Image with MediaPipe ---
        # Process the 'image_rgb' using the initialized 'self.face_mesh' object.
        # The result will contain information about detected faces and their landmarks.
        # Example: results = self.face_mesh.process(image_rgb)

        # --- End TODO Task 1.2 ---

        # Mark image as writeable again
        image_rgb.flags.writeable = True

        face_locations_list = []

        # Check if any faces were detected and landmarks are available
        if results and results.multi_face_landmarks:
            h, w, _ = image.shape # Get image dimensions for coordinate conversion

            # Process each detected face
            for face_landmarks in results.multi_face_landmarks:
                # --- Calculate face location (bounding box) ---
                x_min, y_min = w, h # Initialize with max values
                x_max, y_max = 0, 0 # Initialize with min values
                # Iterate over all landmarks (not just key ones) to find the boundary
                for landmark in face_landmarks.landmark:
                    # Convert normalized coordinates (0.0-1.0) to pixel coordinates
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                # Store bounding box: (top, right, bottom, left) format
                face_locations_list.append((y_min, x_max, y_max, x_min))

                # NOTE: Landmark extraction (Task 2.1) is NOT done here for Task 1.

        # Only return locations for Task 1
        return face_locations_list

    def detect_and_recognize_faces(self, frame):
        """
        Performs face detection and draws bounding boxes for Task 1.
        (Recognition part is omitted for Task 1)

        Args:
            frame (np.ndarray): The input video frame (BGR).

        Returns:
            np.ndarray: The frame with drawn annotations (bounding boxes).
        """
        display_frame = frame.copy() # Work on a copy

        # Get only face locations for Task 1
        face_locations = self.extract_face_locations(frame)

        # Process each detected face
        for face_loc in face_locations:
            top, right, bottom, left = face_loc # Unpack bounding box coordinates

            # --- TODO Task 1.3: Draw Bounding Box ---
            # Draw a rectangle (bounding box) around the detected face.
            # Use the coordinates 'top', 'right', 'bottom', 'left'.
            # Use the OpenCV function 'cv2.rectangle'.
            # Choose a fixed color (e.g., green) and line thickness for Task 1.

            # --- End TODO Task 1.3 ---

            # NOTE: Label drawing and button logic (Task 2) are NOT included here.

        return display_frame # Return the annotated frame

    def run(self):
        """
        Runs the main loop for the face detection system (Task 1).

        Continuously captures frames, processes them, and displays the result.
        """
        window_title = 'Face Detection with MediaPipe (Task 1)'
        cv2.namedWindow(window_title)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            frame = cv2.flip(frame, 1)

            # Perform detection and draw boxes
            display_frame = self.detect_and_recognize_faces(frame)
            cv2.imshow(window_title, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break

        # --- Cleanup ---
        self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

# Entry point of the script
if __name__ == "__main__":
    face_system = FaceRecognitionSystemTask1()
    face_system.run()
