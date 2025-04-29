import os
import cv2
import numpy as np
import time
import mediapipe as mp
from datetime import datetime

# NOTE: Many parts of the original code related to landmark storage,
# comparison, and name entry have been removed for this task
# to focus purely on face detection and drawing bounding boxes.

class FaceDetectionSystem:
    def __init__(self):
        """
        Initializes the FaceDetectionSystem.

        Sets up MediaPipe components, camera, and basic configurations.
        """
        # Initialize MediaPipe Face Mesh components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils # Utility for drawing landmarks (not used in Task 1)

        # --- TODO 1: Initialize MediaPipe FaceMesh ---
        # Initialize the 'FaceMesh' object from 'self.mp_face_mesh' here.
        # Use appropriate parameters. Consult the MediaPipe documentation
        # for parameters like 'max_num_faces', 'min_detection_confidence', 'min_tracking_confidence'.
        # Example: self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.7)
        self.face_mesh = None # Placeholder - Must be replaced!
        # --- End TODO 1 ---

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
        Detects faces using MediaPipe and extracts their bounding box locations.

        Args:
            image (np.ndarray): The input image frame (BGR format from OpenCV).

        Returns:
            list: A list of tuples, each representing the bounding box
                  (top, right, bottom, left) of a detected face.
        """
        # Check if FaceMesh was initialized in __init__
        if self.face_mesh is None:
             print("ERROR: FaceMesh was not initialized (see TODO 1)")
             return [] # Return empty list if not initialized

        # Convert BGR image to RGB, required by MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Mark image as not writeable to pass by reference (improves performance)
        image_rgb.flags.writeable = False

        # --- TODO 2: Process Image with MediaPipe ---
        # Process the 'image_rgb' using the initialized 'self.face_mesh' object.
        # The result will contain information about detected faces and their landmarks.
        # Example: results = self.face_mesh.process(image_rgb)
        results = None # Placeholder - Must be replaced!
        # --- End TODO 2 ---

        # Mark image as writeable again
        image_rgb.flags.writeable = True

        face_locations_list = []

        # Check if any faces were detected and landmarks are available
        if results and results.multi_face_landmarks:
            h, w, _ = image.shape # Get image dimensions for coordinate conversion

            # Iterate through each detected face
            for face_landmarks in results.multi_face_landmarks:
                # Calculate face location (bounding box) by finding min/max coordinates
                x_min, y_min = w, h # Initialize with max values
                x_max, y_max = 0, 0 # Initialize with min values

                # Iterate over all landmarks of the current face to find the boundaries
                for landmark in face_landmarks.landmark:
                    # Convert normalized coordinates (0.0-1.0) to pixel coordinates
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    # Update min/max coordinates
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Add the calculated bounding box coordinates to the list
                # Format: (top, right, bottom, left) - compatible with face_recognition library style
                face_locations_list.append((y_min, x_max, y_max, x_min))

        return face_locations_list

    def detect_and_draw_boxes(self, frame):
        """
        Detects faces in the frame and draws bounding boxes around them.

        Args:
            frame (np.ndarray): The input video frame (BGR).

        Returns:
            np.ndarray: The frame with bounding boxes drawn around detected faces.
        """
        # Create a copy of the frame to draw on, preserving the original
        display_frame = frame.copy()

        # Detect face locations using MediaPipe
        face_locations = self.extract_face_locations(frame)

        # Iterate through detected faces and draw boxes
        for face_loc in face_locations:
            top, right, bottom, left = face_loc # Unpack coordinates

            # --- TODO 3: Draw Bounding Box ---
            # Draw a rectangle (bounding box) around the detected face.
            # Use the coordinates 'top', 'right', 'bottom', 'left'.
            # Use the OpenCV function 'cv2.rectangle'.
            # Choose a color (e.g., green as BGR tuple) and line thickness.
            # Example: cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # --- End TODO 3 ---
            pass # Remove 'pass' once you add the code for TODO 3

        return display_frame # Return the annotated frame

    def run(self):
        """
        Runs the main loop for the face detection system.

        Continuously captures frames from the camera, detects faces,
        draws bounding boxes, and displays the result.
        """
        window_title = 'Face Detection with MediaPipe (Task 1)'
        cv2.namedWindow(window_title) # Create a display window

        while True:
            # Capture frame-by-frame from the camera
            ret, frame = self.cap.read()

            # Check if frame was captured successfully
            if not ret:
                print("Error: Failed to capture frame.")
                break # Exit loop if frame capture fails

            # Flip the frame horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)

            # Detect faces and draw bounding boxes on the frame
            display_frame = self.detect_and_draw_boxes(frame)

            # Display the resulting frame in the window
            cv2.imshow(window_title, display_frame)

            # Process keyboard inputs
            key = cv2.waitKey(1) & 0xFF # Wait for a key press (1ms delay)
            # Check if 'q' was pressed to quit
            if key == ord('q'):
                print("Exiting...")
                break

        # --- Cleanup ---
        # Release the camera resource
        self.cap.release()
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print("Resources released.")

# Script entry point
if __name__ == "__main__":
    # Create an instance of the detection system
    face_system = FaceDetectionSystem()
    # Start the main processing loop
    face_system.run()
