import os
import cv2
import numpy as np
import time
import mediapipe as mp
from datetime import datetime

# HINWEIS: Viele Teile des ursprünglichen Codes, die sich auf Landmarken-Speicherung,
# Vergleich und Namenseingabe beziehen, wurden für diese Aufgabe entfernt,
# um sich auf die reine Gesichtserkennung und Bounding Box zu konzentrieren.

class FaceDetectionSystem:
    def __init__(self):
        # Initialize MediaPipe Face Mesh components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # --- TODO 1: MediaPipe FaceMesh initialisieren ---
        # Initialisieren Sie hier das 'FaceMesh'-Objekt aus 'self.mp_face_mesh'.
        # Verwenden Sie geeignete Parameter. Recherchieren Sie in der MediaPipe-Dokumentation
        # Parameter wie 'max_num_faces', 'min_detection_confidence', 'min_tracking_confidence'.
        # Beispiel: self.face_mesh = self.mp_face_mesh.FaceMesh(...)
        self.face_mesh = None # Platzhalter - Muss ersetzt werden!
        # --- Ende TODO 1 ---

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Camera could not be opened")

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Initialization complete (Task 1). Press 'q' to exit.")

    def extract_face_locations(self, image):
        """
        Detect faces using MediaPipe and extract their bounding box locations.
        """
        if self.face_mesh is None:
             print("ERROR: FaceMesh wurde nicht initialisiert (siehe TODO 1)")
             return [] # Return empty list if not initialized

        # Convert image to RGB (MediaPipe requires RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # Improve performance

        # --- TODO 2: Bild mit MediaPipe verarbeiten ---
        # Verarbeiten Sie das 'image_rgb' mit dem initialisierten 'self.face_mesh'-Objekt.
        # Das Ergebnis enthält Informationen über erkannte Gesichter.
        # Beispiel: results = self.face_mesh.process(...)
        results = None # Platzhalter - Muss ersetzt werden!
        # --- Ende TODO 2 ---

        image_rgb.flags.writeable = True # Re-enable writing

        face_locations_list = []

        if results and results.multi_face_landmarks:
            h, w, _ = image.shape

            # Iterate through each detected face
            for face_landmarks in results.multi_face_landmarks:
                # Calculate face location (bounding box)
                x_min, y_min = w, h
                x_max, y_max = 0, 0

                # Iterate over all landmarks of the current face to find boundaries
                for landmark in face_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    # Update min/max coordinates to find the bounding box
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
        Detect faces and draw bounding boxes on the frame.
        """
        # Create a copy of the frame for display
        display_frame = frame.copy()

        # Detect face locations using MediaPipe
        face_locations = self.extract_face_locations(frame)

        # Iterate through detected faces
        for face_loc in face_locations:
            top, right, bottom, left = face_loc

            # --- TODO 3: Bounding Box zeichnen ---
            # Zeichnen Sie ein Rechteck (Bounding Box) um das erkannte Gesicht.
            # Verwenden Sie die Koordinaten 'top', 'right', 'bottom', 'left'.
            # Nutzen Sie die OpenCV-Funktion 'cv2.rectangle'.
            # Wählen Sie eine Farbe (z.B. Grün) und Linienstärke.
            # Beispiel: cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # --- Ende TODO 3 ---
            pass # Entfernen Sie 'pass', sobald Sie den Code für TODO 3 eingefügt haben

        return display_frame

    def run(self):
        """Main loop for face detection"""
        cv2.namedWindow('Face Detection with MediaPipe (Task 1)')

        while True:
            # Capture frame
            ret, frame = self.cap.read()

            if not ret:
                print("Error capturing frame")
                break

            # Flip frame (selfie view)
            frame = cv2.flip(frame, 1)

            # Detect faces and draw boxes
            display_frame = self.detect_and_draw_boxes(frame)

            # Display result frame
            cv2.imshow('Face Detection with MediaPipe (Task 1)', display_frame)

            # Process keyboard inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # Release camera and close windows
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_system = FaceDetectionSystem()
    face_system.run()
