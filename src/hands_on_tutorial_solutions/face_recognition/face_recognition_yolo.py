import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
import face_recognition
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self):
        """
        Initializes the face recognition system:
        - Loads YOLO model for person detection.
        - Sets up face directory and camera.
        - Loads known faces into memory.
        """
        # Paths and directories
        self.faces_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "img")
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "yolov8n-face.pt")

        # Create faces directory if it doesn't exist
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)

        # Load YOLO model
        self.model = YOLO(self.model_path)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Known faces and names
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

        # UI states and face interaction
        self.button_area = []
        self.face_to_learn = None
        self.current_frame = None
        self.state = "normal"  # Modes: "normal" or "entering_name"
        self.current_text = ""
        self.selected_face_loc = None
        self.text_entry_active = False

        # Recognition frame size
        self.recognition_size = (0, 0)

        print("Initialization complete. Press 'q' to quit.")

    def load_known_faces(self):
        """
        Loads all known faces from the 'faces_dir' into memory.
        Encodes faces for future recognition.
        """
        self.known_face_encodings = []
        self.known_face_names = []

        for filename in os.listdir(self.faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.faces_dir, filename)
                face_image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(face_image)

                if face_locations:
                    face_encoding = face_recognition.face_encodings(face_image, face_locations)[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)

        print(f"Loaded {len(self.known_face_names)} known faces")

    def save_face(self, name, face_location):
        """
        Saves a cropped and expanded face image with a given name to disk.

        Args:
            name (str): Name to associate with the saved face.
            face_location (tuple): Coordinates (top, right, bottom, left) of the face.
        """
        if not name:
            return

        top, right, bottom, left = face_location

        # Expand region
        height = bottom - top
        width = right - left
        expanded_top = max(0, top - int(height * 0.3))
        expanded_bottom = min(self.current_frame.shape[0], bottom + int(height * 0.3))
        expanded_left = max(0, left - int(width * 0.3))
        expanded_right = min(self.current_frame.shape[1], right + int(width * 0.3))

        face_image = self.current_frame[expanded_top:expanded_bottom, expanded_left:expanded_right]

        # Save with timestamp if filename already exists
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}.jpg"
        filepath = os.path.join(self.faces_dir, filename)
        if os.path.exists(filepath):
            filepath = os.path.join(self.faces_dir, f"{name}_{timestamp}.jpg")

        cv2.imwrite(filepath, face_image)
        print(f"Saved face as {filepath}")

        # Reload known faces
        self.load_known_faces()

    def mouse_callback(self, event, x, y, flags, param):
        """
        Handles mouse events:
        - Clicking on a 'Learn Face' button triggers entering name mode.

        Args:
            event: OpenCV mouse event.
            x (int): X coordinate of the click.
            y (int): Y coordinate of the click.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state == "normal":
                for (button_left, button_top, button_right, button_bottom), face_location in self.button_area:
                    if button_left <= x <= button_right and button_top <= y <= button_bottom:
                        self.state = "entering_name"
                        self.selected_face_loc = face_location
                        self.current_text = ""
                        self.text_entry_active = True
                        break
            elif self.state == "entering_name" and not self.text_entry_active:
                self.state = "normal"
                self.text_entry_active = False

    def detect_and_recognize_faces(self, frame):
        """
        Detects people using YOLO and recognizes known faces.

        Args:
            frame (np.array): Current video frame.

        Returns:
            np.array: Frame with bounding boxes and labels drawn.
            list: List of detected face locations.
        """
        display_frame = frame.copy()
        self.current_frame = frame.copy()

        height, width = frame.shape[:2]
        self.recognition_size = (width, height)

        # YOLO detection
        results = self.model(frame, classes=[0])

        if results and len(results) > 0:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        self.button_area = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            is_known_face = False

            if self.known_face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        is_known_face = True

            # Draw face box and label
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            label_top = bottom + 10
            label_bottom = bottom + 45
            cv2.rectangle(display_frame, (left, label_top), (right, label_bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(display_frame, name, (left + 6, label_top + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

            # If face is unknown, offer 'Learn Face'
            if self.state == "normal" and not is_known_face:
                button_left = left
                button_top = top - 30
                button_right = right
                button_bottom = top

                if button_top > 0:
                    cv2.rectangle(display_frame, (button_left, button_top), (button_right, button_bottom), (255, 0, 0), cv2.FILLED)
                    cv2.putText(display_frame, "Learn Face", (button_left + 5, button_top + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                    self.button_area.append(((button_left, button_top, button_right, button_bottom), (top, right, bottom, left)))

        return display_frame, face_locations

    def draw_text_input(self, frame):
        """
        Draws a text input field at the bottom of the frame for entering names.

        Args:
            frame (np.array): Current video frame.

        Returns:
            np.array: Frame with text input field drawn.
        """
        height, width = frame.shape[:2]
        input_height = 40

        # Draw input background
        cv2.rectangle(frame, (0, height - input_height), (width, height), (50, 50, 50), cv2.FILLED)

        # Show text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Enter name: {self.current_text}_"
        cv2.putText(frame, text, (10, height - 15), font, 0.7, (255, 255, 255), 1)

        # Instructions
        instructions = "Press ENTER to save, ESC to cancel"
        cv2.putText(frame, instructions, (width - 300, height - 15), font, 0.5, (200, 200, 200), 1)

        return frame

    def run(self):
        """
        Main loop for running the face recognition system:
        - Handles normal detection mode.
        - Handles entering name mode for new faces.
        """
        cv2.namedWindow('Face Recognition')
        cv2.setMouseCallback('Face Recognition', self.mouse_callback)

        # Short initial delay to control FPS
        time.sleep(0.07)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)

            if self.state == "normal":
                display_frame, _ = self.detect_and_recognize_faces(frame)
                cv2.imshow('Face Recognition', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            elif self.state == "entering_name":
                display_frame = frame.copy()

                if self.selected_face_loc:
                    top, right, bottom, left = self.selected_face_loc
                    height = bottom - top
                    width = right - left
                    expanded_top = max(0, top - int(height * 0.3))
                    expanded_bottom = min(display_frame.shape[0], bottom + int(height * 0.3))
                    expanded_left = max(0, left - int(width * 0.3))
                    expanded_right = min(display_frame.shape[1], right + int(width * 0.3))

                    cv2.rectangle(display_frame, (expanded_left, expanded_top), (expanded_right, expanded_bottom), (255, 255, 0), 2)
                    cv2.putText(display_frame, "Face to save", (expanded_left, expanded_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                display_frame = self.draw_text_input(display_frame)
                cv2.imshow('Face Recognition', display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == 13:  # ENTER
                    if self.current_text:
                        self.save_face(self.current_text, self.selected_face_loc)
                    self.state = "normal"
                elif key == 27:  # ESC
                    self.state = "normal"
                elif key == 8:  # BACKSPACE
                    self.current_text = self.current_text[:-1]
                elif key == ord('q'):  # Quit
                    break
                elif 32 <= key <= 126:
                    self.current_text += chr(key)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_system = FaceRecognitionSystem()
    face_system.run()
