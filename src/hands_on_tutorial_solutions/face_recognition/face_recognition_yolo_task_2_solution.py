import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
import face_recognition
from datetime import datetime

# Based on the original face_recognition_yolo.py

class FaceRecognitionSystemTask2:
    def __init__(self):
        """
        Initializes the face recognition system for Task 2:
        - Loads YOLO model for face detection
        - Sets up face directory and camera
        - Loads known faces into memory
        """
        # Paths and directories
        self.faces_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "img")
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "models", "yolov8n-face.pt")

        # Create faces directory if it doesn't exist
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)

        # Load YOLO model (Assumption: Works after Task 1)
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
        self.current_frame = None
        self.state = "normal"  # Modes: "normal" or "entering_name"
        self.current_text = ""
        self.selected_face_loc = None
        self.text_entry_active = False

        print("Initialization complete (Task 2). Press 'q' to quit.")

    def load_known_faces(self):
        """
        Loads all known faces from the 'faces_dir' into memory.
        Encodes faces for future recognition.
        """
        self.known_face_encodings = []
        self.known_face_names = []

        # --- Solution Task 2.1: Load Known Faces ---
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
        # --- End Solution Task 2.1 ---

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

        # --- Solution Task 2.2: Expand and Save Face Image ---
        # Calculate face dimensions
        height = bottom - top
        width = right - left
        
        # Expand region by 30% in all directions
        expanded_top = max(0, top - int(height * 0.3))
        expanded_bottom = min(self.current_frame.shape[0], bottom + int(height * 0.3))
        expanded_left = max(0, left - int(width * 0.3))
        expanded_right = min(self.current_frame.shape[1], right + int(width * 0.3))
        
        # Crop face region
        face_image = self.current_frame[expanded_top:expanded_bottom, expanded_left:expanded_right]
        
        # Generate filename and save
        base_filename = name.lower().replace(" ", "_")
        file_path = os.path.join(self.faces_dir, f"{base_filename}.jpg")
        
        # Add timestamp if file exists
        if os.path.exists(file_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.faces_dir, f"{base_filename}_{timestamp}.jpg")
        
        cv2.imwrite(file_path, face_image)
        
        # Reload known faces
        self.load_known_faces()
        # --- End Solution Task 2.2 ---

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
            # --- Solution Task 2.3: Mouse Click Handling ---
            # Normal state - check if clicking on a Learn Face button
            if self.state == "normal":
                for i, (button_x1, button_y1, button_x2, button_y2, face_loc) in enumerate(self.button_area):
                    if button_x1 <= x <= button_x2 and button_y1 <= y <= button_y2:
                        # Switch to name entry mode
                        self.state = "entering_name"
                        self.selected_face_loc = face_loc
                        self.current_text = ""
                        break
            
            # Entering name state - cancel if clicking elsewhere (but not during active text entry)
            elif self.state == "entering_name" and not self.text_entry_active:
                self.state = "normal"
            # --- End Solution Task 2.3 ---

    def detect_and_recognize_faces(self, frame):
        """
        Detects faces using YOLO and recognizes known faces.

        Args:
            frame (np.ndarray): Current video frame.

        Returns:
            np.ndarray: Frame with drawn bounding boxes and labels.
            list: List of detected face locations.
        """
        display_frame = frame.copy()
        self.current_frame = frame.copy()

        height, width = frame.shape[:2]

        # YOLO detection
        results = self.model(frame, classes=[0])

        if results and len(results) > 0:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- Solution Task 2.4: Face Recognition and Identification ---
        # Get face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Reset button area
        self.button_area = []
        
        # Process each face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            is_known_face = False
            
            # Compare with known faces
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
            
            # Create label area below face
            label_top = bottom + 10
            label_bottom = bottom + 45
            cv2.rectangle(display_frame, (left, label_top), (right, label_bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(display_frame, name, (left + 6, label_top + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            
            # Add "Learn Face" button for unknown faces
            if not is_known_face and self.state == "normal":
                button_top = label_bottom + 5
                button_bottom = button_top + 35
                button_text = "Learn Face"
                
                cv2.rectangle(display_frame, (left, button_top), (right, button_bottom), (255, 255, 0), cv2.FILLED)
                cv2.putText(display_frame, button_text, (left + 6, button_top + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
                
                # Store button area for click detection
                self.button_area.append((left, button_top, right, button_bottom, (top, right, bottom, left)))
        # --- End Solution Task 2.4 ---

        return display_frame, face_locations

    def draw_text_input(self, frame):
        """
        Draws a text input field at the bottom of the frame for entering names.

        Args:
            frame (np.ndarray): Current video frame.

        Returns:
            np.ndarray: Frame with text input field drawn.
        """
        height, width = frame.shape[:2]
        input_height = 40

        # --- Solution Task 2.5: Draw Text Input Field ---
        # Draw background rectangle
        cv2.rectangle(frame, (0, height - input_height), (width, height), (50, 50, 50), cv2.FILLED)
        
        # Show current text with cursor
        display_text = self.current_text + "_" if self.text_entry_active else self.current_text
        cv2.putText(frame, f"Enter name: {display_text}", (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Show instructions
        instruction_text = "Press ENTER to save or ESC to cancel"
        text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = width - text_size[0] - 10
        cv2.putText(frame, instruction_text, (text_x, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        # --- End Solution Task 2.5 ---

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

                # --- Solution Task 2.6: Display Name Entry UI ---
                # Highlight the face to save
                if self.selected_face_loc:
                    top, right, bottom, left = self.selected_face_loc
                    
                    # Expand region slightly for highlighting
                    pad = 10
                    highlight_top = max(0, top - pad)
                    highlight_bottom = min(frame.shape[0], bottom + pad)
                    highlight_left = max(0, left - pad)
                    highlight_right = min(frame.shape[1], right + pad)
                    
                    # Draw yellow box
                    cv2.rectangle(display_frame, (highlight_left, highlight_top), 
                                (highlight_right, highlight_bottom), (0, 255, 255), 2)
                    
                    # Add label
                    cv2.putText(display_frame, "Face to save", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw text input field
                display_frame = self.draw_text_input(display_frame)
                
                # Show the frame
                cv2.imshow('Face Recognition', display_frame)
                # --- End Solution Task 2.6 ---

                key = cv2.waitKey(1) & 0xFF

                # --- Solution Task 2.7: Handle Keyboard Input in Name Entry Mode ---
                # Handle keyboard input
                self.text_entry_active = True
                
                # Process key presses
                if key == 13:  # ENTER key
                    if self.current_text and self.selected_face_loc:
                        self.save_face(self.current_text, self.selected_face_loc)
                        self.state = "normal"
                        self.text_entry_active = False
                        self.current_text = ""
                        self.selected_face_loc = None
                
                elif key == 27:  # ESC key
                    self.state = "normal"
                    self.text_entry_active = False
                    self.current_text = ""
                    self.selected_face_loc = None
                
                elif key == 8:  # BACKSPACE key
                    if self.current_text:
                        self.current_text = self.current_text[:-1]
                
                elif key == ord('q'):
                    break
                
                elif 32 <= key <= 126:  # Printable characters
                    self.current_text += chr(key)
                # --- End Solution Task 2.7 ---

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_system = FaceRecognitionSystemTask2()
    face_system.run()