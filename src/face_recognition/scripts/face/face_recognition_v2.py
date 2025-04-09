import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
import face_recognition
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self):
        # Paths and directories
        self.faces_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "img")
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "models", "yolov8n.pt")
        
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
        
        # Face click handling
        self.button_area = []
        self.face_to_learn = None
        self.current_frame = None
        
        # UI states
        self.state = "normal"  # "normal" or "entering_name"
        self.current_text = ""
        self.selected_face_loc = None
        self.text_entry_active = False
        
        # Size of resized frame for face recognition
        self.recognition_size = (0, 0)
        
        print("Initialization complete. Press 'q' to quit.")

    def load_known_faces(self):
        """Load known faces from faces directory"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        for filename in os.listdir(self.faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Get the name from the filename (removing extension)
                name = os.path.splitext(filename)[0]
                
                # Load image and get face encoding
                image_path = os.path.join(self.faces_dir, filename)
                face_image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(face_image)
                
                if face_locations:
                    face_encoding = face_recognition.face_encodings(face_image, face_locations)[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
        
        print(f"Loaded {len(self.known_face_names)} known faces")

    def save_face(self, name, face_location):
        """Save a detected face with the provided name"""
        if not name:
            return
            
        # Calculate face coordinates with expansion for larger snapshot
        top, right, bottom, left = face_location
        
        # Expand the region by 30% for a bigger face snapshot
        height = bottom - top
        width = right - left
        
        # Calculate expanded coordinates
        expanded_top = max(0, top - int(height * 0.3))
        expanded_bottom = min(self.current_frame.shape[0], bottom + int(height * 0.3))
        expanded_left = max(0, left - int(width * 0.3))
        expanded_right = min(self.current_frame.shape[1], right + int(width * 0.3))
        
        # Extract face from the current frame with expanded region
        face_image = self.current_frame[expanded_top:expanded_bottom, expanded_left:expanded_right]
        
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}.jpg"
        filepath = os.path.join(self.faces_dir, filename)
        
        # If file already exists, don't overwrite but create a new one
        if os.path.exists(filepath):
            filepath = os.path.join(self.faces_dir, f"{name}_{timestamp}.jpg")
        
        cv2.imwrite(filepath, face_image)
        print(f"Saved face as {filepath}")
        
        # Reload known faces
        self.load_known_faces()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks on the "Learn Face" button"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state == "normal":
                # Check if a Learn Face button was clicked
                for (button_left, button_top, button_right, button_bottom), face_location in self.button_area:
                    if button_left <= x <= button_right and button_top <= y <= button_bottom:
                        # Button clicked, enter name input mode
                        self.state = "entering_name"
                        self.selected_face_loc = face_location
                        self.current_text = ""
                        self.text_entry_active = True
                        break
            elif self.state == "entering_name" and not self.text_entry_active:
                # If we're in name entry mode and clicked outside the text box
                # Reset to normal mode
                self.state = "normal"
                self.text_entry_active = False

    def detect_and_recognize_faces(self, frame):
        """Detect persons and faces with YOLO and recognize known faces"""
        # Make a copy of the frame for display and store the current frame
        display_frame = frame.copy()
        self.current_frame = frame.copy()
        
        # Resize frame for face recognition
        height, width = frame.shape[:2]
        self.recognition_size = (width, height)
        
        # Run YOLOv8 inference for person detection
        results = self.model(frame, classes=[0])  # Class 0 is person in COCO dataset
        
        # Process YOLOv8 results
        if results and len(results) > 0:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw person bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Detect faces in the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Clear previous button areas
        self.button_area = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Try to recognize the face
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
            
            # Draw a box around the face
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Draw a label with a name further below the face (more distance)
            label_top = bottom + 10  # Increased vertical distance from face
            label_bottom = bottom + 45
            cv2.rectangle(display_frame, (left, label_top), (right, label_bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(display_frame, name, (left + 6, label_top + 20), font, 0.7, (255, 255, 255), 1)
            
            # Add "Learn Face" button only for unknown faces
            if self.state == "normal" and not is_known_face:
                button_left = left
                button_top = top - 30
                button_right = right
                button_bottom = top
                
                if button_top > 0:  # Make sure button is within frame
                    cv2.rectangle(display_frame, (button_left, button_top), (button_right, button_bottom), (255, 0, 0), cv2.FILLED)
                    cv2.putText(display_frame, "Learn Face", (button_left + 5, button_top + 20), font, 0.5, (255, 255, 255), 1)
                    
                    # Store button area and associated face location
                    self.button_area.append(((button_left, button_top, button_right, button_bottom), (top, right, bottom, left)))
        
        return display_frame, face_locations

    def draw_text_input(self, frame):
        """Draw text input interface on the frame"""
        height, width = frame.shape[:2]
        
        # Create text input area at the bottom of the frame
        input_height = 40
        cv2.rectangle(frame, (0, height - input_height), (width, height), (50, 50, 50), cv2.FILLED)
        
        # Display current text and instruction
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Enter name: {self.current_text}_"
        cv2.putText(frame, text, (10, height - 15), font, 0.7, (255, 255, 255), 1)
        
        # Display instructions
        instructions = "Press ENTER to save, ESC to cancel"
        cv2.putText(frame, instructions, (width - 300, height - 15), font, 0.5, (200, 200, 200), 1)
        
        return frame

    def run(self):
        """Main loop for face recognition"""
        # Set up the mouse callback once, outside the loop
        cv2.namedWindow('Face Recognition')
        cv2.setMouseCallback('Face Recognition', self.mouse_callback)
        
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Mirror the frame (selfie view)
            frame = cv2.flip(frame, 1)
            
            if self.state == "normal":
                # Normal operation: detect and recognize faces
                display_frame, face_locations = self.detect_and_recognize_faces(frame)
                # Display the resulting frame
                cv2.imshow('Face Recognition', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
            elif self.state == "entering_name":
                # Text input mode
                # Show a simpler frame with the face highlighted
                display_frame = frame.copy()
                
                if self.selected_face_loc:
                    top, right, bottom, left = self.selected_face_loc
                    
                    # Calculate expanded coordinates for preview (same as in save_face)
                    height = bottom - top
                    width = right - left
                    
                    expanded_top = max(0, top - int(height * 0.3))
                    expanded_bottom = min(display_frame.shape[0], bottom + int(height * 0.3))
                    expanded_left = max(0, left - int(width * 0.3))
                    expanded_right = min(display_frame.shape[1], right + int(width * 0.3))
                    
                    # Draw the selected face with expanded highlight
                    cv2.rectangle(display_frame, 
                                 (expanded_left, expanded_top), 
                                 (expanded_right, expanded_bottom), 
                                 (255, 255, 0), 2)
                    
                    # Show what will be saved
                    cv2.putText(display_frame, "Face to save", 
                               (expanded_left, expanded_top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Draw the text input interface
                display_frame = self.draw_text_input(display_frame)
                
                # Display the frame
                cv2.imshow('Face Recognition', display_frame)
                
                # Handle keyboard input for text
                key = cv2.waitKey(1) & 0xFF
                
                # Process key input for text entry
                if key == 13:  # ENTER key - save the face with name
                    if self.current_text:
                        self.save_face(self.current_text, self.selected_face_loc)
                    # Reset to normal mode
                    self.state = "normal"
                elif key == 27:  # ESC key - cancel
                    self.state = "normal"
                elif key == 8:  # BACKSPACE - delete last character
                    self.current_text = self.current_text[:-1]
                elif key == ord('q'):  # q key - quit
                    break
                elif 32 <= key <= 126:  # Printable ASCII characters
                    self.current_text += chr(key)
        
        # Release the capture and destroy windows
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_system = FaceRecognitionSystem()
    face_system.run()