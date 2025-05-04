import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
import face_recognition
from datetime import datetime

# NOTE: This script builds on Task 1.
# Students should have completed Task 1 or be working with a solution.

class FaceRecognitionSystemTask2:
    def __init__(self):
        """
        Initializes the face recognition system for Task 2:
        - Loads YOLO model for face detection
        - Sets up face directory and camera
        - Loads known faces into memory
        """
        # Paths and directories
        self.faces_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "img")
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "yolov8n-face.pt")

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

        # --- TODO Task 2.1: Load Known Faces ---
        # Iterate through all files in the directory self.faces_dir
        # Filter for image files (ending with .jpg, .jpeg, .png)
        # For each image file:
        # 1. Extract the name from the filename (without extension)
        # 2. Load the image using face_recognition.load_image_file()
        # 3. Detect faces in the image using face_recognition.face_locations()
        # 4. If faces are found, encode them using face_recognition.face_encodings()
        # 5. Add the encoding and name to the lists
        
        # Example structure:
        # for filename in os.listdir(self.faces_dir):
        #     if filename.endswith(('.jpg', '.jpeg', '.png')):
        #         # Extract name, load image, etc.

        # --- End TODO Task 2.1 ---

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

        # --- TODO Task 2.2: Expand and Save Face Image ---
        # 1. Calculate the height and width of the face
        # 2. Expand the face region by 30% in all directions
        #    (ensure to stay within the image boundaries)
        # 3. Crop the expanded region from the current frame
        # 4. Generate a filename based on the given name
        # 5. If the file already exists, add a timestamp
        # 6. Save the image using cv2.imwrite()
        # 7. Reload known faces using self.load_known_faces()
        # --- End TODO Task 2.2 ---

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
            # --- TODO Task 2.3: Mouse Click Handling ---
            # Implement mouse click handling for two states:
            # 1. "normal" state: Check if the click is within a "Learn Face" button area
            #    - If yes, change to "entering_name" state
            #    - Store the associated face location
            #    - Reset the text buffer
            # 2. "entering_name" state: If not in active text entry mode,
            #    return to "normal" state
            # --- End TODO Task 2.3 ---
            pass  # Placeholder until implementation is added

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

        # YOLO detection (Assumption: Works after Task 1)
        results = self.model(frame, classes=[0])

        if results and len(results) > 0:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- TODO Task 2.4: Face Recognition and Identification ---
        # 1. Find face locations using face_recognition.face_locations()
        # 2. Encode the found faces using face_recognition.face_encodings()
        # 3. Reset the button_area list
        # 4. For each detected face:
        #    a. Initialize name as "Unknown" and is_known_face as False
        #    b. If known face encodings exist:
        #       - Compare the found face with known faces
        #         (face_recognition.compare_faces and face_recognition.face_distance)
        #       - If a match is found, update name and is_known_face
        #    c. Draw a bounding box and label for the face
        #    d. If the face is unknown and in "normal" state, add a "Learn Face" button
        #
        # This is just the minimum, so that the code works at all. It serves no other purpose.
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        # --- End TODO Task 2.4 ---

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

        # --- TODO Task 2.5: Draw Text Input Field ---
        # 1. Draw a filled rectangle at the bottom of the frame as background
        # 2. Display the current text with a cursor (_)
        # 3. Show instructions for ENTER (save) and ESC (cancel)
        # --- End TODO Task 2.5 ---

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

                # --- TODO Task 2.6: Display Name Entry UI ---
                # 1. If a face location is selected (self.selected_face_loc):
                #    a. Extract coordinates and expand the region
                #    b. Draw a yellow box around the face to be saved
                #    c. Label the face with "Face to save"
                # 2. Call self.draw_text_input() to draw the text input field
                # 3. Show the modified frame
                # --- End TODO Task 2.6 ---

                key = cv2.waitKey(1) & 0xFF

                # --- TODO Task 2.7: Handle Keyboard Input in Name Entry Mode ---
                # Handle the following keys:
                # - ENTER (13): If text has been entered, save face and return to normal mode
                # - ESC (27): Cancel and return to normal mode
                # - BACKSPACE (8): Delete last character
                # - 'q': Exit program
                # - Printable characters (32-126): Add to current text
                # --- End TODO Task 2.7 ---

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_system = FaceRecognitionSystemTask2()
    face_system.run()