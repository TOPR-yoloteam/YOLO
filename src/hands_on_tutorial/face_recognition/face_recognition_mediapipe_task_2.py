import os
import cv2
import numpy as np
import time
import mediapipe as mp
import pickle
from datetime import datetime

# NOTE: This code is based on the original complete script.
# Students should fill in the TODOs here after solving Task 1
# or after being provided with a solution for Task 1.

class FaceRecognitionSystem:
    def __init__(self):
        """
        Initializes the FaceRecognitionSystem.

        Sets up paths, MediaPipe, camera, loads known faces, and initializes
        parameters for recognition, learning, and UI.
        """
        # Paths and directories
        base_dir = os.path.dirname(os.path.realpath(__file__))
        self.landmarks_dir = os.path.join(base_dir, "landmarks_data")

        # Create directory if it doesn't exist
        if not os.path.exists(self.landmarks_dir):
            os.makedirs(self.landmarks_dir)

        # Initialize MediaPipe Face Mesh (Assumption: Works after Task 1)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, # Process video stream
            max_num_faces=10,       # Detect up to 10 faces
            min_detection_confidence=0.8, # Higher confidence for initial detection
            min_tracking_confidence=0.8  # Higher confidence for tracking across frames
        )

        # Initialize camera
        self.cap = cv2.VideoCapture(0) # 0 is usually the default webcam
        if not self.cap.isOpened():
            raise Exception("Camera could not be opened")

        # Set camera resolution (optional, but can improve performance)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Known face landmarks and names storage
        self.known_face_names = []
        self.known_face_landmarks_collection = [] # List of lists (one list per person)
        self.load_known_landmarks() # Load data from file if exists

        # Thresholds for face recognition logic
        self.recognition_threshold = 0.15 # Lower distance = more confident match
        self.learning_threshold = 0.25    # Threshold for considering adding landmarks
        self.max_samples_per_person = 30  # Max landmarks sets per known person

        # Feature weights (optional enhancement)
        self.feature_weights = self.generate_feature_weights()

        # Recognition enhancement using history (optional complexity)
        self.recognition_history = {}
        self.history_max_size = 5
        self.consistency_threshold = 3

        # Face-click handling variables for learning new faces
        self.button_area = [] # Stores clickable areas for "Learn Face"
        self.current_frame = None # Stores the latest frame for saving
        self.state = "normal" # State machine: "normal" or "entering_name"
        self.current_text = "" # Buffer for name input
        self.selected_face_loc = None # Bounding box of the face being learned
        self.text_entry_active = False # Flag for active text input

        # Learning parameters (optional complexity)
        self.learning_cooldown = {} # Prevents adding too many landmarks too quickly
        self.min_learning_interval = 2.0 # Seconds between landmark additions per person
        self.base_diversity_threshold = 0.1 # Minimum difference required to add landmarks

        print("Initialization complete (Task 2). Press 'q' to exit.")

    def generate_feature_weights(self):
        """
        Generates weights for different facial features (optional enhancement).

        Assigns higher weights to more discriminative features like eyes.

        Returns:
            np.ndarray: A NumPy array of weights corresponding to the landmark vector.
        """
        weights = np.ones(100) # Default weight for all landmarks (100 = 50 landmarks * 2 coordinates)
        # Key landmark indices (50 total selected landmarks)
        key_landmarks_indices = [
            33, 133, 160, 158, 153, 144, 362, 263, 385, 380, 387, 373, # Eyes (12)
            1, 2, 3, 4, 5, 6, 19, 94, 195, # Nose (9)
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, # Mouth (10)
            152, 377, # Chin/Cheeks (2)
            70, 63, 105, 66, 107, 336, 296, 334, 293, 300 # Eyebrows (10)
        ]
        # Example: Increase weight for eyes (first 12*2 = 24 elements in the final flat array)
        weights[:24] = 2.5 # Give eyes 2.5 times more importance
        # Can add more specific weights for other features here
        return weights

    def extract_face_landmarks(self, image):
        """
        Extracts face landmarks and bounding box locations using MediaPipe FaceMesh.

        Args:
            image (np.ndarray): The input image frame (BGR format from OpenCV).

        Returns:
            tuple: A tuple containing:
                - list: A list of NumPy arrays, each representing the key landmarks of a detected face.
                - list: A list of tuples, each representing the bounding box (top, right, bottom, left) of a detected face.
        """
        # Convert BGR image to RGB, required by MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Mark image as not writeable to pass by reference (performance)
        image_rgb.flags.writeable = False
        # Process the image and find face landmarks
        results = self.face_mesh.process(image_rgb) # Assumption: Works after Task 1
        # Mark image as writeable again
        image_rgb.flags.writeable = True

        face_landmarks_list = []
        face_locations_list = []

        # Check if any faces were detected
        if results and results.multi_face_landmarks:
            h, w, _ = image.shape # Get image dimensions for coordinate conversion
            # Indices for key landmarks (eyes, nose, mouth, etc.) - 50 landmarks
            key_landmarks_indices = [
                # Eyes (12 landmarks)
                33, 133, 160, 158, 153, 144,  # Right eye
                362, 263, 385, 380, 387, 373,  # Left eye
                # Nose (9 landmarks)
                1, 2, 3, 4, 5, 6, 19, 94, 195,
                # Mouth (10 landmarks)
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                # Chin and cheeks (2 landmarks)
                152, 377,
                # Eyebrows (10 landmarks)
                70, 63, 105, 66, 107,
                336, 296, 334, 293, 300
            ] # Total 50 landmarks

            # Process each detected face
            for face_landmarks in results.multi_face_landmarks:
                # --- Calculate face location (bounding box) - same as Task 1 ---
                x_min, y_min = w, h
                x_max, y_max = 0, 0
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

                # --- TODO 1: Extract Specific Landmarks ---
                # Create a NumPy array ('landmarks_array') containing the x and y coordinates
                # of the landmarks defined in 'key_landmarks_indices'.
                # The coordinates should be normalized (directly from landmark.x, landmark.y).
                # The resulting array should have 100 elements (50 landmarks * 2 coordinates).
                # Iterate through 'key_landmarks_indices', access the corresponding
                # landmark in 'face_landmarks.landmark', and add its x and y to the array.

                # Convert the list to a NumPy array

                # --- End TODO 1 ---

                # Append the extracted landmark vector for this face
                face_landmarks_list.append(landmarks_array)

        return face_landmarks_list, face_locations_list

    def load_known_landmarks(self):
        """
        Loads known face landmarks and names from a pickle file.
        """
        self.known_face_landmarks_collection = []
        self.known_face_names = []
        landmarks_file = os.path.join(self.landmarks_dir, "face_landmarks.pkl")
        if os.path.exists(landmarks_file):
            try:
                with open(landmarks_file, 'rb') as f:
                    # Load the dictionary from the pickle file
                    data = pickle.load(f)
                    # Get landmarks and names, provide empty lists as default
                    self.known_face_landmarks_collection = data.get('landmarks_collection', [])
                    self.known_face_names = data.get('names', [])
                print(f"{len(self.known_face_names)} known persons loaded.")
            except Exception as e:
                # Handle potential errors during file loading (e.g., corrupted file)
                print(f"Error loading landmarks: {e}. Creating a new file.")
                self.save_landmarks_data() # Create a fresh, empty file
        else:
            print("No saved landmarks found. A new file will be created.")

    def save_landmarks_data(self):
        """
        Saves the current known face landmarks and names to a pickle file.
        """
        landmarks_file = os.path.join(self.landmarks_dir, "face_landmarks.pkl")
        # Prepare data in a dictionary
        data = {
            'landmarks_collection': self.known_face_landmarks_collection,
            'names': self.known_face_names
        }
        try:
            with open(landmarks_file, 'wb') as f:
                # Dump the dictionary into the file using pickle
                pickle.dump(data, f)
            # Optional: Print confirmation
            total_landmarks = sum(len(landmarks) for landmarks in self.known_face_landmarks_collection)
            print(f"Landmarks saved for {len(self.known_face_names)} persons with {total_landmarks} total samples.")
        except Exception as e:
            print(f"Error saving landmarks: {e}")

    def save_face(self, name, face_location):
        """
        Saves the landmarks of the face at the specified location with the given name.

        Args:
            name (str): The name to associate with the face.
            face_location (tuple): The bounding box (top, right, bottom, left) of the face to save.

        Returns:
            bool: True if the face was successfully saved, False otherwise.
        """
        if not name: return False # Don't save if name is empty

        # Re-extract landmarks from the *current* frame to ensure we get the correct face
        all_landmarks, all_locations = self.extract_face_landmarks(self.current_frame)

        # Find the landmarks corresponding to the clicked face location
        for landmarks, loc in zip(all_landmarks, all_locations):
            # Check if the detected location is close to the clicked location
            if self._locations_are_close(loc, face_location):
                # Check if the person is already known
                if name in self.known_face_names:
                    # Add new landmark sample to existing person
                    index = self.known_face_names.index(name)
                    self.known_face_landmarks_collection[index].append(landmarks)
                    print(f"Landmarks added for existing name '{name}'.")
                else:
                    # Add a new person
                    self.known_face_landmarks_collection.append([landmarks]) # Start new list with this sample
                    self.known_face_names.append(name)
                    print(f"New person '{name}' added.")

                # Save the updated data to the file
                self.save_landmarks_data()
                return True # Indicate success

        # If no matching face was found at the specified location
        print(f"WARNING: No matching face found for '{name}' at save location.")
        return False

    def _locations_are_close(self, loc1, loc2, tolerance=30):
        """
        Compares two face bounding boxes to see if they are close enough.

        Args:
            loc1 (tuple): First bounding box (top, right, bottom, left).
            loc2 (tuple): Second bounding box (top, right, bottom, left).
            tolerance (int): Maximum pixel difference allowed for coordinates.

        Returns:
            bool: True if locations are considered close, False otherwise.
        """
        # Check if the absolute difference for each coordinate is within tolerance
        return all(abs(a - b) < tolerance for a, b in zip(loc1, loc2))

    def compare_landmarks(self, landmarks):
        """
        Compares detected landmarks with known face landmarks to find the best match.

        Args:
            landmarks (np.ndarray): The landmark vector of the detected face.

        Returns:
            tuple: A tuple containing:
                - str: The name of the best matching known face ("Unknown" if no match).
                - bool: True if a known face was recognized, False otherwise.
                - float: A confidence score (higher means better match, based on distance).
        """
        # Check if there are any known faces or if input landmarks are valid
        if not self.known_face_landmarks_collection or landmarks is None or len(landmarks) == 0:
            return "Unknown", False, 0

        min_distance = float('inf') # Initialize with a very large distance
        best_match_index = -1       # Index of the best matching person

        # Iterate through each known person
        for i, person_landmarks_list in enumerate(self.known_face_landmarks_collection):
            # Iterate through each saved landmark sample for the current person
            for known_landmarks in person_landmarks_list:
                # Basic check: ensure landmark vectors have the same length
                if len(known_landmarks) != len(landmarks):
                    # This might happen if key_landmarks_indices changes or data is corrupt
                    # print(f"Warning: Landmark length mismatch. Known: {len(known_landmarks)}, Current: {len(landmarks)}")
                    continue # Skip this comparison

                # --- TODO 2: Calculate Distance ---

                # --- End TODO 2 ---

                # Check if this distance is the smallest found so far
                if distance < min_distance:
                    min_distance = distance
                    best_match_index = i # Store the index of this person

        # If no match was found (best_match_index remains -1)
        if best_match_index == -1:
            return "Unknown", False, 0

        # Get the name of the best matching person
        best_match_name = self.known_face_names[best_match_index]
        # Calculate a simple confidence score (inversely related to distance)
        confidence = max(0.0, 1.0 - (min_distance / (self.recognition_threshold * 1.5))) # Normalize roughly

        # --- TODO 3: Apply Thresholds ---
        # Compare 'min_distance' with 'self.recognition_threshold'.
        # If the distance is smaller, the face is recognized ('is_known_face' = True).
        # Return 'best_match_name', 'is_known_face', and 'confidence'.
        # If the distance is greater or equal, the face is unknown ('is_known_face' = False).
        # Return "Unknown", 'is_known_face', and 'confidence'.
        # Optional: Also consider 'self.learning_threshold' for continuous learning decisions later.

        # --- End TODO 3 ---


    # --- The following methods handle continuous learning, UI, etc. ---
    # Students don't need to modify these directly for Task 2's core goals,
    # but they are necessary for the "Learn Face" functionality.

    def add_landmark_to_person(self, name, landmarks):
        """
        Adds a new landmark set to an already known person, checking for diversity and cooldown.

        Args:
            name (str): The name of the known person.
            landmarks (np.ndarray): The new landmark vector to potentially add.

        Returns:
            bool: True if the landmark was added, False otherwise.
        """
        # (Code remains largely unchanged from the German version)
        if name in self.known_face_names:
            person_index = self.known_face_names.index(name)
            landmarks_list = self.known_face_landmarks_collection[person_index]

            # Check learning cooldown
            current_time = time.time()
            last_add_time = self.learning_cooldown.get(name, 0)
            if current_time - last_add_time < self.min_learning_interval:
                return False # Still in cooldown

            # Check maximum samples
            if len(landmarks_list) >= self.max_samples_per_person:
                # print(f"Max samples reached for '{name}'.")
                return False # Max samples reached

            # Check diversity
            landmarks_count = len(landmarks_list)
            dynamic_threshold = self.base_diversity_threshold * (1.0 + (landmarks_count / self.max_samples_per_person))
            diversity_score = self._calculate_landmark_diversity(landmarks, landmarks_list)

            if diversity_score > dynamic_threshold:
                landmarks_list.append(landmarks)
                print(f"New diverse landmark set added for '{name}' (now {len(landmarks_list)}, diversity: {diversity_score:.3f})")
                self.learning_cooldown[name] = current_time # Update cooldown time
                # Periodically clean outliers
                if len(landmarks_list) > 5 and len(landmarks_list) % 5 == 0:
                    self._clean_landmark_outliers(person_index)
                self.save_landmarks_data() # Save after adding
                return True
        return False

    def _calculate_landmark_diversity(self, new_landmark, existing_landmarks):
        """
        Calculates a score representing how different a new landmark set is from existing ones.

        Args:
            new_landmark (np.ndarray): The candidate landmark vector.
            existing_landmarks (list): List of existing landmark vectors for the person.

        Returns:
            float: A diversity score (higher means more diverse).
        """
        # (Code remains largely unchanged)
        if not existing_landmarks: return 1.0 # Max diversity if it's the first sample
        avg_landmark = np.mean(existing_landmarks, axis=0)
        existing_distances = [np.linalg.norm(lm - avg_landmark) for lm in existing_landmarks]
        avg_existing_distance = np.mean(existing_distances) if existing_distances else 0
        new_distance_to_avg = np.linalg.norm(new_landmark - avg_landmark)
        individual_distances = [np.linalg.norm(new_landmark - lm) for lm in existing_landmarks]
        min_individual_distance = min(individual_distances) if individual_distances else 0

        # Calculate diversity based on distance to average and distance to nearest sample
        if avg_existing_distance > 1e-6 and min_individual_distance > 1e-6: # Avoid division by zero
            avg_dist_factor = new_distance_to_avg / avg_existing_distance
            individual_factor = min_individual_distance / avg_existing_distance
            # Combine factors (weights can be tuned)
            diversity = 0.7 * avg_dist_factor + 0.3 * individual_factor
            return min(max(diversity, 0), 2.0) # Clamp score
        elif new_distance_to_avg > 1e-6:
             return 1.0 # If only distance to avg is meaningful
        else:
             return 0.0 # If it's identical to the average

    def _clean_landmark_outliers(self, person_index):
        """
        Removes outlier landmark sets from a person's collection based on distance from the average.

        Args:
            person_index (int): The index of the person in the known lists.
        """
        # (Code remains largely unchanged)
        landmarks_list = self.known_face_landmarks_collection[person_index]
        if len(landmarks_list) <= 5: return # Need enough samples to identify outliers

        avg_landmark = np.mean(landmarks_list, axis=0)
        distances = [np.linalg.norm(lm - avg_landmark) for lm in landmarks_list]
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        if std_dist > 1e-6: # Avoid cleaning if standard deviation is near zero
            # Identify outliers (e.g., more than 2.5 standard deviations from the mean distance)
            outlier_indices = [i for i, d in enumerate(distances) if d > mean_dist + 2.5 * std_dist]

            if outlier_indices:
                 # Remove outliers (in reverse order to maintain indices)
                outlier_indices = sorted(outlier_indices, reverse=True)
                for i in outlier_indices:
                    print(f"Removing outlier landmark sample {i} for {self.known_face_names[person_index]}. Distance: {distances[i]:.3f}")
                    landmarks_list.pop(i)
                # No need to save here, saving happens when new landmarks are added


    def mouse_callback(self, event, x, y, flags, param):
        """
        Handles mouse click events, primarily for the 'Learn Face' button.

        Args:
            event: The type of mouse event (e.g., cv2.EVENT_LBUTTONDOWN).
            x (int): The x-coordinate of the mouse click.
            y (int): The y-coordinate of the mouse click.
            flags: Additional flags associated with the event.
            param: User-defined parameters passed to the callback.
        """
        # (Code remains largely unchanged)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state == "normal":
                # Check if the click was inside any "Learn Face" button area
                for (button_left, button_top, button_right, button_bottom), face_location in self.button_area:
                    if button_left <= x <= button_right and button_top <= y <= button_bottom:
                        # Button clicked, switch to name entry mode
                        print("Learn Face button clicked. Enter name.")
                        self.state = "entering_name"
                        self.selected_face_loc = face_location # Store which face was clicked
                        self.current_text = "" # Clear text buffer
                        self.text_entry_active = True
                        break # Stop checking other buttons
            elif self.state == "entering_name" and not self.text_entry_active:
                 # Clicking outside the (future) text box area while in entry mode cancels it
                 print("Name entry cancelled.")
                 self.state = "normal"
                 self.text_entry_active = False

    def detect_and_recognize_faces(self, frame):
        """
        Performs the main detection and recognition loop for a single frame.

        Detects faces, extracts landmarks, compares them to known faces,
        and draws bounding boxes and labels on the frame.

        Args:
            frame (np.ndarray): The input video frame (BGR).

        Returns:
            np.ndarray: The frame with drawn annotations (bounding boxes, labels).
        """
        # (Code uses results from extract_face_landmarks and compare_landmarks)
        display_frame = frame.copy() # Work on a copy to avoid modifying the original
        self.current_frame = frame.copy() # Keep a copy for saving faces
        # Get landmarks and locations for all detected faces
        face_landmarks_list, face_locations = self.extract_face_landmarks(frame)
        self.button_area = [] # Reset button areas for this frame

        # Process each detected face
        for i, (landmarks, face_loc) in enumerate(zip(face_landmarks_list, face_locations)):
            top, right, bottom, left = face_loc # Unpack bounding box coordinates

            # Compare landmarks to known faces (Calls the function with TODOs 2 & 3)
            name, is_known_face, confidence = self.compare_landmarks(landmarks)

            # --- Continuous Learning Logic (Optional Enhancement) ---
            # If a known face is detected with reasonable confidence, but not perfectly,
            # consider adding its landmarks to improve the model.
            # Check if the distance was calculated and available (needed for learning_threshold)
            # This requires modifying compare_landmarks to return min_distance or handling it here.
            # Example (assuming min_distance is accessible):
            # if is_known_face and self.recognition_threshold <= min_distance < self.learning_threshold:
            #    self.add_landmark_to_person(name, landmarks)
            # Simplified version without direct distance access:
            if is_known_face and confidence > 0.5 and confidence < 0.9: # Learn uncertain but likely matches
                 self.add_landmark_to_person(name, landmarks)


            # --- Draw Bounding Box ---
            color = (0, 255, 0) if is_known_face else (0, 0, 255) # Green if known, Red if unknown
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2) # Draw the box

            # --- Draw Label ---
            label_top = bottom + 10 # Position label below the box
            label_bottom = bottom + 35
            # Draw a filled rectangle for the label background
            cv2.rectangle(display_frame, (left, label_top), (right, label_bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            # Format confidence text
            conf_text = f"{confidence:.2f}" if confidence > 0 else "N/A"
            # Put name and confidence text on the label background
            cv2.putText(display_frame, f"{name} ({conf_text})", (left + 6, label_top + 20), font, 0.5, (255, 255, 255), 1) # White text

            # --- Add "Learn Face" Button for Unknown Faces ---
            if self.state == "normal" and not is_known_face:
                # Position button above the bounding box
                button_left, button_top = left, top - 30
                button_right, button_bottom = right, top
                # Ensure button is within frame bounds
                if button_top > 0:
                    # Draw button rectangle
                    cv2.rectangle(display_frame, (button_left, button_top), (button_right, button_bottom), (255, 0, 0), cv2.FILLED) # Blue button
                    # Draw button text
                    cv2.putText(display_frame, "Learn Face", (button_left + 5, button_top + 20), font, 0.5, (255, 255, 255), 1) # White text
                    # Store button area and the associated face location for the mouse callback
                    self.button_area.append(((button_left, button_top, button_right, button_bottom), face_loc))

        return display_frame # Return the annotated frame

    def draw_text_input(self, frame):
        """
        Draws the text input interface at the bottom of the frame when learning a new face.

        Args:
            frame (np.ndarray): The frame to draw on.

        Returns:
            np.ndarray: The frame with the text input interface drawn.
        """
        # (Code remains largely unchanged)
        height, width = frame.shape[:2]
        input_height = 40
        # Draw a dark background rectangle at the bottom
        cv2.rectangle(frame, (0, height - input_height), (width, height), (50, 50, 50), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Display the text being entered with a blinking cursor simulation
        text = f"Enter name: {self.current_text}"
        # Add underscore toggling based on time for a simple cursor effect
        if int(time.time() * 2) % 2 == 0:
             text += "_"
        cv2.putText(frame, text, (10, height - 15), font, 0.7, (255, 255, 255), 1) # White text
        # Display instructions
        instructions = "Press ENTER to save, ESC to cancel"
        cv2.putText(frame, instructions, (width - 300, height - 15), font, 0.5, (200, 200, 200), 1) # Gray text
        return frame

    def run(self):
        """
        Runs the main loop for the face recognition system.

        Continuously captures frames, processes them, handles UI state,
        and responds to keyboard/mouse inputs.
        """
        # (Code structure remains the same, controls state machine and main loop)
        window_title = 'Face Recognition with MediaPipe (Task 2)'
        cv2.namedWindow(window_title)
        # Set the mouse callback function for the window
        cv2.setMouseCallback(window_title, self.mouse_callback)

        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break # Exit loop if frame capture fails

            # Flip the frame horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)

            # --- State Machine Logic ---
            if self.state == "normal":
                # Perform detection and recognition
                display_frame = self.detect_and_recognize_faces(frame)
                # Display the resulting frame
                cv2.imshow(window_title, display_frame)

                # --- Handle Keyboard Input in Normal Mode ---
                key = cv2.waitKey(1) & 0xFF # Wait for a key press (1ms delay)
                if key == ord('q'):
                    print("Exiting...")
                    break # Exit loop if 'q' is pressed
                elif key == ord('c'): # Shortcut to clean outliers
                    print("Cleaning outlier landmarks for all known persons...")
                    for i in range(len(self.known_face_names)):
                        self._clean_landmark_outliers(i)
                    self.save_landmarks_data() # Save after cleaning
                    print("Outlier cleaning complete.")

            elif self.state == "entering_name":
                # --- Handle Name Entry Mode ---
                display_frame = frame.copy() # Start with the raw frame
                # Highlight the selected face
                if self.selected_face_loc:
                    top, right, bottom, left = self.selected_face_loc
                    # Draw a yellow rectangle around the face being learned
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 255), 2) # Yellow box
                    cv2.putText(display_frame, "Face to be saved", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) # Yellow text

                # Draw the text input bar at the bottom
                display_frame = self.draw_text_input(display_frame)
                # Display the frame
                cv2.imshow(window_title, display_frame)

                # --- Handle Keyboard Input in Entry Mode ---
                key = cv2.waitKey(1) & 0xFF
                if key == 13: # ENTER key
                    if self.current_text: # Only save if a name was entered
                        print(f"Attempting to save face as '{self.current_text}'...")
                        self.save_face(self.current_text, self.selected_face_loc)
                    else:
                        print("Name entry cancelled (no name entered).")
                    # Return to normal mode regardless of save success
                    self.state = "normal"
                    self.text_entry_active = False
                elif key == 27: # ESC key
                    print("Name entry cancelled.")
                    self.state = "normal"
                    self.text_entry_active = False
                elif key == 8: # BACKSPACE key
                    # Remove the last character from the text buffer
                    self.current_text = self.current_text[:-1]
                elif key == ord('q'):
                     print("Exiting...")
                     break # Allow quitting from entry mode too
                elif 32 <= key <= 126: # Printable ASCII characters
                    # Append the character to the text buffer
                    self.current_text += chr(key)

        # --- Cleanup ---
        # Release the camera resource
        self.cap.release()
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print("Resources released.")

# Entry point of the script
if __name__ == "__main__":
    # Create an instance of the system
    face_system = FaceRecognitionSystem()
    # Run the main loop
    face_system.run()
