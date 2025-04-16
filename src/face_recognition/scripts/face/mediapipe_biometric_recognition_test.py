import os
import cv2
import numpy as np
import time
import mediapipe as mp
import pickle
from datetime import datetime


class FaceRecognitionSystem:
    def __init__(self):
        # Paths and directories
        base_dir = os.path.dirname(os.path.realpath(__file__))
        self.landmarks_dir = os.path.join(base_dir, "landmarks_data")

        # Create directory if it doesn't exist
        if not os.path.exists(self.landmarks_dir):
            os.makedirs(self.landmarks_dir)

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Camera could not be opened")

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Known face landmarks and names
        self.known_face_names = []
        self.known_face_landmarks_collection = []  # List of lists for multiple landmarks per person
        self.load_known_landmarks()

        # Thresholds for face recognition
        self.recognition_threshold = 0.15  # For confident recognition
        self.learning_threshold = 0.25     # Higher threshold for continuous learning
        self.max_samples_per_person = 30   # Erhöht von 10 auf 30 für bessere Lernfähigkeit
        
        # Feature weights for different facial regions (eyes are more important for recognition)
        self.feature_weights = self.generate_feature_weights()
        
        # Recognition enhancement
        self.recognition_history = {}  # Store recent recognitions for consistency check
        self.history_max_size = 5      # Maximum number of historical frames to consider
        self.consistency_threshold = 3  # Minimum consistent recognitions to confirm identity

        # Face-click handling
        self.button_area = []
        self.current_frame = None

        # UI state
        self.state = "normal"  # "normal" or "entering_name"
        self.current_text = ""
        self.selected_face_loc = None
        self.text_entry_active = False

        # Neue Parameter für optimiertes Lernen
        self.learning_cooldown = {}  # Dictionary um den letzten Lernzeitpunkt zu speichern
        self.min_learning_interval = 2.0  # Mindestens 2 Sekunden zwischen Landmark-Hinzufügungen
        self.base_diversity_threshold = 0.1  # Basiswert für Diversitätsschwelle
        
        print("Initialization complete. Press 'q' to exit.")

    def generate_feature_weights(self):
        """Generate weights for different facial features to improve recognition"""
        weights = np.ones(100)  # Default weight for all landmarks
        
        # Key landmark indices from extract_face_landmarks method
        # Eyes (more important for recognition)
        eye_indices = list(range(0, 12*2))  # 12 eye landmarks, each with x,y
        # Nose
        nose_indices = list(range(12*2, (12+9)*2))
        # Mouth
        mouth_indices = list(range((12+9)*2, (12+9+10)*2))
        # Eyebrows
        eyebrow_indices = list(range((12+9+10+2)*2, 100))
        
        # Apply weights: eyes most important, then eyebrows, nose, mouth
        for i in eye_indices:
            weights[i] = 2.5
        for i in eyebrow_indices:
            weights[i] = 1.8
        for i in nose_indices:
            weights[i] = 1.2
        for i in mouth_indices:
            weights[i] = 1.0
            
        return weights

    def extract_face_landmarks(self, image):
        """Extract face landmarks using MediaPipe"""
        # Convert image to RGB (MediaPipe requires RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipe processing
        results = self.face_mesh.process(image_rgb)

        face_landmarks_list = []
        face_locations_list = []

        if results.multi_face_landmarks:
            h, w, _ = image.shape

            for face_landmarks in results.multi_face_landmarks:
                # Calculate face location (bounding box)
                x_min, y_min = w, h
                x_max, y_max = 0, 0

                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Convert landmarks to a vector
                # We use a reduced number of landmarks for better comparability
                # These landmarks are key facial features
                key_landmarks_indices = [
                    # Eyes
                    33, 133, 160, 158, 153, 144,  # Right eye
                    362, 263, 385, 380, 387, 373,  # Left eye
                    # Nose
                    1, 2, 3, 4, 5, 6, 19, 94, 195,
                    # Mouth
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                    # Chin and cheeks
                    152, 377,
                    # Eyebrows
                    70, 63, 105, 66, 107,
                    336, 296, 334, 293, 300
                ]

                landmarks_array = []
                for idx in key_landmarks_indices:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        # Save x, y (normalized coordinates)
                        landmarks_array.extend([landmark.x, landmark.y])

                face_landmarks_list.append(np.array(landmarks_array))
                face_locations_list.append((y_min, x_max, y_max, x_min))  # top, right, bottom, left

        return face_landmarks_list, face_locations_list

    def load_known_landmarks(self):
        """Load saved landmarks"""
        self.known_face_landmarks_collection = []
        self.known_face_names = []

        landmarks_file = os.path.join(self.landmarks_dir, "face_landmarks.pkl")
        if os.path.exists(landmarks_file):
            try:
                with open(landmarks_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_landmarks_collection = data.get('landmarks_collection', [])
                    self.known_face_names = data.get('names', [])
                print(f"{len(self.known_face_names)} known persons with a total of {sum(len(landmarks) for landmarks in self.known_face_landmarks_collection)} landmark sets loaded")
            except Exception as e:
                print(f"Error loading landmarks: {e}")
                # Overwrite corrupt file
                self.save_landmarks_data()
        else:
            print("No saved landmarks found")

    def save_landmarks_data(self):
        """Save known landmarks"""
        landmarks_file = os.path.join(self.landmarks_dir, "face_landmarks.pkl")
        data = {
            'landmarks_collection': self.known_face_landmarks_collection,
            'names': self.known_face_names
        }
        try:
            with open(landmarks_file, 'wb') as f:
                pickle.dump(data, f)
            total_landmarks = sum(len(landmarks) for landmarks in self.known_face_landmarks_collection)
            print(f"Landmarks saved: {len(self.known_face_names)} persons with a total of {total_landmarks} landmark sets")
        except Exception as e:
            print(f"Error saving landmarks: {e}")

    def save_face(self, name, face_location):
        """Save face landmarks with the given name"""
        if not name:
            return False

        # Extract all faces in the entire frame
        all_landmarks, all_locations = self.extract_face_landmarks(self.current_frame)

        # Try to find the correct face based on position
        for landmarks, loc in zip(all_landmarks, all_locations):
            if self._locations_are_close(loc, face_location):
                if name in self.known_face_names:
                    # Person already exists, add new landmark set
                    index = self.known_face_names.index(name)
                    self.known_face_landmarks_collection[index].append(landmarks)
                    print(f"Landmarks added for existing name '{name}'")
                else:
                    # Create new person
                    self.known_face_landmarks_collection.append([landmarks])  # List with one landmark set
                    self.known_face_names.append(name)
                    print(f"New person '{name}' added")

                self.save_landmarks_data()
                return True

        print(f"WARNING: No matching face found for '{name}'")
        return False

    def _locations_are_close(self, loc1, loc2, tolerance=30):
        """Compare two face locations with a tolerance range"""
        return all(abs(a - b) < tolerance for a, b in zip(loc1, loc2))

    def compare_landmarks(self, landmarks):
        """Compare landmarks with known faces using weighted distance"""
        if not self.known_face_landmarks_collection or len(self.known_face_landmarks_collection) == 0:
            return "Unknown", False, 0

        if landmarks is None or len(landmarks) == 0:
            return "Unknown", False, 0

        # Calculate weighted Euclidean distance between landmarks
        min_distance = float('inf')
        best_match_index = -1
        best_match_landmark_index = -1
        all_distances = {}  # Store distances for each person for consistency analysis

        # For each known person
        for i, person_landmarks_list in enumerate(self.known_face_landmarks_collection):
            person_name = self.known_face_names[i]
            person_distances = []
            
            # For each landmark set of this person
            for j, known_landmarks in enumerate(person_landmarks_list):
                if len(known_landmarks) != len(landmarks):
                    print(f"Warning: Different landmark lengths - Known: {len(known_landmarks)}, Current: {len(landmarks)}")
                    continue

                # Calculate weighted Euclidean distance - give more importance to key features
                if len(self.feature_weights) == len(landmarks):
                    diff = (landmarks - known_landmarks) * self.feature_weights
                    distance = np.linalg.norm(diff)
                else:
                    # Fallback to standard Euclidean distance if weight dimensions don't match
                    distance = np.linalg.norm(landmarks - known_landmarks)
                
                person_distances.append(distance)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match_index = i
                    best_match_landmark_index = j

            # Store the minimum distance for this person
            if person_distances:
                all_distances[person_name] = min(person_distances)

        if best_match_index == -1:
            return "Unknown", False, 0

        best_match_name = self.known_face_names[best_match_index]
        
        # Confidence calculation based on distance
        confidence = 1.0 / (1.0 + min_distance)
        
        # Check recognition history for consistency
        face_id = self._get_face_spatial_id(landmarks)
        recognition_result = self._check_recognition_consistency(face_id, best_match_name, all_distances)
        
        # Return best match if consistent, otherwise return the highest confidence match
        if recognition_result:
            return recognition_result, True, confidence
        
        # Compare with threshold for direct recognition
        if min_distance < self.recognition_threshold:
            return best_match_name, True, confidence
        elif min_distance < self.learning_threshold:
            # Face recognized with lower confidence, but enough for continuous learning
            return best_match_name, True, confidence
        else:
            return "Unknown", False, confidence

    def _get_face_spatial_id(self, landmarks):
        """Generate a spatial ID for a face to track it across frames"""
        # Use the average position of key facial features as a simple spatial ID
        if len(landmarks) >= 10:  # Ensure we have some landmarks
            # Calculate mean for each coordinate (x and y separately)
            # landmarks is flat array with [x1, y1, x2, y2, ...], reshape to [[x1,y1], [x2,y2], ...]
            landmarks_pairs = landmarks[:10].reshape(-1, 2)
            mean_landmark = np.mean(landmarks_pairs, axis=0)
            spatial_id = hash(tuple(mean_landmark))
            return spatial_id
        return None
        
    def _check_recognition_consistency(self, face_id, current_match, all_distances):
        """Check if recognition is consistent across multiple frames"""
        if face_id is None:
            return None
            
        # Initialize history for this face if not exists
        if face_id not in self.recognition_history:
            self.recognition_history[face_id] = []
            
        # Add current recognition with all person distances
        self.recognition_history[face_id].append((current_match, all_distances))
        
        # Keep only recent history
        if len(self.recognition_history[face_id]) > self.history_max_size:
            self.recognition_history[face_id].pop(0)
            
        # Count occurrences of each name in history
        name_counts = {}
        for name, _ in self.recognition_history[face_id]:
            name_counts[name] = name_counts.get(name, 0) + 1
            
        # Find the most consistent name
        most_consistent_name = None
        max_count = 0
        
        for name, count in name_counts.items():
            if count > max_count:
                max_count = count
                most_consistent_name = name
                
        # Return the consistent name if it meets the threshold
        if max_count >= self.consistency_threshold:
            return most_consistent_name
            
        return None

    def add_landmark_to_person(self, name, landmarks):
        """Add a new landmark set to an already known person with improved diversity check and cooldown"""
        if name in self.known_face_names:
            person_index = self.known_face_names.index(name)
            landmarks_list = self.known_face_landmarks_collection[person_index]
            
            # Überprüfen, ob die Abklingzeit noch aktiv ist
            current_time = time.time()
            last_add_time = self.learning_cooldown.get(name, 0)
            time_since_last_add = current_time - last_add_time
            
            if time_since_last_add < self.min_learning_interval:
                return False  # Cooldown noch aktiv, keine neue Landmark hinzufügen
                
            # Berechne Diversitätsschwellenwert basierend auf bestehender Samplegröße
            # Je mehr Samples, desto höher muss die Diversität für neue Samples sein
            landmarks_count = len(landmarks_list)
            dynamic_threshold = self.base_diversity_threshold * (1.0 + (landmarks_count / self.max_samples_per_person))
            
            # Diversitätsbewertung berechnen
            diversity_score = self._calculate_landmark_diversity(landmarks, landmarks_list)
            
            # Überprüfe maximale Anzahl von Samples
            if len(landmarks_list) >= self.max_samples_per_person:
                # Maximum erreicht, ersetze nur wenn besonders divers
                high_diversity_threshold = dynamic_threshold * 2.0  # Höhere Schwelle für Ersetzungen
                
                if diversity_score > high_diversity_threshold:
                    least_diverse_idx = self._find_least_diverse_landmark(landmarks_list)
                    if least_diverse_idx is not None:
                        landmarks_list[least_diverse_idx] = landmarks
                        print(f"Replaced least diverse landmark for '{name}' (diversity: {diversity_score:.3f})")
                        self.learning_cooldown[name] = current_time  # Cooldown aktualisieren
                        self.save_landmarks_data()
                        return True
                return False
            
            # Hinzufügen nur wenn divers genug (höherer Wert = diverser)
            if diversity_score > dynamic_threshold:
                # Landmark hinzufügen und Daten speichern
                landmarks_list.append(landmarks)
                print(f"New landmark set added for '{name}' (now {len(landmarks_list)}, diversity: {diversity_score:.3f})")
                
                # Cooldown aktualisieren
                self.learning_cooldown[name] = current_time
                
                # Periodisch auf Ausreißer prüfen und bereinigen
                if len(landmarks_list) > 5 and len(landmarks_list) % 5 == 0:
                    self._clean_landmark_outliers(person_index)
                    
                self.save_landmarks_data()
                return True
        
        return False

    def _find_least_diverse_landmark(self, landmarks_list):
        """Findet den Landmarkensatz, der am wenigsten zur Diversität beiträgt"""
        if len(landmarks_list) <= 1:
            return None
            
        # Calculate average landmark
        avg_landmark = np.mean(landmarks_list, axis=0)
        
        # Calculate how much each landmark contributes to diversity
        distances = [np.linalg.norm(lm - avg_landmark) for lm in landmarks_list]
        
        # Landmarks close to average contribute less to diversity
        if distances:
            return np.argmin(distances)
        return None

    def _calculate_landmark_diversity(self, new_landmark, existing_landmarks):
        """Verbesserte Berechnung der Diversität einer neuen Landmark im Vergleich zu bestehenden"""
        if not existing_landmarks:
            return 1.0  # Maximale Diversität wenn keine bestehenden Landmarks
            
        # Durchschnitt aller bestehenden Landmarks berechnen
        avg_landmark = np.mean(existing_landmarks, axis=0)
        
        # Berechne durchschnittliche Distanz zwischen bestehenden Landmarks und ihrem Durchschnitt
        existing_distances = [np.linalg.norm(lm - avg_landmark) for lm in existing_landmarks]
        avg_existing_distance = np.mean(existing_distances) if existing_distances else 0
        
        # Berechne Distanz der neuen Landmark zum Durchschnitt
        new_distance = np.linalg.norm(new_landmark - avg_landmark)
        
        # Berechne auch Distanzen zu ALLEN bestehenden Landmarks, nicht nur zum Durchschnitt
        # Dies hilft bei der Erkennung von wirklich unterschiedlichen Posen/Ausdrücken
        individual_distances = [np.linalg.norm(new_landmark - lm) for lm in existing_landmarks]
        min_individual_distance = min(individual_distances) if individual_distances else 0
        
        # Kombinierte Diversitätsbewertung: Berücksichtigt sowohl Distanz zum Durchschnitt als auch
        # zum nächstgelegenen bestehenden Sample
        if avg_existing_distance > 0 and min_individual_distance > 0:
            avg_dist_factor = new_distance / avg_existing_distance
            # Wenn das Sample keinem bestehenden Sample sehr ähnlich ist, erhöhe die Diversitätsbewertung
            individual_factor = min_individual_distance / avg_existing_distance
            
            # Kombinierte Bewertung: Gewichteter Durchschnitt
            diversity = 0.7 * avg_dist_factor + 0.3 * individual_factor
            
            # Begrenzung auf sinnvollen Bereich
            return min(max(diversity, 0), 2.0)
        return 1.0

    def _clean_landmark_outliers(self, person_index):
        """Clean outliers from a person's landmark collection"""
        landmarks_list = self.known_face_landmarks_collection[person_index]
        if len(landmarks_list) <= 3:  # Need at least a few samples to determine outliers
            return
            
        # Calculate average landmark for this person
        avg_landmark = np.mean(landmarks_list, axis=0)
        
        # Calculate distances to average
        distances = [np.linalg.norm(lm - avg_landmark) for lm in landmarks_list]
        
        # Calculate mean and standard deviation of distances
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Identify outliers (more than 2.5 standard deviations from mean - erhöht von 2.0)
        if std_dist > 0:
            outlier_indices = [i for i, d in enumerate(distances) if d > mean_dist + 2.5*std_dist]
            
            # Maximal 2 Ausreißer pro Durchlauf entfernen
            outlier_indices = sorted(outlier_indices, reverse=True)[:2]
            
            # Remove outliers (in reverse order to not mess up indices)
            for i in outlier_indices:
                print(f"Removing outlier landmark for {self.known_face_names[person_index]}")
                landmarks_list.pop(i)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks on the 'Learn Face' button"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state == "normal":
                # Check if a Learn Face button was clicked
                for (button_left, button_top, button_right, button_bottom), face_location in self.button_area:
                    if button_left <= x <= button_right and button_top <= y <= button_bottom:
                        # Button clicked, activate name entry mode
                        self.state = "entering_name"
                        self.selected_face_loc = face_location
                        self.current_text = ""
                        self.text_entry_active = True
                        break
            elif self.state == "entering_name" and not self.text_entry_active:
                # If we are in name entry mode and click outside the text box
                # Return to normal mode
                self.state = "normal"
                self.text_entry_active = False

    def detect_and_recognize_faces(self, frame):
        """Detect faces using MediaPipe and recognize known faces"""
        # Create a copy of the frame for display and save the current frame
        display_frame = frame.copy()
        self.current_frame = frame.copy()

        # Detect face landmarks using MediaPipe
        face_landmarks_list, face_locations = self.extract_face_landmarks(frame)

        # Clear previous button areas
        self.button_area = []

        # Iterate through detected faces
        for i, (landmarks, face_loc) in enumerate(zip(face_landmarks_list, face_locations)):
            top, right, bottom, left = face_loc

            # Recognize face
            name, is_known_face, confidence = self.compare_landmarks(landmarks)

            # Continuous learning: Verbesserte Strategie für kontinuierliches Lernen
            # Nur lernen wenn die Erkennung relativ sicher, aber nicht perfekt ist
            if is_known_face:
                # Niedrige Konfidenz: Versuche zu lernen
                if 0.6 < confidence < 0.95:
                    self.add_landmark_to_person(name, landmarks)
                # Sehr niedrige Konfidenz: Nur gelegentlich lernen (reduziert falsche Zuordnungen)
                elif 0.4 < confidence <= 0.6:
                    # Nur in 30% der Fälle lernen, um Fehler zu reduzieren
                    if np.random.random() < 0.3:
                        self.add_landmark_to_person(name, landmarks)

            # Draw a rectangle around the face
            color = (0, 255, 0) if is_known_face else (0, 0, 255)  # Green for recognized, red for unknown
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)

            # Draw label with name below the face
            label_top = bottom + 10
            label_bottom = bottom + 35
            cv2.rectangle(display_frame, (left, label_top), (right, label_bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            conf_text = f"{confidence:.2f}" if confidence > 0 else "N/A"
            cv2.putText(display_frame, f"{name} ({conf_text})", (left + 6, label_top + 20), font, 0.5,
                        (255, 255, 255), 1)

            # Add "Learn Face" button only for unknown faces
            if self.state == "normal" and not is_known_face:
                button_left = left
                button_top = top - 30
                button_right = right
                button_bottom = top

                if button_top > 0:  # Ensure button is within the frame
                    cv2.rectangle(display_frame, (button_left, button_top), (button_right, button_bottom), (255, 0, 0),
                                  cv2.FILLED)
                    cv2.putText(display_frame, "Learn Face", (button_left + 5, button_top + 20), font, 0.5,
                                (255, 255, 255), 1)

                    # Save button area and associated face location
                    self.button_area.append(((button_left, button_top, button_right, button_bottom), face_loc))

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
        # Set up mouse callback once, outside the loop
        cv2.namedWindow('Face Recognition with MediaPipe')
        cv2.setMouseCallback('Face Recognition with MediaPipe', self.mouse_callback)

        while True:
            # Capture frame
            ret, frame = self.cap.read()

            if not ret:
                print("Error capturing frame")
                break

            # Flip frame (selfie view)
            frame = cv2.flip(frame, 1)

            if self.state == "normal":
                # Normal operation: detect and identify faces
                display_frame, face_locations_list = self.detect_and_recognize_faces(frame)
                # Display result frame
                cv2.imshow('Face Recognition with MediaPipe', display_frame)

                # Process keyboard inputs
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):  # New keyboard shortcut to clean outliers for all persons
                    print("Cleaning outlier landmarks for all persons...")
                    for i in range(len(self.known_face_names)):
                        self._clean_landmark_outliers(i)
                    self.save_landmarks_data()

            elif self.state == "entering_name":
                # Text entry mode
                # Display a simpler frame with highlighted face
                display_frame = frame.copy()

                if self.selected_face_loc:
                    top, right, bottom, left = self.selected_face_loc

                    # Highlight selected face
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (255, 255, 0), 2)

                    # Display what will be saved
                    cv2.putText(display_frame, "Face to be saved",
                                (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Draw text input interface
                display_frame = self.draw_text_input(display_frame)

                # Display frame
                cv2.imshow('Face Recognition with MediaPipe', display_frame)

                # Process key inputs for text entry
                key = cv2.waitKey(1) & 0xFF

                if key == 13:  # ENTER key - save face with name
                    if self.current_text:
                        success = self.save_face(self.current_text, self.selected_face_loc)
                        if not success:
                            print("Error saving face")
                    # Return to normal mode
                    self.state = "normal"
                elif key == 27:  # ESC key - cancel
                    self.state = "normal"
                elif key == 8:  # BACKSPACE - delete last character
                    self.current_text = self.current_text[:-1]
                elif key == ord('q'):  # q key - exit
                    break
                elif 32 <= key <= 126:  # Printable ASCII characters
                    self.current_text += chr(key)

        # Release camera and close windows
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_system = FaceRecognitionSystem()
    face_system.run()
