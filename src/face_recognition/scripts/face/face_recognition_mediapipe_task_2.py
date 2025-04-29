import os
import cv2
import numpy as np
import time
import mediapipe as mp
import pickle
from datetime import datetime

# HINWEIS: Dieser Code basiert auf dem ursprünglichen vollständigen Skript.
# Die Studierenden sollen die TODOs hier ausfüllen, nachdem sie Aufgabe 1 gelöst haben
# oder nachdem ihnen eine Lösung für Aufgabe 1 bereitgestellt wurde.

class FaceRecognitionSystem:
    def __init__(self):
        # Paths and directories
        base_dir = os.path.dirname(os.path.realpath(__file__))
        self.landmarks_dir = os.path.join(base_dir, "landmarks_data")

        # Create directory if it doesn't exist
        if not os.path.exists(self.landmarks_dir):
            os.makedirs(self.landmarks_dir)

        # Initialize MediaPipe Face Mesh (Annahme: Funktioniert nach Aufgabe 1)
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
        self.known_face_landmarks_collection = []
        self.load_known_landmarks()

        # Thresholds for face recognition
        self.recognition_threshold = 0.15
        self.learning_threshold = 0.25
        self.max_samples_per_person = 30

        # Feature weights (optional, can be ignored for simplicity initially)
        self.feature_weights = self.generate_feature_weights()

        # Recognition enhancement (optional complexity)
        self.recognition_history = {}
        self.history_max_size = 5
        self.consistency_threshold = 3

        # Face-click handling for learning new faces
        self.button_area = []
        self.current_frame = None
        self.state = "normal"
        self.current_text = ""
        self.selected_face_loc = None
        self.text_entry_active = False

        # Learning parameters (optional complexity)
        self.learning_cooldown = {}
        self.min_learning_interval = 2.0
        self.base_diversity_threshold = 0.1

        print("Initialization complete (Task 2). Press 'q' to exit.")

    def generate_feature_weights(self):
        """Generate weights for different facial features (optional)"""
        weights = np.ones(100) # Default weight for all landmarks (100 = 50 landmarks * 2 coordinates)
        # Key landmark indices (50 total)
        key_landmarks_indices = [
            33, 133, 160, 158, 153, 144, 362, 263, 385, 380, 387, 373, # Eyes (12)
            1, 2, 3, 4, 5, 6, 19, 94, 195, # Nose (9)
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, # Mouth (10)
            152, 377, # Chin/Cheeks (2)
            70, 63, 105, 66, 107, 336, 296, 334, 293, 300 # Eyebrows (10)
        ]
        # Example: Increase weight for eyes (first 12*2 = 24 elements in the final array)
        weights[:24] = 2.5
        return weights

    def extract_face_landmarks(self, image):
        """Extract face landmarks and locations using MediaPipe"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb) # Annahme: Funktioniert nach Aufgabe 1
        image_rgb.flags.writeable = True

        face_landmarks_list = []
        face_locations_list = []

        if results and results.multi_face_landmarks:
            h, w, _ = image.shape
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

            for face_landmarks in results.multi_face_landmarks:
                # Calculate face location (bounding box) - same as Task 1
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                face_locations_list.append((y_min, x_max, y_max, x_min))

                # --- TODO 1: Spezifische Landmarken extrahieren ---
                # Erstellen Sie ein NumPy-Array ('landmarks_array'), das die x- und y-Koordinaten
                # der Landmarken enthält, die in 'key_landmarks_indices' definiert sind.
                # Die Koordinaten sollten normalisiert sein (direkt aus landmark.x, landmark.y).
                # Das resultierende Array sollte 100 Elemente haben (50 Landmarken * 2 Koordinaten).
                # Iterieren Sie durch 'key_landmarks_indices', greifen Sie auf die entsprechende
                # Landmarke in 'face_landmarks.landmark' zu und fügen Sie x und y zum Array hinzu.
                # Beispiel:
                # landmarks_array = []
                # for idx in key_landmarks_indices:
                #     if idx < len(face_landmarks.landmark):
                #         landmark = face_landmarks.landmark[idx]
                #         landmarks_array.extend([landmark.x, landmark.y])
                # face_landmarks_list.append(np.array(landmarks_array))

                landmarks_array = np.zeros(100) # Platzhalter - Muss ersetzt werden!
                # --- Ende TODO 1 ---
                face_landmarks_list.append(landmarks_array) # Fügen Sie das erstellte Array hinzu

        return face_landmarks_list, face_locations_list

    def load_known_landmarks(self):
        """Load saved landmarks"""
        # (Code bleibt unverändert)
        self.known_face_landmarks_collection = []
        self.known_face_names = []
        landmarks_file = os.path.join(self.landmarks_dir, "face_landmarks.pkl")
        if os.path.exists(landmarks_file):
            try:
                with open(landmarks_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_landmarks_collection = data.get('landmarks_collection', [])
                    self.known_face_names = data.get('names', [])
                print(f"{len(self.known_face_names)} known persons loaded.")
            except Exception as e:
                print(f"Error loading landmarks: {e}")
                self.save_landmarks_data()
        else:
            print("No saved landmarks found")

    def save_landmarks_data(self):
        """Save known landmarks"""
        # (Code bleibt unverändert)
        landmarks_file = os.path.join(self.landmarks_dir, "face_landmarks.pkl")
        data = {
            'landmarks_collection': self.known_face_landmarks_collection,
            'names': self.known_face_names
        }
        try:
            with open(landmarks_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Landmarks saved for {len(self.known_face_names)} persons.")
        except Exception as e:
            print(f"Error saving landmarks: {e}")

    def save_face(self, name, face_location):
        """Save face landmarks with the given name"""
        # (Code bleibt unverändert, nutzt extract_face_landmarks)
        if not name: return False
        all_landmarks, all_locations = self.extract_face_landmarks(self.current_frame)
        for landmarks, loc in zip(all_landmarks, all_locations):
            if self._locations_are_close(loc, face_location):
                if name in self.known_face_names:
                    index = self.known_face_names.index(name)
                    self.known_face_landmarks_collection[index].append(landmarks)
                    print(f"Landmarks added for existing name '{name}'")
                else:
                    self.known_face_landmarks_collection.append([landmarks])
                    self.known_face_names.append(name)
                    print(f"New person '{name}' added")
                self.save_landmarks_data()
                return True
        print(f"WARNING: No matching face found for '{name}' at save")
        return False

    def _locations_are_close(self, loc1, loc2, tolerance=30):
        """Compare two face locations"""
        # (Code bleibt unverändert)
        return all(abs(a - b) < tolerance for a, b in zip(loc1, loc2))

    def compare_landmarks(self, landmarks):
        """Compare detected landmarks with known faces."""
        if not self.known_face_landmarks_collection or landmarks is None or len(landmarks) == 0:
            return "Unknown", False, 0

        min_distance = float('inf')
        best_match_index = -1

        # Iterate through each known person
        for i, person_landmarks_list in enumerate(self.known_face_landmarks_collection):
            # Iterate through each saved landmark set for that person
            for known_landmarks in person_landmarks_list:
                if len(known_landmarks) != len(landmarks):
                    # print(f"Warning: Landmark length mismatch. Known: {len(known_landmarks)}, Current: {len(landmarks)}")
                    continue # Skip if lengths don't match

                # --- TODO 2: Distanz berechnen ---
                # Berechnen Sie die euklidische Distanz zwischen den aktuellen Landmarken ('landmarks')
                # und den gespeicherten Landmarken ('known_landmarks').
                # Verwenden Sie 'np.linalg.norm'.
                # Optional: Berücksichtigen Sie die Gewichtung 'self.feature_weights'.
                # distance = np.linalg.norm(landmarks - known_landmarks) # Einfache Version
                # oder gewichtet:
                # diff = (landmarks - known_landmarks) * self.feature_weights
                # distance = np.linalg.norm(diff)
                distance = float('inf') # Platzhalter - Muss ersetzt werden!
                # --- Ende TODO 2 ---

                # Update minimum distance found so far
                if distance < min_distance:
                    min_distance = distance
                    best_match_index = i

        # If no match was found (e.g., due to errors or no known faces)
        if best_match_index == -1:
            return "Unknown", False, 0

        best_match_name = self.known_face_names[best_match_index]
        confidence = 1.0 / (1.0 + min_distance) # Simple confidence measure

        # --- TODO 3: Schwellenwerte anwenden ---
        # Vergleichen Sie 'min_distance' mit 'self.recognition_threshold'.
        # Wenn die Distanz kleiner ist, wurde das Gesicht erkannt ('is_known_face' = True).
        # Geben Sie den 'best_match_name', 'is_known_face' und 'confidence' zurück.
        # Wenn die Distanz größer oder gleich ist, ist das Gesicht unbekannt ('is_known_face' = False).
        # Geben Sie "Unknown", 'is_known_face' und 'confidence' zurück.
        # Optional: Berücksichtigen Sie auch 'self.learning_threshold' für kontinuierliches Lernen.
        is_known_face = False # Platzhalter
        name_to_return = "Unknown" # Platzhalter

        # Beispiel-Logik (ohne learning_threshold):
        # if min_distance < self.recognition_threshold:
        #     name_to_return = best_match_name
        #     is_known_face = True
        # else:
        #     name_to_return = "Unknown"
        #     is_known_face = False
        # return name_to_return, is_known_face, confidence
        # --- Ende TODO 3 ---

        # Platzhalter Rückgabe - Muss durch Logik aus TODO 3 ersetzt werden!
        return "Unknown", False, 0


    # --- Die folgenden Methoden bleiben größtenteils unverändert ---
    # Sie behandeln das kontinuierliche Lernen, UI-Interaktionen etc.
    # Die Studierenden müssen sie nicht direkt für Aufgabe 2 ändern,
    # aber sie sind notwendig, damit das Speichern neuer Gesichter funktioniert.

    def add_landmark_to_person(self, name, landmarks):
        """Add a new landmark set to an already known person (with diversity check)."""
        # (Code bleibt unverändert)
        if name in self.known_face_names:
            person_index = self.known_face_names.index(name)
            landmarks_list = self.known_face_landmarks_collection[person_index]
            current_time = time.time()
            last_add_time = self.learning_cooldown.get(name, 0)
            if current_time - last_add_time < self.min_learning_interval: return False
            landmarks_count = len(landmarks_list)
            dynamic_threshold = self.base_diversity_threshold * (1.0 + (landmarks_count / self.max_samples_per_person))
            diversity_score = self._calculate_landmark_diversity(landmarks, landmarks_list)
            if len(landmarks_list) >= self.max_samples_per_person: return False # Max samples reached
            if diversity_score > dynamic_threshold:
                landmarks_list.append(landmarks)
                print(f"New landmark set added for '{name}' (now {len(landmarks_list)})")
                self.learning_cooldown[name] = current_time
                if len(landmarks_list) > 5 and len(landmarks_list) % 5 == 0: self._clean_landmark_outliers(person_index)
                self.save_landmarks_data()
                return True
        return False

    def _calculate_landmark_diversity(self, new_landmark, existing_landmarks):
        """Calculate diversity of a new landmark."""
        # (Code bleibt unverändert)
        if not existing_landmarks: return 1.0
        avg_landmark = np.mean(existing_landmarks, axis=0)
        existing_distances = [np.linalg.norm(lm - avg_landmark) for lm in existing_landmarks]
        avg_existing_distance = np.mean(existing_distances) if existing_distances else 0
        new_distance = np.linalg.norm(new_landmark - avg_landmark)
        individual_distances = [np.linalg.norm(new_landmark - lm) for lm in existing_landmarks]
        min_individual_distance = min(individual_distances) if individual_distances else 0
        if avg_existing_distance > 0 and min_individual_distance > 0:
            avg_dist_factor = new_distance / avg_existing_distance
            individual_factor = min_individual_distance / avg_existing_distance
            diversity = 0.7 * avg_dist_factor + 0.3 * individual_factor
            return min(max(diversity, 0), 2.0)
        return 1.0

    def _clean_landmark_outliers(self, person_index):
        """Clean outliers from a person's landmark collection."""
        # (Code bleibt unverändert)
        landmarks_list = self.known_face_landmarks_collection[person_index]
        if len(landmarks_list) <= 3: return
        avg_landmark = np.mean(landmarks_list, axis=0)
        distances = [np.linalg.norm(lm - avg_landmark) for lm in landmarks_list]
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        if std_dist > 0:
            outlier_indices = [i for i, d in enumerate(distances) if d > mean_dist + 2.5*std_dist]
            outlier_indices = sorted(outlier_indices, reverse=True)[:2]
            for i in outlier_indices:
                print(f"Removing outlier landmark for {self.known_face_names[person_index]}")
                landmarks_list.pop(i)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for learning faces."""
        # (Code bleibt unverändert)
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
        """Detect faces, extract landmarks, recognize, and draw results."""
        # (Code bleibt unverändert, nutzt die Ergebnisse von extract und compare)
        display_frame = frame.copy()
        self.current_frame = frame.copy()
        face_landmarks_list, face_locations = self.extract_face_landmarks(frame)
        self.button_area = []

        for i, (landmarks, face_loc) in enumerate(zip(face_landmarks_list, face_locations)):
            top, right, bottom, left = face_loc
            name, is_known_face, confidence = self.compare_landmarks(landmarks) # HIER werden die TODOs relevant

            # Continuous learning logic (optional)
            if is_known_face and confidence < 0.95 and min_distance < self.learning_threshold: # Check min_distance if available
                 self.add_landmark_to_person(name, landmarks)

            # Draw bounding box
            color = (0, 255, 0) if is_known_face else (0, 0, 255)
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)

            # Draw label with name and confidence
            label_top = bottom + 10
            label_bottom = bottom + 35
            cv2.rectangle(display_frame, (left, label_top), (right, label_bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            conf_text = f"{confidence:.2f}" if confidence > 0 else "N/A"
            cv2.putText(display_frame, f"{name} ({conf_text})", (left + 6, label_top + 20), font, 0.5, (255, 255, 255), 1)

            # Add "Learn Face" button for unknown faces
            if self.state == "normal" and not is_known_face:
                button_left, button_top = left, top - 30
                button_right, button_bottom = right, top
                if button_top > 0:
                    cv2.rectangle(display_frame, (button_left, button_top), (button_right, button_bottom), (255, 0, 0), cv2.FILLED)
                    cv2.putText(display_frame, "Learn Face", (button_left + 5, button_top + 20), font, 0.5, (255, 255, 255), 1)
                    self.button_area.append(((button_left, button_top, button_right, button_bottom), face_loc))

        return display_frame

    def draw_text_input(self, frame):
        """Draw text input interface."""
        # (Code bleibt unverändert)
        height, width = frame.shape[:2]
        input_height = 40
        cv2.rectangle(frame, (0, height - input_height), (width, height), (50, 50, 50), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Enter name: {self.current_text}_"
        cv2.putText(frame, text, (10, height - 15), font, 0.7, (255, 255, 255), 1)
        instructions = "Press ENTER to save, ESC to cancel"
        cv2.putText(frame, instructions, (width - 300, height - 15), font, 0.5, (200, 200, 200), 1)
        return frame

    def run(self):
        """Main loop for face recognition"""
        # (Code bleibt unverändert, steuert den Ablauf und die UI-Modi)
        cv2.namedWindow('Face Recognition with MediaPipe (Task 2)')
        cv2.setMouseCallback('Face Recognition with MediaPipe (Task 2)', self.mouse_callback)

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            if self.state == "normal":
                display_frame = self.detect_and_recognize_faces(frame)
                cv2.imshow('Face Recognition with MediaPipe (Task 2)', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('c'): # Clean outliers shortcut
                    print("Cleaning outlier landmarks...")
                    for i in range(len(self.known_face_names)): self._clean_landmark_outliers(i)
                    self.save_landmarks_data()

            elif self.state == "entering_name":
                display_frame = frame.copy()
                if self.selected_face_loc:
                    top, right, bottom, left = self.selected_face_loc
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (255, 255, 0), 2)
                    cv2.putText(display_frame, "Face to be saved", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                display_frame = self.draw_text_input(display_frame)
                cv2.imshow('Face Recognition with MediaPipe (Task 2)', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 13: # ENTER
                    if self.current_text: self.save_face(self.current_text, self.selected_face_loc)
                    self.state = "normal"
                elif key == 27: self.state = "normal" # ESC
                elif key == 8: self.current_text = self.current_text[:-1] # BACKSPACE
                elif key == ord('q'): break
                elif 32 <= key <= 126: self.current_text += chr(key)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_system = FaceRecognitionSystem()
    face_system.run()