import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
import face_recognition
from datetime import datetime
import statistics

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
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "models", "yolov8n-face.pt")

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

        # Performance-Metriken
        frame_times = []
        process_times = []
        faces_detected = []
        instant_fps_values = []
        fps_values = []
        last_record_time = time.time()
        record_interval = 5  # Alle 5 Sekunden aufzeichnen
        total_duration = 60  # Gesamtdauer der Messung in Sekunden
        start_time = time.time()

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time > total_duration:
                break

            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)

            # Zeitmessung für die Verarbeitung
            process_start = time.time()
            if self.state == "normal":
                display_frame, _ = self.detect_and_recognize_faces(frame)
            elif self.state == "entering_name":
                display_frame = frame.copy()
                display_frame = self.draw_text_input(display_frame)
            process_end = time.time()

            # Verarbeitungszeit speichern
            process_time = process_end - process_start
            process_times.append(process_time)

            # FPS berechnen
            frame_times.append(current_time)
            if len(frame_times) > 2:
                avg_frame_time = (frame_times[-1] - frame_times[0]) / (len(frame_times) - 1)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                instant_fps_values.append(fps)
            else:
                fps = 0

            # Stabilitätsbewertung
            avg_process_time = sum(process_times) / len(process_times) if process_times else 0
            face_count = len(self.button_area)
            faces_detected.append(face_count)

            stability_score = 100
            face_stability = 0
            fps_stability = 0

            if len(faces_detected) > 1:
                try:
                    face_stability = statistics.stdev(faces_detected)
                except statistics.StatisticsError:
                    face_stability = 0

            if len(instant_fps_values) > 1:
                try:
                    fps_stability = statistics.stdev(instant_fps_values)
                except statistics.StatisticsError:
                    fps_stability = 0

            if fps > 0:
                stability_score = round(100 - min(100, (10 * fps_stability / max(fps, 1)) + (20 * face_stability)))
                stability_score = max(0, min(100, stability_score))

            # Alle 5 Sekunden Performance-Metriken speichern
            if current_time - last_record_time >= record_interval:
                performance_data = {
                    "time": int(elapsed_time),
                    "fps": round(fps, 2),
                    "avg_process_time": round(avg_process_time * 1000, 2),  # In ms
                    "avg_faces": round(sum(faces_detected) / len(faces_detected), 2),
                    "stability": stability_score
                }
                fps_values.append(performance_data)
                print(f"Zeit: {performance_data['time']}s, FPS: {performance_data['fps']}, "
                      f"Verarbeitungszeit: {performance_data['avg_process_time']}ms, "
                      f"Erkannte Gesichter: {performance_data['avg_faces']}, "
                      f"Stabilität: {performance_data['stability']}%")
                last_record_time = current_time

            # Performance-Infos anzeigen
            cv2.putText(display_frame, f"FPS: {round(fps, 1)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Verarbeitung: {round(avg_process_time * 1000, 1)}ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Gesichter: {face_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Stabilität: {stability_score}%", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Zeit: {int(elapsed_time)}s/{total_duration}s", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Face Recognition', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # Performance-Daten ausgeben
        print("\nPerformance-Aufzeichnung abgeschlossen!")
        print("Zeit (s) | FPS | Verarbeitung (ms) | Gesichter | Stabilität (%)")
        print("-" * 65)
        for data in fps_values:
            print(
                f"{data['time']:8} | {data['fps']:>4} | {data['avg_process_time']:>15} | {data['avg_faces']:>9} | {data['stability']:>13}")

        # Durchschnittswerte berechnen
        if fps_values:
            avg_fps = statistics.mean([data['fps'] for data in fps_values])
            avg_process = statistics.mean([data['avg_process_time'] for data in fps_values])
            avg_stability = statistics.mean([data['stability'] for data in fps_values])
            print(f"\nDurchschnittliche Werte:")
            print(f"FPS: {round(avg_fps, 2)}")
            print(f"Verarbeitungszeit: {round(avg_process, 2)}ms")
            print(f"Stabilität: {round(avg_stability, 2)}%")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_system = FaceRecognitionSystem()
    face_system.run()
