import os
import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO

class FaceRecognitionSystem:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initializes the YOLO model for person detection and face recognition system.

        Args:
            model_path (str): Path to the pre-trained YOLO model.
        """
        self.model = YOLO(model_path)
        self.known_face_encodings = []
        self.known_face_names = []
        self.button_area = []
        self.state = "normal"
        self.current_frame = None
        self.recognition_size = None

    def get_images(self, image_folder="data/images", file_extension=[".png", ".jpg", ".jpeg"]):
        """
        Collects all image files with specified extensions from a folder.

        Args:
            image_folder (str): Path to the folder containing images.
            file_extension (list): List of allowed file extensions.

        Returns:
            list: A list of absolute paths to all valid image files.
        """
        return [
            os.path.join(image_folder, file)
            for file in os.listdir(image_folder)
            if file.endswith(tuple(file_extension))
        ]

    def save_image(self, image_name, image, subfolder="data/recognized_faces"):
        """
        Saves an image to the desired folder.

        Args:
            image_name (str): Name of the image file to save.
            image (np.ndarray): The image (as a NumPy array) to be saved.
            subfolder (str): Destination path where the image will be saved.

        Returns:
            None
        """
        os.makedirs(subfolder, exist_ok=True)
        output_path = os.path.join(subfolder, image_name)
        cv2.imwrite(output_path, image)

    def detect_and_recognize_faces(self, frame):
        """
        Detect persons using YOLO and recognize faces using face_recognition.

        Args:
            frame (np.ndarray): Input image frame.

        Returns:
            np.ndarray: Output image frame with drawings.
            list: Face locations detected.
        """
        display_frame = frame.copy()
        self.current_frame = frame.copy()

        # Resize frame for face recognition
        height, width = frame.shape[:2]
        self.recognition_size = (width, height)

        # --- TODO 1: Person detection only ---
        results = self.model(frame, classes=[0])  # Class 0 = person in COCO
        if results and len(results) > 0:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # --- End of TODO 1 ---

        # --- TODO 2: Face recognition ---
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

            # Draw box and label
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            label_top = bottom + 10
            label_bottom = bottom + 45
            cv2.rectangle(display_frame, (left, label_top), (right, label_bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(display_frame, name, (left + 6, label_top + 20), font, 0.7, (255, 255, 255), 1)

            # Learn face button for unknowns
            if self.state == "normal" and not is_known_face:
                button_left = left
                button_top = top - 30
                button_right = right
                button_bottom = top

                if button_top > 0:
                    cv2.rectangle(display_frame, (button_left, button_top), (button_right, button_bottom), (255, 0, 0), cv2.FILLED)
                    cv2.putText(display_frame, "Learn Face", (button_left + 5, button_top + 20), font, 0.5, (255, 255, 255), 1)
                    self.button_area.append(((button_left, button_top, button_right, button_bottom), (top, right, bottom, left)))
        # --- End of TODO 2 ---

        return display_frame, face_locations

    def process_images(self, images):
        """
        Processes a list of images: detects persons, recognizes faces.

        Args:
            images (list): List of file paths to images.

        Returns:
            None
        """
        for file in images:
            image = cv2.imread(file)
            if image is None:
                print(f"Error: Failed to load image: {file}. Skipping...")
                continue

            processed_frame, _ = self.detect_and_recognize_faces(image)

            # Save processed frame (optional)
            image_name = os.path.basename(file)
            self.save_image(image_name, processed_frame)

    def run(self):
        """
        Entry point for the system:
        - Loads images
        - Detects persons and recognizes faces
        - Handles errors if no images found

        Returns:
            None
        """
        images = self.get_images()
        if not images:
            print("No images found. Please add images to the 'data/images' directory.")
        else:
            self.process_images(images)

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()
