import os
import cv2
from ultralytics import YOLO


class LicensePlateDetector:
    def __init__(self):
        """
        Initializes the YOLO model for license plate detection.

        Args:
            model_path (str): Path to the pre-trained YOLO model.
        """
        self.model = YOLO("model/license_plate_detector_ncnn_model")
        self.plate_counter = 0


    def get_images(self, image_folder="data/images", file_extension=[".png", ".jpg", ".jpeg"]):
        """
        Collects all image files with a specified extension from a folder.

        Args:
            image_folder (str): Path to the folder containing images.
            file_extension (str): File extension to filter by (e.g., ".png").

        Returns:
            list: A list of absolute paths to all valid image files.
        """
        return [
            os.path.join(image_folder, file)
            for file in os.listdir(image_folder)
            if file.endswith(tuple(file_extension))
        ]

    def save_image(self, image_name, image, subfolder="data/detected_plates"):
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

    def process_images(self, dir):
        """
        Processes a list of images: detects, extracts, and saves license plates found in them.

        Args:
            dir (list): List of file paths for the images to process.

        Returns:
            None
        """
        for file in dir:
            image = cv2.imread(file)
            if image is None:
                print(f"Error: Failed to load image: {file}. Skipping...")
                continue

            results = self.model(file)


            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())

                    if confidence > 0.4 and class_id == 1:
                        license_plate = image[y1:y2, x1:x2]

                        plate_filename = f"plate_{os.path.splitext(os.path.basename(file))[0]}_{self.plate_counter}.png"
                        self.plate_counter += 1

                        self.save_image(plate_filename, license_plate, subfolder="data/detected_plates/license_plates")

                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            image,
                            f"license_plate {confidence:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2
                        )
            self.save_image(os.path.basename(file), image)

    def run(self):
        """
        Entry point for processing:
        - Loads images
        - Detects and saves license plates
        - Handles errors if no images are found

        Returns:
            None
        """
        images = self.get_images()
        if not images:
            print("No images were found. Ensure there are images in the 'data/images' directory.")
        else:
            self.process_images(images)


if __name__ == "__main__":
    detector = LicensePlateDetector()
    detector.run()
