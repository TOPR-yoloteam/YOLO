import os
import cv2
from ultralytics import YOLO


def get_images(image_folder="data/images", file_extension=".png"):
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
        if file.endswith(file_extension)
    ]


def save_image(image_name, image, subfolder="data/detected_plates"):
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


def process_images(model, dir):
    """
    Processes a list of images: detects, extracts, and saves license plates found in them.

    Args:
        model (YOLO): Pre-trained YOLO model for object detection.
        dir (list): List of file paths for the images to process.

    Returns:
        None
    """
    #TODO


if __name__ == "__main__":
    """
    Entry point of the script.

    Steps:
        1. Load the YOLO model preconfigured for license plate detection.
        2. Retrieve images from the specified directory.
        3. Process the images to detect and extract license plates.
           - Annotate images with bounding boxes and save results.
        4. Handle cases where no images are found.
    """
    model = YOLO("model/licence_plate_ncnn_model")

    images = get_images()

    if not images:
        print("No images were found. Ensure there are images in the 'data/images' directory.")
    else:
        process_images(model, images)
