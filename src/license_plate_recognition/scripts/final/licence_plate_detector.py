import os
import cv2
from ultralytics import YOLO


# Function to retrieve full paths of all images in a specified folder
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


# Function to save an image to a specified folder
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
    # Ensure the target folder exists, create if necessary
    os.makedirs(subfolder, exist_ok=True)
    output_path = os.path.join(subfolder, image_name)

    # Save the image using OpenCV
    cv2.imwrite(output_path, image)


# Main function to process images and detect license plates
def process_images(model, images):
    """
    Processes a list of images: detects, extracts, and saves license plates found in them.

    Args:
        model (YOLO): Pre-trained YOLO model for object detection.
        images (list): List of file paths for the images to process.

    Returns:
        None
    """
    for file_path in images:
        # Load the image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Failed to load image: {file_path}. Skipping...")
            continue

        # Perform detection using the YOLO model
        results = model(file_path)

        plate_counter = 0  # Counter for multiple license plates in one image

        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates and validate the confidence level
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
                confidence = box.conf[0].item()  # Confidence score of the prediction
                class_id = int(box.cls[0].item())  # Class ID of the detection

                if confidence > 0.4 and class_id == 0:  # Check if the object is a license plate
                    # Extract the license plate region from the image
                    license_plate = image[y1:y2, x1:x2]

                    # Generate a unique filename for the license plate
                    plate_filename = f"plate_{os.path.splitext(os.path.basename(file_path))[0]}_{plate_counter}.png"

                    plate_counter += 1  # Increment the counter

                    # Save the extracted license plate
                    save_image(plate_filename, license_plate, subfolder="data/detected_plates/license_plates")


                    # Draw the bounding box and label on the original image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
                    cv2.putText(
                        image,
                        f"license_plate {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )
        # Save the annotated image with bounding boxes
        save_image(os.path.basename(file_path), image)


# Main program: Load the model and process the images
if __name__ == "__main__":
    # Load the YOLO model (pre-configured for license plate detection)
    model = YOLO("model/licence_plate_v11_ncnn_model")

    # Ensure the image directory exists and retrieve images
    images = get_images()

    if not images:
        print("No images were found. Ensure there are images in the 'data/images' directory.")
    else:
        # Process the retrieved images
        process_images(model, images)
