import re
import easyocr
import cv2
import os


def get_images(image_folder = "data/detected_plates/license_plates", file_extension=".png"):
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


def save_image(image_name, image, subfolder="data/ocr_images"):
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


def process_image(image):
    """
    Process an image by converting it to grayscale and applying a binary threshold.

    Args:
        image (numpy.ndarray): The input image in BGR (Blue-Green-Red) format.

    Returns:
        numpy.ndarray: The processed image after applying binary thresholding.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding with a fixed threshold value
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    return thresh





def filter_uppercase_and_numbers(input_string):
    """
    Filter the input string to retain only uppercase letters, numbers, and whitespace characters.

    Args:
        input_string (str): The input string to be filtered.

    Returns:
        str: The filtered string with only uppercase letters and numbers.
    """
    result = re.sub(r"[^A-Z0-9\s]", "", input_string)
    return result


def read_text(images):
    """
    Perform Optical Character Recognition (OCR) on the provided list of images.
    The function processes each image, extracts text, and annotates detected text regions.

    Args:
        images (list): A list of file paths representing images to be processed.
    """
    # Initialize the EasyOCR reader with English language support
    reader = easyocr.Reader(['en'])
    for file in images:
        image = cv2.imread(file)
        if image is None:
            print(f"Error: Unable to read the image {file}. Skipping.")
            continue

        # Preprocess the image for OCR
        processed_image = process_image(image)

        # Perform OCR using EasyOCR on the processed image
        result = reader.readtext(processed_image)

        # Convert the processed grayscale image back to BGR format for annotation purposes
        image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        counter = 0
        # Iterate through the OCR results and draw bounding boxes on the detected text
        for (bbox, text, prob) in result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            prob_text = filter_uppercase_and_numbers(text)
            print(f"Text: {prob_text}, Probability: {prob}")
            cv2.rectangle(image, top_left, bottom_right, (238, 130, 238), 2)


            """      
            # Only annotate results where the probability exceeds a threshold
            if prob > 0.3:
                # Filter the detected text to include only valid characters
                text = filter_uppercase_and_numbers(text)
                print(f"Text: {text}, Probability: {prob}")

                # Draw a rectangle around the detected text region
                cv2.rectangle(image, top_left, bottom_right, (238, 130, 238), 2)
            """


        # Save the annotated image to the output folder
        save_image(os.path.basename(file), image)


if __name__ == "__main__":
    """
    Main entry point of the script.
    The script initializes the output folder, retrieves images, and executes the OCR pipeline.
    """
    # Ensure the output folder exists, creating it if necessary
    if not os.path.exists("data/ocr_images"):
        os.makedirs("data/ocr_images")

    # Retrieve the list of images for processing
    images = get_images()

    # Execute the OCR pipeline on the retrieved images
    read_text(images)
