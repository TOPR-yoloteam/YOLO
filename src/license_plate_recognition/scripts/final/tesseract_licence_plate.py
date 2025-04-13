import re
import pytesseract
import cv2
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Valentin.Talmon\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
#pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

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
    #thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
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
        images (list): A list of image file paths to process.
    """
    for file in images:
        image = cv2.imread(file)
        if image is None:
            print(f"Error: Unable to read the image {file}. Skipping.")
            continue

        # Preprocess the image: convert to grayscale and apply threshold
        processed_image = process_image(image)

        # Perform OCR on the processed image
        result = pytesseract.image_to_data(processed_image,config ="--oem 3 --psm 7", output_type=pytesseract.Output.DICT)

        # Convert the grayscale image back to BGR for annotation
        image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)


        # Collect detected text and corresponding probabilities
        texts = []
        probs = []
        counter = 0
        # Iterate through the OCR results and draw bounding boxes on the detected text

        for i in range(len(result["text"])):
            text = result["text"][i]
            conf = int(result["conf"][i])
            if conf > 30:
                filtered = filter_uppercase_and_numbers(text)
                if filtered:
                    texts.append(filtered)
                    probs.append(conf)

                    x,y,w,h = result["left"][i],result["top"][i],result["width"][i],result["height"][i]
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)



        """
        for (bbox, text, prob) in result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))


            # Clean text: keep only uppercase letters and numbers
            filtered_text = filter_uppercase_and_numbers(text)

            if filtered_text != "":
                texts.append(filtered_text)
                probs.append(round(prob, 2))  # Round for cleaner output
                # Draw bounding box around detected text region

            cv2.rectangle(image, top_left, bottom_right, (238, 130, 238), 2)
            """



        """
        # Only annotate results where the probability exceeds a threshold
        if prob > 0.3:
            # Filter the detected text to include only valid characters
            text = filter_uppercase_and_numbers(text)
            print(f"Text: {text}, Probability: {prob}")

            # Draw a rectangle around the detected text region

            cv2.rectangle(image, top_left, bottom_right, (238, 130, 238), 2)
        """
        # Display the combined result
        if texts:
            combined_text = ' '.join(texts)  # Use ' '.join(texts) if spaces are preferred
            print(f"Image: {os.path.basename(file)}")
            print(f"Text: {combined_text} | Probabilities: {probs}")
        else:
            print(f"Image: {os.path.basename(file)} → No valid text detected.")

        # Save the annotated image
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
