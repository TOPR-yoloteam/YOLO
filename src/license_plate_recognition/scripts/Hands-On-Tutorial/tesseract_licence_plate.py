import numpy as np
import pytesseract
import cv2
import os

class OCR:


def configure_tesseract():
    """
    Configures the Tesseract-OCR executable path.
    This is platform-specific and should be set to the correct path for Tesseract-OCR.
    """
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


def get_images(image_folder="data/detected_plates/license_plates", file_extension=".png"):
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
    Saves an image to the specified folder.

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


def preprocess_image(image):
    """
    Processes the input image to prepare it for contour detection and OCR.

    Steps include:
        1. Grayscale conversion.
        2. Resizing for better OCR performance.
        3. Various blurring techniques to reduce noise.
        4. Thresholding for binary image creation.

    Args:
        image (np.ndarray): The input image to preprocess.

    Returns:
        tuple: A tuple containing the grayscale image and thresholded binary image.
    """
    #TODO



def apply_dilation(thresh):
    """
    Applies dilation to enhance binary image regions for contour detection.

    Args:
        thresh (np.ndarray): Thresholded binary image.

    Returns:
        np.ndarray: Dilated binary image.
    """
    #TODO


def find_and_sort_contours(dilation):
    """
    Finds and sorts contours from a dilated binary image.

    Args:
        dilation (np.ndarray): Dilated binary image.

    Returns:
        list: A list of contours sorted by their x-coordinate.
    """
    try:
        contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        _, contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])


def extract_text_from_contours(contours, gray, dialtion):
    """
    Extracts text from the detected contours using Tesseract OCR.

    - Filters contours based on specific size, aspect ratio, and area criteria.
    - For each valid ROI, text and confidence scores are extracted using OCR.

    Args:
        contours (list): A list of contours detected in the image.
        gray (np.ndarray): Grayscale version of the original image.
        dilation (np.ndarray): Dilated binary image.

    Returns:
        tuple: Extracted license plate text, the average confidence score, and the annotated image.
    """
    plate_num = ""
    total_confidence = 0
    num_chars = 0

    im2 = gray.copy()
    height, width = gray.shape

    for cnt in contours:
        roi = None
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtering out unwanted contours based on heuristics
        if height / float(h) > 6:
            continue
        ratio = h / float(w)
        if ratio < 1.5:
            continue
        area = h * w
        if width / float(w) > 15:
            continue
        if area < 100:
            continue

        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #Expand the region of interest (ROI) for OCR
        x_start = max(0, x - 5)
        y_start = max(0, y - 5)
        x_end = min(width, x + w + 5)
        y_end = min(height, y + h + 5)
        roi = dialtion[y_start:y_end, x_start:x_end]


        ocr_data = pytesseract.image_to_data(
            roi,
            config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3',
            output_type=pytesseract.Output.DICT
        )

        for i, text in enumerate(ocr_data["text"]):
            if text.strip():
                plate_num += text.strip()
                conf = int(ocr_data["conf"][i])
                if conf > 0:
                    total_confidence += conf
                    num_chars += 1

    avg_confidence = total_confidence / num_chars if num_chars > 0 else 0
    return plate_num, avg_confidence, im2


def get_text(dir):
    """
    Processes each image and extracts license plate text using OCR.

    - Loops through the provided image paths.
    - Preprocesses each image.
    - Applies contour detection, text extraction, and displays results.

    Args:
        dir (list): A list of file paths pointing to images.

    Returns:
        None
    """
    #TODO


if __name__ == "__main__":
    """
        Entry point of the script.

        - Configures Tesseract.
        - Collects image paths.
        - Processes images for license plate text detection.

        Returns:
            None
        """
    configure_tesseract()
    images = get_images()
    get_text(images)
