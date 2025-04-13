import re

import numpy as np
import pytesseract
import cv2
import os


def configure_tesseract():
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



def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return gray, thresh


def apply_dilation(thresh):
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(thresh, rect_kern, iterations=1)


def find_and_sort_contours(dilation):
    try:
        contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        _, contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])


def extract_text_from_contours(contours, gray, thresh):
    plate_num = ""
    total_confidence = 0
    num_chars = 0

    im2 = gray.copy()
    height, width = gray.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

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
        roi = thresh[y - 5:y + h + 5, x - 5:x + w + 5]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 5)

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




def get_text(image_path):

    for file in image_path:
        image = cv2.imread(file)
        if image is None:
            print(f"Error: Unable to read the image {file}. Skipping.")
            continue


        gray, thresh = preprocess_image(image)

        dilation = apply_dilation(thresh)

        contours = find_and_sort_contours(dilation)

        result, avg_confidence, im2 = extract_text_from_contours(contours, gray, thresh)


        # Display the combined result
        if result:
            print(f"Image: {os.path.basename(file)}")
            print(f"Text: {result} | Probabilities: {avg_confidence}")
        else:
            print(f"Image: {os.path.basename(file)} → No valid text detected.")

        # Save the annotated image
        save_image(os.path.basename(file), im2)


def main():
    configure_tesseract()
    images = get_images()
    get_text(images)

if __name__ == "__main__":
    main()