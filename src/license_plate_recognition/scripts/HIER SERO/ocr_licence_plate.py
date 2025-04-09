import easyocr
import cv2
import os


def get_images():
    image_folder = "data/detected_plates"
    images = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(".png")]
    return images


def process_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return gray


def save_image(image_name, image):
    output_folder = "data/ocr_images"
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image)


def read_text(images):
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])
    for file in images:
        # Load the image as a numpy array using cv2.imread
        image = cv2.imread(file)
        if image is None:
            print(f"Error: Unable to read the image {file}. Skipping.")
            continue

        # Process the image for OCR
        processed_image = process_image(file)

        # Perform OCR using EasyOCR
        result = reader.readtext(processed_image)

        # Draw bounding boxes and text on the original image
        for (bbox, text, prob) in result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save both the processed image for verification and OCR-ed image if needed
        save_image(os.path.basename(file), processed_image)


if __name__ == "__main__":
    # Ensure output folder exists
    if not os.path.exists("data/ocr_images"):
        os.makedirs("data/ocr_images")

    images = get_images()
    read_text(images)
    # Get images and execute the OCR pipeline