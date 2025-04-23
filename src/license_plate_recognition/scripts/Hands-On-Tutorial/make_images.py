import cv2
import time
import os

if __name__ == "__main__":
    """
    Main entry point of the script.
    The script captures images from the default camera, saves them to a specified directory,
    and terminates after 5 seconds or when an error occurs.
    """

    # Access the default camera (0 specifies the default primary camera of the device)
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the camera frame width
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)  # Set the camera frame height

    # Create the output directory to store captured images if it does not already exist
    if not os.path.exists("data/images"):
        os.makedirs("data/images")

    # Check if the camera is successfully opened
    if not camera.isOpened():
        print("Error: The camera could not be opened.")
        exit()

    # Initialize the counter for numbering images
    image_counter = 0

    # Record the starting time to track elapsed time during execution
    start_time = time.time()

    try:
        while True:
            # Calculate the elapsed time since the start of image capture
            elapsed_time = time.time() - start_time

            # Stop image capture after 5 seconds
            if elapsed_time > 5:
                break

            # Capture the current frame from the camera
            ret, frame = camera.read()

            # Check if the frame was successfully captured
            if not ret:
                print("Error: An image could not be captured from the camera.")
                break

            # Save the captured frame to the specified directory
            image_filename = f"image_{image_counter}.png"  # Define the filename with the counter
            cv2.imwrite("data/images/" + image_filename, frame)  # Save the image to the output directory
            print(f"Image saved: {image_filename}")  # Log the filename of the saved image

            # Increment the image counter to ensure unique filenames
            image_counter += 1

            # Wait for 0.5 seconds before capturing the next image
            time.sleep(0.5)

    finally:
        # Release the camera resource and close any opened OpenCV windows
        camera.release()
        cv2.destroyAllWindows()
