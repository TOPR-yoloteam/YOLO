import cv2
import time
import os

if __name__ == "__main__":

    # access the camera (0 for the default camera)
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    # create directory
    if not os.path.exists("data/images"):
        os.makedirs("data/images")

    # check if the camera could be opened
    if not camera.isOpened():
        print("error: camera could not be opened.")
        exit()

    # set the image counter
    image_counter = 0

    # record the start time
    start_time = time.time()

    try:
        while True:
            # check if 5 seconds have passed
            elapsed_time = time.time() - start_time
            if elapsed_time > 5:
                break

            # capture image from the camera
            ret, frame = camera.read()

            if not ret:
                print("error: image could not be captured.")
                break

            # save the image to the directory
            image_filename = f"image_{image_counter}.png"
            cv2.imwrite("data/images/" + image_filename, frame)
            print(f"image saved: {image_filename}")

            # increment the counter
            image_counter += 1

            time.sleep(0.5)

    finally:
        # release the camera and clean up resources
        camera.release()
        cv2.destroyAllWindows()
