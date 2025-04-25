import cv2
import time
import os


class Image:
    image_counter = 0

    def saveImage(self, image, subfolder="data/images"):
        """
        Saves the provided image to a specified folder with an incrementing filename.

        Args:
            image (ndarray): The image frame to be saved (as captured by OpenCV).
            subfolder (str): Optional; the folder path where images will be saved. Defaults to 'data/images'.

        Returns:
            None
        """
        image_filename = f"image_{self.image_counter}.png"
        self.image_counter += 1

        os.makedirs(subfolder, exist_ok=True)
        output_path = os.path.join(subfolder, image_filename)

        print(f"Image saved: {image_filename}")
        cv2.imwrite(output_path, image)

    def getFrame(self):
        """
        Initializes the default webcam and sets the resolution to 640x640 pixels.

        Args:
            None

        Returns:
            cv2.VideoCapture: An OpenCV VideoCapture object used for reading frames from the webcam.
        """
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        return camera

    def initStream(self):
        """
        Starts capturing images from the webcam every 0.5 seconds for a duration of 5 seconds.
        Each captured image is saved to disk using the `saveImage` method.

        Args:
            None

        Returns:
            None
        """
        camera = self.getFrame()

        if not camera.isOpened():
            print("Error: The camera could not be opened.")
            exit()

        start_time = time.time()

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > 5:
                break
            ret, frame = camera.read()
            if not ret:
                print("Error: An image could not be captured from the camera.")
                break
            self.saveImage(frame)
            time.sleep(0.5)

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Image().initStream()
