import os

# List of required libraries for face recognition and license plate recognition
libraries = [
    "cv2",  # For image processing and real-time recognition
    "numpy",  # Mathematical operations on images/arrays
    "ultralytics" #import YOLO
    "matplotlib",  # For visualizing results
    "easyocr",  # Optical character recognition (OCR) for license plates
    "mediapipe",  # Image segmentation and processing
    "face_recognition",  # YOLO lib for face recognition
    "time"
    "pygame"
    "threading" #import Thread
    "os"
    "queue" #import Queue

]
#mediapipe 0.10.21 requires numpy<2, but you have numpy 2.2.4 which is incompatible.
#ultralytics 8.3.100 requires numpy<=2.1.1,>=1.23.0, but you have numpy 2.2.4 which is incompatible.
#numpy==1.26.4
#python==3.12.9

# Install each library
print("Starting installation of required libraries...")
for library in libraries:
    try:
        print(f"Installing {library}...")
        os.system(f"pip install {library}")
        print(f"{library} installed successfully!")
    except Exception as e:
        print(f"Error installing {library}: {e}")

print("All installations completed. System is ready!")
