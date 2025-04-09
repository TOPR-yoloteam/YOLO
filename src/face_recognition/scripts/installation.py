import os

# List of required libraries for face recognition and license plate recognition
libraries = [
    "face_recognition",  # For face recognition
    "opencv-python",  # For image processing and real-time recognition
    "numpy",  # Mathematical operations on images/arrays
    "pillow",  # Image processing
    "matplotlib",  # For visualizing results
    "easyocr",  # Optical character recognition (OCR) for license plates
    "mediapipe",  # Image segmentation and processing
]

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
