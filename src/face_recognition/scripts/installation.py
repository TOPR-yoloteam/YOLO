import os
import sys
import subprocess

# === GENERAL SETUP ==========================================================
# Create and activate a virtual environment for safe package installation

venv_dir = "venv"

# Check if venv already exists
if not os.path.exists(venv_dir):
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", venv_dir])
    print("Virtual environment created.")

# Determine the platform-specific activation command
if os.name == "nt":
    activate_script = os.path.join(venv_dir, "Scripts", "activate")
else:
    activate_script = f"source {venv_dir}/bin/activate"

print("\nTo activate the virtual environment, run:")
print(f"    {activate_script}\n")

# === LIBRARY CATEGORIES =====================================================

# General purpose libraries
general_libraries = [
    "numpy==1.26.4",  # For mathematical operations
    #mediapipe 0.10.21 requires numpy<2, but you have numpy 2.2.4 which is incompatible.
    #ultralytics 8.3.100 requires numpy<=2.1.1,>=1.23.0, but you have numpy 2.2.4 which is incompatible.
    #numpy==1.26.4
    #python==3.12.9
    "opencv-python",              # For image processing and real-time recognition
    "matplotlib",     # For visualizing results
    "time",           # Built-in, included for completeness
    "os",             # Built-in, included for completeness
    "threading",      # Built-in, included for completeness
    "queue",          # Built-in, included for completeness
    "pygame"          # For multimedia applications
]

# License plate recognition libraries
license_plate_libraries = [
    "easyocr",        # OCR for license plate text recognition
    "ultralytics",    # YOLO models for license plate detection
]

# Face recognition libraries
face_recognition_libraries = [
    "mediapipe",       # Face detection and image processing
    "face_recognition",# Face recognition using deep learning
]

# Combine all non-built-in libraries into one list
all_libraries = (
    [lib for lib in general_libraries if lib not in ("time", "os", "threading", "queue")] +
    license_plate_libraries +
    face_recognition_libraries
)

# === INSTALLATION ===========================================================

print("Installing required libraries into the virtual environment...")

# Use the pip from within the virtual environment
pip_path = os.path.join(venv_dir, "bin", "pip") if os.name != "nt" else os.path.join(venv_dir, "Scripts", "pip.exe")

for lib in all_libraries:
    try:
        print(f"Installing {lib}...")
        subprocess.run([pip_path, "install", lib])
        print(f"{lib} installed successfully.\n")
    except Exception as e:
        print(f"Error installing {lib}: {e}\n")

print("? All installations completed. The system is ready!")
