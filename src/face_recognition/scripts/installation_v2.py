import os
import sys
import subprocess


# === CHECK IF CONDA IS INSTALLED ==================================================

def install_miniconda():
    print("Miniconda is not installed. Installing Miniconda...")

    # Miniconda download and installation
    if os.name == "posix":  # Linux/macOS
        print("Downloading Miniconda for Linux/macOS...")
        subprocess.run(
            ["wget", "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh", "-O", "miniconda.sh"])
        subprocess.run(["bash", "miniconda.sh", "-b", "-p", "/home/user/miniconda"])  # Change path if needed
        subprocess.run(["rm", "miniconda.sh"])
        print("Miniconda installed successfully.")
        subprocess.run(
            ["source", "/home/user/miniconda/bin/activate"])  # Activate miniconda (you may need to adjust the path)
    elif os.name == "nt":  # Windows
        print("Miniconda is not available for installation on Windows in this script.")
        sys.exit(1)


# === CHECK IF CONDA IS INSTALLED ==================================================

try:
    # Try running `conda --version` to check if Conda is installed
    subprocess.run(["conda", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    print("Conda is already installed.")
except subprocess.CalledProcessError:
    install_miniconda()

# === GENERAL SETUP ==========================================================
# Create and activate a Miniconda environment for safe package installation

env_name = "yolo_env"
# Check if conda environment exists
env_check_command = ["conda", "env", "list"]
try:
    result = subprocess.run(env_check_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if env_name in result.stdout:
        print(f"Environment '{env_name}' already exists.")
    else:
        print(f"Creating conda environment '{env_name}'...")
        subprocess.run(["conda", "create", "--name", env_name, "python=3.8", "-y"])  # Choose Python version as required
        print(f"Conda environment '{env_name}' created.")
except Exception as e:
    print(f"Error checking or creating conda environment: {e}")

# Activate the conda environment
activate_command = f"conda activate {env_name}"

print("\nTo activate the conda environment, run:")
print(f"    {activate_command}\n")

# === LIBRARY CATEGORIES =====================================================

# General purpose libraries
general_libraries = [
    "numpy==1.26.4",  # For mathematical operations
    "opencv-python",  # For image processing and real-time recognition
    "matplotlib",  # For visualizing results
    "pygame",  # For multimedia applications
]

# License plate recognition libraries
license_plate_libraries = [
    "easyocr",  # OCR for license plate text recognition
    "ultralytics",  # YOLO models for license plate detection
]

# Face recognition libraries
face_recognition_libraries = [
    "mediapipe",  # Face detection and image processing
    "face_recognition",  # Face recognition using deep learning
]

# Combine all non-built-in libraries into one list
all_libraries = (
        [lib for lib in general_libraries] +
        license_plate_libraries +
        face_recognition_libraries
)

# === INSTALLATION ===========================================================

print("Installing required libraries into the conda environment...")

# Use conda to install dependencies
try:
    for lib in all_libraries:
        print(f"Installing {lib} with conda...")
        subprocess.run(["conda", "install", "-c", "conda-forge", lib, "-y"])
        print(f"{lib} installed successfully.\n")

    # Install dlib and face_recognition with conda (as per requirement)
    subprocess.run(["conda", "install", "-c", "conda-forge", "dlib", "-y"])
    subprocess.run(["conda", "install", "-c", "conda-forge", "face_recognition", "-y"])

    print("All installations completed. The system is ready!")

except Exception as e:
    print(f"Error during installation: {e}")
