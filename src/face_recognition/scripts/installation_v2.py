import os
import subprocess
import sys

# === CHECK IF CONDA IS INSTALLED ===================================================
def check_conda_installed():
    try:
        subprocess.run(["conda", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("Conda is already installed.")
        return True
    except FileNotFoundError:
        print("Conda is not installed.")
        return False

# === INSTALL CONDA IF NECESSARY ===================================================
def install_conda():
    print("Installing Miniconda...")

    # Get the path to the current script directory and build the relative path to miniconda.sh
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
    miniconda_script = os.path.join(script_dir, "miniconda.sh")  # Relative path to miniconda.sh

    # Check if the Miniconda installation file exists
    if not os.path.exists(miniconda_script):
        print(f"Miniconda installation script not found at {miniconda_script}. Please ensure it is downloaded.")
        return

    # Install Miniconda
    subprocess.run(["bash", miniconda_script, "-b", "-p", "$HOME/miniconda"])

    # Add Conda to PATH
    subprocess.run('echo "export PATH=\"$HOME/miniconda/bin:$PATH\"" >> ~/.bashrc', shell=True)

    # Reload the shell configuration to update PATH
    subprocess.run("source ~/.bashrc", shell=True, executable="/bin/bash")

    # Check if Conda is now available
    subprocess.run(["conda", "--version"])

# === CREATE AND ACTIVATE CONDA ENVIRONMENT ====================================
def create_conda_env():
    env_name = "env_yolo"
    print(f"Creating conda environment '{env_name}'...")

    # Create Conda environment
    subprocess.run([f"conda", "create", "--name", env_name, "-y"])

    # Activate the environment (this may vary depending on shell)
    print(f"To activate the conda environment, run:\n    conda activate {env_name}\n")

# === INSTALL LIBRARIES USING CONDA ============================================
def install_libraries(env_name):
    print("Installing required libraries in the conda environment...")

    # Activate the environment
    subprocess.run([f"conda", "activate", env_name])

    # General purpose libraries
    general_libraries = [
        "numpy==1.26.4",
        "opencv-python",
        "matplotlib",
        "pygame"
    ]

    # License plate recognition libraries
    license_plate_libraries = [
        "easyocr",
        "ultralytics",
    ]

    # Face recognition libraries
    face_recognition_libraries = [
        "mediapipe",
        "face_recognition",
    ]

    # Combine all libraries into one list
    all_libraries = general_libraries + license_plate_libraries + face_recognition_libraries

    # Install libraries one by one
    for lib in all_libraries:
        try:
            print(f"Installing {lib}...")
            subprocess.run([f"conda", "install", lib, "-c", "conda-forge", "-y"])
            print(f"{lib} installed successfully.\n")
        except Exception as e:
            print(f"Error installing {lib}: {e}\n")

    print("All libraries have been installed!")

# === MAIN SCRIPT ================================================================
if not check_conda_installed():
    install_conda()

create_conda_env()

# After creating the environment and activating it, install the libraries
env_name = "env_yolo"
install_libraries(env_name)

print("Setup is complete! Now activate your environment using:")
print(f"    conda activate {env_name}")
