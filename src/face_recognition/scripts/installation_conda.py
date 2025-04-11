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
    miniconda_script_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    miniconda_script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "miniconda.sh")

    # Download Miniconda installation script
    subprocess.run(["wget", miniconda_script_url, "-O", miniconda_script_path])

    # Resolve the Home directory correctly
    home_dir = os.path.expanduser("~")
    miniconda_install_path = f"{home_dir}/miniconda"

    # Install Miniconda
    subprocess.run(["bash", miniconda_script_path, "-b", "-p", miniconda_install_path])

    # Add Conda to PATH
    subprocess.run(f'echo "export PATH=\"{miniconda_install_path}/bin:$PATH\"" >> ~/.bashrc', shell=True)
    subprocess.run("source ~/.bashrc", shell=True)

    # Check if Conda is now available
    subprocess.run(["conda", "--version"])


# === CREATE AND ACTIVATE CONDA ENVIRONMENT ====================================
def create_conda_env():
    env_name = "env_yolo"
    print(f"Creating conda environment '{env_name}'...")

    # Create Conda environment
    subprocess.run([f"conda", "create", "--name", env_name, "-y"])


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
