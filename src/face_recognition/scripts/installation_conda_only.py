import os
import subprocess
import sys
import shutil
import platform
from pathlib import Path

HOME = str(Path.home())
MINICONDA_DIR = os.path.join(HOME, "miniconda3")
MINICONDA_BIN = os.path.join(MINICONDA_DIR, "bin", "conda")
ENV_NAME = "yolo_env_conda"

def is_miniconda_installed():
    return os.path.exists(MINICONDA_BIN)

def install_miniconda():
    print("?? Miniconda wird installiert...")
    url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    installer_path = os.path.join(HOME, "miniconda_installer.sh")

    subprocess.run(["wget", url, "-O", installer_path], check=True)
    subprocess.run(["bash", installer_path, "-b", "-p", MINICONDA_DIR], check=True)

    os.remove(installer_path)
    print("? Miniconda wurde installiert.")

def create_env():
    print(f"?? Erstelle Environment: {ENV_NAME}")
    subprocess.run([MINICONDA_BIN, "create", "-y", "-n", ENV_NAME, "python=3.10"], check=True)
    print(f"? Environment '{ENV_NAME}' erstellt.")

def print_activation_instructions():
    print("\n??  Um das Environment zu aktivieren, führe folgendes im Terminal aus:")
    print(f"\n  source {MINICONDA_DIR}/bin/activate {ENV_NAME}\n")

if __name__ == "__main__":
    if not shutil.which("wget"):
        print("? wget ist nicht installiert. Bitte installiere es zuerst (sudo apt install wget).")
        sys.exit(1)

    if not is_miniconda_installed():
        install_miniconda()
    else:
        print("? Miniconda ist bereits installiert.")

    create_env()
    print_activation_instructions()
