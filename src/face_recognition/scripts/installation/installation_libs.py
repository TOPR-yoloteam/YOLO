import subprocess

# === KONFIGURATION ==========================================================

env_name = "yolo_env_conda"

# Pakete sortiert nach Kategorien (nur conda)
general_libraries = [
    "numpy=1.26.4",
    "opencv",          # conda-Paket für OpenCV
    "matplotlib",
    "pygame"
]

license_plate_libraries = [
    "easyocr",         # wird von conda-forge unterstützt
    "ultralytics"      # nicht direkt verfügbar, evtl. per pip nötig
]

face_recognition_libraries = [
    "mediapipe",
    "face_recognition"
]

all_libraries = general_libraries + license_plate_libraries + face_recognition_libraries

# === KONSTRUKT FÜR SHELL ====================================================

activate_and_run = f"""
source ~/.bashrc
conda activate {env_name}
"""

def conda_install(package: str):
    print(f"??  Installiere {package}...")
    command = f"{activate_and_run} && conda install -y -c conda-forge {package}"
    subprocess.run(command, shell=True, executable="/bin/bash")

# === INSTALLATION ===========================================================

print(f"?? Aktiviere Environment '{env_name}' und installiere Pakete...\n")

for lib in all_libraries:
    conda_install(lib)

print("\n? Setup abgeschlossen. Alle Pakete wurden über Conda installiert.")
