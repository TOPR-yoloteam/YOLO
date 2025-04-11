#!/bin/bash

set -e

# Farben für lesbare Ausgabe
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}?? Lade Miniconda für ARM (aarch64)...${NC}"
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh

echo -e "${GREEN}?? Installiere Miniconda ins Home-Verzeichnis...${NC}"
bash miniconda.sh -b -p "$HOME/miniconda3"

# bashrc dauerhaft erweitern (nur wenn nötig)
if ! grep -q 'miniconda3/bin' "$HOME/.bashrc"; then
  echo -e "${GREEN}?? Trage Miniconda PATH dauerhaft in ~/.bashrc ein...${NC}"
  echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> "$HOME/.bashrc"
else
  echo -e "${GREEN}? PATH ist bereits in ~/.bashrc vorhanden.${NC}"
fi

# PATH für aktuelle Session setzen
export PATH="$HOME/miniconda3/bin:$PATH"

# conda init (setzt u.a. base automatisch in neue Shells)
echo -e "${GREEN}?? Initialisiere Conda (fügt base in Shell ein)...${NC}"
conda init bash

# ~/.bashrc neu laden
source ~/.bashrc

# Environment-Namen
ENV_NAME="yolo_env_conda"

# Prüfen, ob Environment bereits existiert
if conda env list | grep -q "$ENV_NAME"; then
  echo -e "${GREEN}?? Environment '$ENV_NAME' existiert bereits.${NC}"
else
  echo -e "${GREEN}?? Erstelle Conda-Environment '$ENV_NAME'...${NC}"
  conda create -y -n "$ENV_NAME" python=3.9
fi

# Environment aktivieren
echo -e "${GREEN}?? Aktiviere Environment '$ENV_NAME'...${NC}"
source activate "$ENV_NAME"

# Test
echo -e "${GREEN}? Setup abgeschlossen. Conda-Version:${NC}"
conda --version
echo -e "${GREEN}Aktuelles Environment:${NC}"
conda info --envs
echo -e "${GREEN}Führe Befehl <source ~/.bashr> aus und aktiviere Environment mit <conda activate yolo_env_conda>"