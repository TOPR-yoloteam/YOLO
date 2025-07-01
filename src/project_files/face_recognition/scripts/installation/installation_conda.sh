#!/bin/bash

# Install Miniconda if not installed
if ! command -v conda &> /dev/null; then
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh
  bash miniconda.sh -b -p "$HOME/miniconda3"
  echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
  export PATH="$HOME/miniconda3/bin:$PATH"
  conda init bash
  source ~/.bashrc
fi

# Create conda env from YAML
conda env create -f yolo_env_conda.yml
