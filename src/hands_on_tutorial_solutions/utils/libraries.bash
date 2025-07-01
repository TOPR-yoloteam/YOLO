#!/bin/bash

pip install opencv-python

pip install ultralytics
pip install mediapipe
pip install pygame

sudo apt install -y build-essential cmake libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev python3-dev python3-pip

pip install dlib       # takes a while to compile!
pip install face_recognition

# License plate recognition
sudo apt install tesseract-ocr
pip install pytesseract