import glob

import easyocr
import cv2
import os
import imutils
import numpy as np

current_dir_toggle = False

def toggle_working_directory():
    """
    Wechselt zwischen zwei Arbeitsverzeichnissen basierend auf dem Status der globalen Variable current_dir_toggle.
    """
    global current_dir_toggle  # Zugriff auf die globale Variable zum Ändern
    if current_dir_toggle:
        os.chdir("/src/img")
        print("Wechsel zum Verzeichnis: img")
    else:
        os.chdir(
            "/src/img/output")
        print("Wechsel zum Verzeichnis: output")

    # Status toggeln
    current_dir_toggle = not current_dir_toggle


def getImages():
    path = "src/img"
    for file_name in os.listdir(path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            preProcessing(file_name)


def preProcessing(fileName):


    image = cv2.imread(fileName)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, rectKern)
    toggle_working_directory()
    cv2.imwrite("blackhat_" + fileName, blackhat)
    toggle_working_directory()
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #toggle_working_directory()
    #cv2.imwrite("gray_" + fileName, thresh)
    #toggle_working_directory()
#
    #dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    #dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    #dist = (dist * 255).astype("uint8")
    #toggle_working_directory()
    #cv2.imwrite("dist_" + fileName, dist)
    #toggle_working_directory()
#
    #dist = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #toggle_working_directory()
    #cv2.imwrite("dist_thresh" + fileName, dist)
    #toggle_working_directory()


os.chdir("/src/img")

getImages()






















