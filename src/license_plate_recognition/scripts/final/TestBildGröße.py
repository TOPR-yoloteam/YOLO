import cv2

# Bild einlesen
img = cv2.imread('C:/Users/serha/PycharmProjects/YOLO/src/license_plate_recognition/scripts/final/data/ocr_images/plate_image_0_0.png')  # oder nimm dein Kamera-Frame hier

# Größe anzeigen
print("Bildgröße:", img.shape)