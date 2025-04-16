import easyocr
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext('C:/Users/serha/PycharmProjects/YOLO/src/license_plate_recognition/scripts/final/data/ocr_images/plate_image_0_0.png', detail=1)

for res in results:
    print(res)
