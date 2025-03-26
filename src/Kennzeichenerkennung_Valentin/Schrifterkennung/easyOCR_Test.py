import easyocr
import cv2
import numpy as np
from openvino.runtime import Core
import os

os.chdir("C:/Users/Valentin.Talmon/PycharmProjects/YOLO/")
print(os.getcwd())

# Load OpenVINO model
core = Core()
model_xml = "src/openvino_model/license_plate_detector.xml"
model_bin = "src/openvino_model/license_plate_detector.bin"
model = core.read_model(model_xml)
compiled_model = core.compile_model(model)
output_layer = compiled_model.output(0)

cap = cv2.VideoCapture(0)  # 0 für die Standard-Webcam
cap.set(cv2.CAP_PROP_FPS, 5)

reader = easyocr.Reader(['en'])

# Initialize frame counter
frame_counter = 0
frame_interval = 30  # Process every 30th frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Ende des Videos

    # Increment frame counter
    frame_counter += 1

    # Only process every 30th frame
    if frame_counter % frame_interval == 0:
        # Prepare input for OpenVINO
        input_image = cv2.resize(frame, (640, 640))  # Adjust size based on your model requirements
        input_image = input_image.transpose((2, 0, 1))  # HWC to CHW
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype(np.float32) / 255.0  # Normalize to [0,1]

        # Run inference
        results = compiled_model([input_image])[output_layer]

        # Process detections (adjust based on your model's output format)
        for detection in results[0]:
            confidence = detection[4]

            if confidence > 0.4:
                # Get class id (assuming class scores start at index 5)
                class_id = np.argmax(detection[5:])

                if class_id == 0:  # Assuming 0 is license plate class
                    # Get bounding box coordinates
                    x1 = int(detection[0] * frame.shape[1])
                    y1 = int(detection[1] * frame.shape[0])
                    x2 = int(detection[2] * frame.shape[1])
                    y2 = int(detection[3] * frame.shape[0])

                    license_plate = frame[y1:y2, x1:x2]

                    # Make sure the cropped region is valid
                    if license_plate.size == 0:
                        continue

                    # Preprocessing
                    gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                    # OCR with EasyOCR
                    result = reader.readtext(gray)

                    for (bbox, text, prob) in result:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if prob > 0.9:
                            text_gesamt = " ".join([text for (_, text, _) in result])
                            print(f'Text: {text_gesamt}, Probability: {prob}')
                            prob_text = text_gesamt
                            # Draw license plate box and text
                            cv2.putText(frame, prob_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("Kennzeichen-Erkennung", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()