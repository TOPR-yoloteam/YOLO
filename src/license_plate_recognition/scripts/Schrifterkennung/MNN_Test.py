import cv2
import easyocr
import MNN
import numpy as np
import os

os.chdir("/home/jan/Downloads/YOLO-ValentinTestUmgebung/")
print(os.getcwd())

# Load MNN model - first convert your PT model to MNN format using MNN converter tools
interpreter = MNN.Interpreter("src/mnn_model/license_plate_detector.mnn")
session = interpreter.createSession()
input_tensor = interpreter.getSessionInput(session)

cap = cv2.VideoCapture(0)  # 0 for default webcam
cap.set(cv2.CAP_PROP_FPS, 5)

reader = easyocr.Reader(['en'])


# Function to process YOLO output
# Add this after creating the session
print("Model input shape:", input_tensor.getShape())
for name in interpreter.getSessionOutputAll(session).keys():
    out = interpreter.getSessionOutput(session, name)
    print(f"Output '{name}' shape:", out.getShape())


# Modify your process_output function to analyze the output data
def process_output(output_data, conf_threshold=0.4, frame=None):
    # Reshape to 8400 detections with 5 values each (x, y, w, h, conf)
    detections = output_data.reshape(8400, 5)

    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        confidence = detection[4]
        if confidence > conf_threshold and frame is not None:
            x, y, w, h = detection[0:4]
            # Default to class 0 (license plate)
            class_id = 0

            img_h, img_w = frame.shape[:2]
            x1 = max(0, int((x - w / 2) * img_w))
            y1 = max(0, int((y - h / 2) * img_h))
            x2 = min(img_w, int((x + w / 2) * img_w))
            y2 = min(img_h, int((y + h / 2) * img_h))

            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    return boxes, confidences, class_ids


def apply_nms(boxes, confidences, class_ids, iou_threshold=0.5):
    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, iou_threshold)

    if len(indices) > 0:
        return [boxes[i] for i in indices], [confidences[i] for i in indices], [class_ids[i] for i in indices]
    return [], [], []


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess image for MNN
    resized = cv2.resize(frame, (640, 640))  # Adjust size to match your model's input
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # Normalize to 0-1

    # Convert to MNN tensor format and reshape
    tmp_input = MNN.Tensor((1, 3, 640, 640), MNN.Halide_Type_Float, img.transpose(2, 0, 1).reshape(1, 3, 640, 640),
                           MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)

    # Run inference
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)

    # Get output
    tmp_output = MNN.Tensor((output_tensor.getShape()), MNN.Halide_Type_Float,
                            np.zeros(output_tensor.getShape(), dtype=np.float32), MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(tmp_output)
    output_data = np.array(tmp_output.getData())

    # Process output to get boxes, confidences, and class IDs
    # After process_output call:
    boxes, confidences, class_ids = process_output(output_data, conf_threshold=0.4, frame=frame)
    boxes, confidences, class_ids = apply_nms(boxes, confidences, class_ids)

    # Process detections
    for i, box in enumerate(boxes):
        if class_ids[i] == 0:  # License plate class
            if confidences[i] > 0.4:
                x1, y1, x2, y2 = box

                # Make sure coordinates are valid
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue

                # Extract license plate region
                license_plate = frame[y1:y2, x1:x2]

                # Check if license plate region is empty
                if license_plate.size == 0 or license_plate.shape[0] <= 0 or license_plate.shape[1] <= 0:
                    continue

                # Preprocessing
                try:
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
                            cv2.putText(frame, prob_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error processing license plate: {e}")
                    continue

    # Show video with detected license plates
    cv2.imshow("License Plate Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()