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
def process_output(output_data, conf_threshold=0.4, frame=None):
    # Get output shape
    output_shape = output_data.shape
    print(f"Output shape: {output_shape}")

    boxes = []
    confidences = []
    class_ids = []

    # Reshape output based on your model's actual output format
    # YOLO models typically output in format [batch, num_detections, attributes]
    # where attributes are usually [x, y, w, h, confidence, class_prob1, class_prob2, ...]

    # This is a generic approach - adjust based on your actual output format:
    if len(output_shape) == 1:
        # Try reshaping based on common YOLO output formats
        try:
            num_detections = int(output_shape[0] / 85)  # For YOLOv5/v8 (85 = 4 box + 1 obj + 80 classes)
            detections = output_data.reshape(num_detections, 85)
        except:
            print("Cannot reshape output data. Check your model's output format.")
            return [], [], []
    else:
        # For models that directly output [num_detections, attributes]
        detections = output_data

    # Process detections
    for i in range(len(detections)):
        detection = detections[i]

        # Extract box coordinates and confidence
        # Adjust indices based on your model's output format
        try:
            x, y, w, h = detection[0:4]
            confidence = detection[4]

            # Class probabilities start from index 5
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]

            # Only process if object confidence is above threshold
            if confidence > conf_threshold:
                if frame is not None:
                    # Convert from normalized coordinates to pixel coordinates
                    img_h, img_w = frame.shape[:2]
                    x1 = max(0, int((x - w / 2) * img_w))
                    y1 = max(0, int((y - h / 2) * img_h))
                    x2 = min(img_w, int((x + w / 2) * img_w))
                    y2 = min(img_h, int((y + h / 2) * img_h))

                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence * class_score))
                    class_ids.append(class_id)
        except IndexError:
            print(f"Detection format unexpected: {detection}")
            continue

    return boxes, confidences, class_ids


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
    boxes, confidences, class_ids = process_output(output_data)

    # Process detections
    for i, box in enumerate(boxes):
        if class_ids[i] == 0:  # License plate class
            if confidences[i] > 0.4:
                x1, y1, x2, y2 = box

                # Extract license plate region
                license_plate = frame[y1:y2, x1:x2]

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
                        cv2.putText(frame, prob_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show video with detected license plates
    cv2.imshow("License Plate Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()