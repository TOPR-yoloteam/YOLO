import cv2
import easyocr
import MNN
import numpy as np
import os

os.chdir("/home/jan/Downloads/YOLO-ValentinTestUmgebung/")
print(os.getcwd())

# Load MNN model - first convert your PT model to MNN format using MNN converter tools
interpreter = MNN.Interpreter("src/onnx_model/license_plate_detector.onnx")
session = interpreter.createSession()
input_tensor = interpreter.getSessionInput(session)

cap = cv2.VideoCapture(0)  # 0 for default webcam
cap.set(cv2.CAP_PROP_FPS, 5)

reader = easyocr.Reader(['en'])


# Function to process YOLO output
def process_output(output, conf_threshold=0.4):
    # Parse MNN output based on your YOLO model structure
    # This is a simplified example - you'll need to adjust based on your specific model output format
    boxes = []
    confidences = []
    class_ids = []

    # Process output tensors (this part depends on your specific YOLO model structure)
    # You'll need to extract boxes, confidences, and class IDs from the output

    # For example:
    for detection in output[0]:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            center_x = detection[0] * frame.shape[1]
            center_y = detection[1] * frame.shape[0]
            width = detection[2] * frame.shape[1]
            height = detection[3] * frame.shape[0]

            # Calculate top-left corner
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(x1 + width)
            y2 = int(y1 + height)

            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))
            class_ids.append(class_id)

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