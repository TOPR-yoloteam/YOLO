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
    output_shape = output_data.shape
    print(f"Output shape: {output_data.shape}")

    # Print sample values to understand the format
    print(f"Sample values (first 10):", output_data[:10])
    print(f"Sample values (last 10):", output_data[-10:])

    # For YOLOv8 format, try 8400 detections with 5 values each
    try:
        # Common YOLOv8 output formats
        detections_per_shape = [
            (8400, 5),  # 8400 boxes with 5 values (x, y, w, h, conf)
            (7000, 6),  # 7000 boxes with 6 values
            (84, 500),  # 84 classes with 500 values per class
            (42000, 1)  # Flat tensor
        ]

        for num_det, values_per_det in detections_per_shape:
            if output_shape[0] == num_det * values_per_det:
                print(f"Trying reshape to {num_det} detections with {values_per_det} values each")
                detections = output_data.reshape(num_det, values_per_det)

                # Check if this format makes sense (non-zero values)
                non_zero = np.count_nonzero(detections[:, 4])  # Check confidence values
                print(f"Found {non_zero} detections with non-zero confidence")

                if non_zero > 0:
                    print("This format seems correct!")
                    # Process using this format
                    boxes = []
                    confidences = []
                    class_ids = []

                    for detection in detections:
                        confidence = detection[4]
                        if confidence > conf_threshold and frame is not None:
                            x, y, w, h = detection[0:4]
                            class_id = 0
                            if values_per_det > 5:
                                class_id = int(detection[5])

                            img_h, img_w = frame.shape[:2]
                            x1 = max(0, int((x - w / 2) * img_w))
                            y1 = max(0, int((y - h / 2) * img_h))
                            x2 = min(img_w, int((x + w / 2) * img_w))
                            y2 = min(img_h, int((y + h / 2) * img_h))

                            boxes.append([x1, y1, x2, y2])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                    return boxes, confidences, class_ids
    except Exception as e:
        print(f"Error processing output: {e}")

    print("Could not determine correct output format")
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
    boxes, confidences, class_ids = process_output(output_data, conf_threshold=0.4, frame=frame)

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