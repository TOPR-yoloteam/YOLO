import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Standard value for height calibration
DEFAULT_HEIGHT_CM = 170  # Average body height in cm

# Conversion factor - adjusted through calibration
conversion_factor = None

def calculate_pixel_distance(landmarks, point1_index, point2_index, image_height, image_width):
    """Calculates the distance between two landmarks in pixels"""
    point1 = landmarks.landmark[point1_index]
    point2 = landmarks.landmark[point2_index]
    
    # Convert normalized coordinates to pixel coordinates
    x1, y1 = int(point1.x * image_width), int(point1.y * image_height)
    x2, y2 = int(point2.x * image_width), int(point2.y * image_height)
    
    # Euclidean distance
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def estimate_height(landmarks, image_height, image_width, factor=None):
    """Estimates body height based on landmarks"""
    # Check if important landmarks are visible
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_heel = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
    right_heel = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
    
    # Choose the more visible foot
    if left_heel.visibility > right_heel.visibility:
        heel_index = mp_pose.PoseLandmark.LEFT_HEEL
        heel = left_heel
    else:
        heel_index = mp_pose.PoseLandmark.RIGHT_HEEL
        heel = right_heel
    
    # Check if important points are sufficiently visible
    if nose.visibility < 0.5 or heel.visibility < 0.5:
        return 0
    
    # Calculate pixel distance from head to foot
    pixel_height = calculate_pixel_distance(landmarks, 
                                         mp_pose.PoseLandmark.NOSE, 
                                         heel_index, 
                                         image_height, image_width)
    
    # If no factor is provided, calibration is needed
    if factor is None:
        return pixel_height
    
    # Convert to cm using factor
    height_cm = pixel_height * factor
    return int(height_cm)

def calibrate(pixel_height, actual_height_cm):
    """Calculates the conversion factor from pixels to cm"""
    return actual_height_cm / pixel_height

# Status variables
is_calibrated = False
calibration_active = False
my_height_cm = DEFAULT_HEIGHT_CM

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Mirror image for more intuitive display
    frame = cv2.flip(frame, 1)
    image_height, image_width, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(rgb_frame)

    # Display status information
    if not is_calibrated:
        cv2.putText(frame, "Not calibrated - Press 'k' to calibrate",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if calibration_active:
        cv2.putText(frame, f"Calibration active - Height: {my_height_cm} cm",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        cv2.putText(frame, "Stand straight and press 'k' again",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    # When body landmarks are detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Estimate body height (in pixels if not calibrated)
        pixel_height = estimate_height(results.pose_landmarks, image_height, image_width)
        
        if pixel_height > 0:
            # If in calibration mode and key 'k' is pressed again
            if calibration_active and cv2.waitKey(1) & 0xFF == ord('k'):
                conversion_factor = calibrate(pixel_height, my_height_cm)
                calibration_active = False
                is_calibrated = True
                print(f"Calibrated with factor: {conversion_factor}")
            
            # If calibrated, calculate and display height in cm
            if is_calibrated:
                height_cm = estimate_height(results.pose_landmarks, image_height, 
                                           image_width, conversion_factor)
                cv2.putText(frame, f"Body height: {height_cm} cm",
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Visualize the measurement line
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
            
            if nose.visibility > 0.5 and heel.visibility > 0.5:
                nose_px = int(nose.x * image_width), int(nose.y * image_height)
                heel_px = int(heel.x * image_width), int(heel.y * image_height)
                cv2.line(frame, nose_px, heel_px, (0, 255, 0), 2)
                
                # Show pixel height if not calibrated
                if not is_calibrated:
                    cv2.putText(frame, f"Pixel height: {int(pixel_height)}",
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display results
    cv2.imshow("Body Height Measurement", frame)

    # Process keyboard inputs
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('k') and not calibration_active:
        calibration_active = True
        # Here we could optionally add code to set the current height
    elif key == ord('+') and calibration_active:
        my_height_cm += 1
    elif key == ord('-') and calibration_active:
        my_height_cm = max(100, my_height_cm - 1)

cap.release()
cv2.destroyAllWindows()
