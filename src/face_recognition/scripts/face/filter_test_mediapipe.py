import os
import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Build absolute path to the sunglasses image relative to this script
base_dir = os.path.dirname(__file__)
glasses_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'img', 'sunglasses.png'))
original_glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)

# Check if the image was loaded successfully
if original_glasses is None:
    print(f"Error: Could not load image.\nChecked path: {glasses_path}")
    exit()
else:
    print(f"Image loaded successfully: {glasses_path}")

# Start webcam capture
cap = cv2.VideoCapture(0)

# Variables for smoothing
prev_angle = 0
prev_eye_distance = 0
smoothing_factor = 0.7  # Higher values mean more smoothing


def rotate_and_scale_glasses(image, angle, scale):
    """
    Resize and rotate the glasses image without cutting it off,
    preserving transparency.
    """
    # Calculate new dimensions
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)

    if new_width <= 0 or new_height <= 0:
        return None

    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Center of the resized image
    center = (new_width // 2, new_height // 2)

    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding dimensions
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    bound_w = int(new_height * sin + new_width * cos)
    bound_h = int(new_height * cos + new_width * sin)

    # Adjust rotation matrix
    rotation_matrix[0, 2] += (bound_w / 2) - center[0]
    rotation_matrix[1, 2] += (bound_h / 2) - center[1]

    # Create a transparent background
    rotated = cv2.warpAffine(
        resized,
        rotation_matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)  # FULLY transparent background
    )

    return rotated


def apply_glasses(frame, glasses, position):
    if glasses is None:
        return frame

    x, y = position
    frame_h, frame_w = frame.shape[:2]
    glasses_h, glasses_w = glasses.shape[:2]

    # Calculate valid overlay regions
    frame_x1 = max(0, x)
    frame_y1 = max(0, y)
    frame_x2 = min(frame_w, x + glasses_w)
    frame_y2 = min(frame_h, y + glasses_h)

    glasses_x1 = frame_x1 - x
    glasses_y1 = frame_y1 - y
    glasses_x2 = glasses_x1 + (frame_x2 - frame_x1)
    glasses_y2 = glasses_y1 + (frame_y2 - frame_y1)

    # Check if regions are valid
    if (glasses_x1 >= glasses_w or glasses_y1 >= glasses_h or
            glasses_x2 <= 0 or glasses_y2 <= 0 or
            frame_x1 >= frame_w or frame_y1 >= frame_h or
            frame_x2 <= 0 or frame_y2 <= 0):
        return frame

    # Make sure regions are within bounds
    glasses_x1, glasses_y1 = max(0, glasses_x1), max(0, glasses_y1)
    glasses_x2, glasses_y2 = min(glasses_w, glasses_x2), min(glasses_h, glasses_y2)
    frame_x1, frame_y1 = max(0, frame_x1), max(0, frame_y1)
    frame_x2, frame_y2 = min(frame_w, frame_x2), min(frame_h, frame_y2)

    # Adjust for potential size mismatches
    w_diff = (frame_x2 - frame_x1) - (glasses_x2 - glasses_x1)
    h_diff = (frame_y2 - frame_y1) - (glasses_y2 - glasses_y1)

    if w_diff > 0:
        glasses_x2 = min(glasses_w, glasses_x2 + w_diff)
    elif w_diff < 0:
        frame_x2 = min(frame_w, frame_x2 - w_diff)

    if h_diff > 0:
        glasses_y2 = min(glasses_h, glasses_y2 + h_diff)
    elif h_diff < 0:
        frame_y2 = min(frame_h, frame_y2 - h_diff)

    result = frame.copy()

    try:
        glasses_region = glasses[glasses_y1:glasses_y2, glasses_x1:glasses_x2]
        frame_region = result[frame_y1:frame_y2, frame_x1:frame_x2]

        # Extract alpha channel
        alpha = glasses_region[:, :, 3] / 255.0  # (H, W)

        # Only update pixels where alpha > 0.1
        mask = alpha > 0.1

        # For each color channel separately
        for c in range(3):  # B, G, R channels
            frame_region[:, :, c][mask] = (glasses_region[:, :, c][mask] * alpha[mask] +
                                           frame_region[:, :, c][mask] * (1 - alpha[mask]))

        result[frame_y1:frame_y2, frame_x1:frame_x2] = frame_region

        return result

    except Exception as e:
        print(f"Error applying glasses: {e}")
        return frame



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Always work on a FRESH copy of the webcam frame
    output_frame = frame.copy()

    h, w = output_frame.shape[:2]
    rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get key facial landmarks
            left_eye = face_landmarks.landmark[33]  # Left eye outer corner
            right_eye = face_landmarks.landmark[263]  # Right eye outer corner

            # Convert to pixel coordinates
            left_x, left_y = int(left_eye.x * w), int(left_eye.y * h)
            right_x, right_y = int(right_eye.x * w), int(right_eye.y * h)

            # Calculate center point between eyes
            center_x = (left_x + right_x) // 2
            center_y = (left_y + right_y) // 2

            # Calculate distance and rotation
            dx = right_x - left_x
            dy = right_y - left_y
            eye_distance = math.sqrt(dx * dx + dy * dy)

            # Smoothing (optional, for stability)
            eye_distance = prev_eye_distance * smoothing_factor + eye_distance * (1 - smoothing_factor)
            prev_eye_distance = eye_distance

            scale_factor = eye_distance / original_glasses.shape[1] * 1.65

            angle_radians = math.atan2(dy, dx)
            angle_degrees = math.degrees(angle_radians)
            angle_degrees = -angle_degrees

            angle_degrees = prev_angle * smoothing_factor + angle_degrees * (1 - smoothing_factor)
            prev_angle = angle_degrees

            # Rotate and scale glasses
            processed_glasses = rotate_and_scale_glasses(original_glasses, angle_degrees, scale_factor)

            if processed_glasses is not None:
                glasses_h, glasses_w = processed_glasses.shape[:2]
                x_pos = center_x - glasses_w // 2
                y_pos = center_y - glasses_h // 2

                # Apply glasses
                output_frame = apply_glasses(output_frame, processed_glasses, (x_pos, y_pos))
            break  # Only process first face

    # Show the result
    cv2.imshow('Sunglasses Filter', output_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()