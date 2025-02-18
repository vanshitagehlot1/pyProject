import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands and Drawing tools
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load emoji images
heart_img = cv2.imread("G:\\mca\\sem 2\\python\\projects\\heart.png", cv2.IMREAD_UNCHANGED)
balloon_img = cv2.imread("G:\\mca\\sem 2\\python\\projects\\balloon.png", cv2.IMREAD_UNCHANGED)
thumbs_up_img = cv2.imread("G:\\mca\\sem 2\\python\\projects\\thumbsup.png", cv2.IMREAD_UNCHANGED)

# Check if images are loaded properly
if heart_img is None or balloon_img is None or thumbs_up_img is None:
    print("Error: One or more images not loaded. Check the file paths.")
else:
    print("Images loaded successfully.")

# Parameters for emoji display
initial_scale = 0.1  # Starting scale factor for the emoji animation
max_scale = 1.5      # Maximum scale factor for the emoji animation
scale_factor = initial_scale
scaling_up = True
show_emoji = False
emoji_img = None
balloon_y_offset = 0  # Offset for the balloon animation

# Function to detect heart gesture from a single hand
def detect_heart_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    if (abs(thumb_tip.x - middle_tip.x) < 0.05 and
        abs(thumb_tip.y - middle_tip.y) < 0.05 and
        abs(index_tip.x - middle_tip.x) < 0.05 and
        abs(index_tip.y - middle_tip.y) < 0.05):
        return True
    return False

# Function to detect heart gesture from both hands
def detect_full_heart_gesture(hand_landmarks_list):
    if len(hand_landmarks_list) < 2:
        return False
    
    # Detect gesture for both hands
    hand1 = hand_landmarks_list[0]
    hand2 = hand_landmarks_list[1]
    
    hand1_detected = detect_heart_gesture(hand1)
    hand2_detected = detect_heart_gesture(hand2)

    return hand1_detected and hand2_detected

# Function to detect victory gesture (peace sign)
def detect_victory_gesture(hand_landmarks):
    # Check the landmarks for the peace sign (index and middle fingers extended)
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

    if (index_tip.y < middle_tip.y and
        abs(index_tip.x - middle_tip.x) < 0.1 and
        thumb_tip.y > thumb_ip.y and
        thumb_tip.y > index_tip.y):
        return True
    return False

# Function to detect thumbs-up gesture
def detect_thumbs_up_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    if (thumb_tip.y < index_tip.y and
        thumb_tip.y < middle_tip.y and
        thumb_tip.y < ring_tip.y and
        thumb_tip.y < pinky_tip.y and
        abs(index_tip.y - middle_tip.y) > 0.1 and
        abs(index_tip.y - ring_tip.y) > 0.1):
        return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_detected = False
    emoji_img = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if detect_full_heart_gesture(results.multi_hand_landmarks):
                gesture_detected = True
                emoji_img = heart_img
                break
            elif detect_victory_gesture(hand_landmarks):
                gesture_detected = True
                emoji_img = balloon_img
                break
            elif detect_thumbs_up_gesture(hand_landmarks):
                gesture_detected = True
                emoji_img = thumbs_up_img
                break

        if gesture_detected:
            show_emoji = True
            if scale_factor == initial_scale:  # Start scaling up animation when gesture is detected
                scaling_up = True
        else:
            if not scaling_up:  # Only reset scale_factor if not scaling up
                scale_factor = initial_scale
                show_emoji = False

    if show_emoji and emoji_img is not None:
        h, w, _ = emoji_img.shape

        # Scaling animation logic for iPhone-like pop effect
        if scaling_up:
            scale_factor += 0.15  # Rapidly scale up
            if scale_factor >= max_scale:
                scaling_up = False

        # Resize the emoji image according to the scale factor
        resized_emoji = cv2.resize(emoji_img, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_LINEAR)

        # Calculate new position to center the resized image or for balloon animation
        if emoji_img is balloon_img:
            y_offset = balloon_y_offset
            balloon_y_offset += 5
            if balloon_y_offset > frame.shape[0]:
                balloon_y_offset = 0
        else:
            y_offset = (frame.shape[0] - resized_emoji.shape[0]) // 2

        x_offset = (frame.shape[1] - resized_emoji.shape[1]) // 2

        # Ensure the resized emoji image fits within the frame
        x_offset = max(0, min(x_offset, frame.shape[1] - resized_emoji.shape[1]))
        y_offset = max(0, min(y_offset, frame.shape[0] - resized_emoji.shape[0]))

        # Prepare to blend the resized emoji image with the frame
        b, g, r, a = cv2.split(resized_emoji)
        overlay = cv2.merge((b, g, r)).astype('float32')
        mask = cv2.merge((a, a, a)).astype('float32') / 255.0

        # Extract the ROI from the frame
        roi = frame[y_offset:y_offset + resized_emoji.shape[0], x_offset:x_offset + resized_emoji.shape[1]].astype('float32')

        # Ensure that the ROI and resized emoji have the same size
        if roi.shape[:2] == resized_emoji.shape[:2]:
            blended = cv2.multiply(overlay, mask) + cv2.multiply(roi, 1.0 - mask)
            frame[y_offset:y_offset + resized_emoji.shape[0], x_offset:x_offset + resized_emoji.shape[1]] = blended.clip(0, 255).astype('uint8')

    cv2.imshow('Hand Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
