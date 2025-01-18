import mediapipe as mp
import cv2
import os
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Directory containing sign images
data_dir = './data'

data = []
labels = []

# Process each label directory
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    if not os.path.isdir(label_dir):
        continue

    # Process each image in the label directory
    for img_file in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Convert image to RGB
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgrgb)

        # Extract hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                rawdata = []
                for landmark in hand_landmarks.landmark:
                    rawdata.append(landmark.x)
                    rawdata.append(landmark.y)
                data.append(rawdata)
                labels.append(label)

# Save data to a file
with open('sign_data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Feature extraction completed.")
