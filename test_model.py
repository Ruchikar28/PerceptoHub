import cv2
import numpy as np
import mediapipe as mp
# import joblib
import pickle

# # Load the trained Random Forest model
# rf_model = joblib.load('random_forest_model.pkl')
# print("Model loaded successfully.")
# Load the saved model
with open('svm_sign_detection_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)
print("Model loaded successfully.")


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Start the webcam feed
cap = cv2.VideoCapture(0)

print("Starting real-time testing. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Normalize landmarks
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            min_x, min_y = min(x_coords), min(y_coords)

            normalized_landmarks = []
            for i in range(len(hand_landmarks.landmark)):
                normalized_landmarks.append(hand_landmarks.landmark[i].x - min_x)
                normalized_landmarks.append(hand_landmarks.landmark[i].y - min_y)

            # Predict using the model
            features = np.array(normalized_landmarks).reshape(1, -1)
            prediction = rf_model.predict(features)

            # Display the predicted label
            label = prediction[0]
            cv2.putText(frame, f"Prediction: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video feed with predictions
    cv2.imshow("Sign Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
