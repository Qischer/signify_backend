import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary to map prediction to letters
labels_dict = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G',
 'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N',
 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U',
 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z'}


def process_image(image_path):
    try:
        # Load the PNG image
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error: Could not open or find the image {image_path}.")
            return

        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape

        # Convert the frame to RGB as required by MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image (optional)
                mp_drawing.draw_landmarks(
                    frame,  # Image to draw
                    hand_landmarks,  # Model output
                    mp_hands.HAND_CONNECTIONS,  # Hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract landmarks and calculate relative positions
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Predict the character using the model
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[prediction[0]]
            print(f"Predicted character: {predicted_character}")

        else:
            print("No hand landmarks detected.")

    except Exception as e:
        print(f"An error occurred: {e}")

process_image("test_A.png")
process_image("test_B.png")

# Clean up
hands.close()
