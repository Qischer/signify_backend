import cv2
import time
import mediapipe as mp
import numpy as np
import pickle

# Load the model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary to map predictions to words

labels_dict = {'hello': 'hello', 'thank you': 'thank you', 'fine': 'fine',
               'good': 'good', 'you': 'you?', 'my name is': 'my name is', 'A': 'A', 'L': 'L', 'I': 'I'}


def recognize_word(letter_array, threshold_divider):
    runs = []
    current_letter = letter_array[0]
    current_length = 1

    # Step 1: Group consecutive letters into runs
    for i in range(1, len(letter_array)):
        if letter_array[i] == current_letter:
            current_length += 1
        else:
            runs.append({'letter': current_letter, 'length': current_length})
            current_letter = letter_array[i]
            current_length = 1
    runs.append({'letter': current_letter, 'length': current_length})

    # Step 2: Filter out short misrecognized runs
    filtered_runs = []
    for i in range(len(runs)):
        prev_length = runs[i - 1]['length'] if i > 0 else runs[i + 1]['length']
        next_length = runs[i +
                           1]['length'] if i < len(runs) - 1 else runs[i - 1]['length']
        avg_adjacent_length = (prev_length + next_length) / 2
        threshold = avg_adjacent_length / threshold_divider

        if runs[i]['length'] >= threshold:
            filtered_runs.append(runs[i])

    # Step 3: Construct the final word
    word = ' '.join([run['letter'] for run in filtered_runs])
    return word


def clean_array(arr):
    result = []
    i = 0

    while i < len(arr):
        count = 1
        while i + 1 < len(arr) and arr[i] == arr[i + 1]:
            count += 1
            i += 1

        if count > 5:
            result.append(arr[i])
        i += 1  # Move to the next distinct element

    return result


# Parameters for video capture
frames_per_second = 10
duration = 20  # Capture for 10 seconds
frame_interval = 1 / frames_per_second  # Time between frames

# Initialize video capture (0 for the default webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open webcam")

# List to store captured frames
frames = []

# Capture frames for the given duration
start_time = time.time()
while time.time() - start_time < duration:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frames.append(frame)  # Store the frame
    cv2.imshow('Frame', frame)  # Display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(frame_interval)  # Wait for the next frame

cap.release()
cv2.destroyAllWindows()

print(f"Number of frames captured: {len(frames)}")

# Process each captured frame
predictions = []
for frame in frames:
    if frame is None:
        print("Error: Could not open frame.")
        continue

    data_aux = []  # Store the landmarks data
    x_ = []
    y_ = []

    H, W, _ = frame.shape

    # Convert frame to RGB (MediaPipe requires RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract landmarks and calculate relative positions
        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)

        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))  # Normalize x-coordinates
            data_aux.append(landmark.y - min(y_))  # Normalize y-coordinates

        # Ensure the feature size matches the model's expectation (42 features)
        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(prediction[0], 'unknown')
            predictions.append(predicted_character)
        else:
            print(f"Unexpected number of features: {len(data_aux)}")
    else:
        print("No hand landmarks detected.")

print(predictions)
print(clean_array(predictions) if predictions else 'nothing!!!!')
