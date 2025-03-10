from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from utils import getCompleteSentece

import pickle
import base64

import mediapipe as mp
import numpy as np
import cv2

class ImageData(BaseModel):
    time: int
    data: str

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Load the model at startup using pickle
model_dict = pickle.load(open('model/model.p', 'rb'))
model = model_dict['model']

word_buffer = []
labels_dict = {'hello': 'hello', 'thank you': 'thank you', 'fine': 'fine',
               'good': 'good', 'you': 'you?', 'my name is': 'my name is', 'A': 'A', 'L': 'L', 'I': 'I'}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def getAuxNp(img):
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = img.shape

    # Convert the frame to RGB as required by MediaPipe
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image (optional)
            mp_drawing.draw_landmarks(
                img,  # Image to draw
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

    return data_aux


@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/postFrame")
async def post_frame(data: ImageData):

    b64_string = data.data[22:]

    image_str = base64.b64decode(b64_string)
    image_np = np.frombuffer(image_str, dtype=np.uint8)

    image = cv2.imdecode(image_np, flags=1)
    if image is None:
        return {"msg": "ERROR! can't create cv2 img"}
    
    data_aux = getAuxNp(image)

    if len(data_aux) == 42:
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict.get(prediction[0], 'unknown')
        word_buffer.append(predicted_character)
    else:
        print(f"Unexpected number of features: {len(data_aux)}")

    return {"msg": "we got em"}

@app.get("/getTranslation")
async def get_translation():
    print("request the translation")
    arr = clean_array(word_buffer)

    print(arr)
    msg = getCompleteSentece(arr)

    word_buffer = []
    return {"msg":msg}

@app.get("/clearBuffer")
async def clear_buffer():
    print("clear buffer")
    word_buffer = []

