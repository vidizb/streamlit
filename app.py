import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import mediapipe as mp
import numpy as np
from PIL import Image

mp_hands = mp.solutions.hands
mp_hands_connections = mp.solutions.hands_connections
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils 

connections = {
    'HAND_CONNECTIONS': mp_hands_connections.HAND_CONNECTIONS,
    'HAND_PALM_CONNECTIONS': mp_hands_connections.HAND_PALM_CONNECTIONS,
    'HAND_THUMB_CONNECTIONS': mp_hands_connections.HAND_THUMB_CONNECTIONS,
    'HAND_INDEX_FINGER_CONNECTIONS': mp_hands_connections.HAND_INDEX_FINGER_CONNECTIONS,
    'HAND_MIDDLE_FINGER_CONNECTIONS': mp_hands_connections.HAND_MIDDLE_FINGER_CONNECTIONS,
    'HAND_RING_FINGER_CONNECTIONS': mp_hands_connections.HAND_RING_FINGER_CONNECTIONS,
    'HAND_PINKY_FINGER_CONNECTIONS': mp_hands_connections.HAND_PINKY_FINGER_CONNECTIONS,
}

def process_hands(img):
  results = hands.process(img)
  output_img = img if draw_background else np.zeros_like(img)  
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      mp_draw.draw_landmarks(output_img, hand_landmarks, connections[selected_connection])    
  return output_img

st.title('Hand & Finger Tracking')
st.markdown("This is a demo of hand and finger tracking using [Google's MediaPipe](https://google.github.io/mediapipe/solutions/hands.html).")

col1, col2 = st.columns(2)

with col1:
  picture = st.camera_input("Take a picture with one or both hands in the shot")
  draw_background = st.checkbox("Draw background", value=True)
  selected_connection = st.selectbox("Select connections to draw", list(connections.keys()))

with col2:
  if picture is not None:
    img = Image.open(picture)
    img_array = np.array(img)
    processed_img = process_hands(img_array)
    st.image(processed_img)


