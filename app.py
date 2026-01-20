import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import urllib.parse
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------- Page config ----------------
st.set_page_config(page_title="Emotion Based Music Recommender")
st.title("🎵 Emotion Based Music Recommender")

# ---------------- Load emotion model ----------------
@st.cache_resource
def load_emotion_model():
    return load_model("CNN_Model.h5")

model = load_emotion_model()

Emotion_Classes = [
    "Angry", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise"
]

# ---------------- Load music dataset ----------------
@st.cache_data
def load_music_data():
    df = pd.read_csv("spotify_tracks.csv")
    df = df.dropna(subset=["track_name", "artists", "valence", "energy"])
    return df

music_df = load_music_data()

# ---------------- Emotion → feature mapping ----------------
emotion_ranges = {
    "Happy": {"valence": (0.7, 1.0), "energy": (0.6, 1.0)},
    "Sad": {"valence": (0.0, 0.3), "energy": (0.0, 0.4)},
    "Angry": {"valence": (0.0, 0.4), "energy": (0.7, 1.0)},
    "Neutral": {"valence": (0.4, 0.6), "energy": (0.4, 0.6)},
    "Fear": {"valence": (0.2, 0.4), "energy": (0.3, 0.5)},
    "Surprise": {"valence": (0.6, 0.8), "energy": (0.5, 0.7)},
    "Disgust": {"valence": (0.0, 0.3), "energy": (0.0, 0.3)}
}

# ---------------- Song recommender ----------------
def recommend_songs(df, emotion, n=6):
    r = emotion_ranges[emotion]

    filtered = df[
        (df["valence"] >= r["valence"][0]) &
        (df["valence"] <= r["valence"][1]) &
        (df["energy"] >= r["energy"][0]) &
        (df["energy"] <= r["energy"][1])
    ]

    if len(filtered) == 0:
        return pd.DataFrame()

    return filtered.sample(n=min(n, len(filtered)))

# ---------------- YouTube redirect ----------------
def youtube_link(track, artist):
    query = f"{track} {artist} official audio"
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)

def spotify_link(track_id):
    return f"https://open.spotify.com/track/{track_id}"


# ---------------- MediaPipe Face Detection ----------------
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# ---------------- Camera input ----------------
img_file = st.camera_input("Take a photo")

current_emotion = None

if img_file is not None:
    image = Image.open(img_file)
    frame = np.array(image)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    if results.detections:
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(box.xmin * w)
            y = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)

            face = frame[y:y + bh, x:x + bw]
            if face.size == 0:
                continue

            # FER2013 preprocessing
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized / 255.0
            rgb_like = np.stack((normalized,) * 3, axis=-1)
            input_face = rgb_like.reshape(1, 48, 48, 3)

            prediction = model.predict(input_face, verbose=0)
            emotion_index = np.argmax(prediction)
            current_emotion = Emotion_Classes[emotion_index]

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                frame,
                current_emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        st.image(frame, caption=f"Detected Emotion: {current_emotion}", channels="BGR")

    else:
        st.warning("No face detected. Try again.")

# ---------------- Recommend songs ----------------
st.divider()

if st.button("🎧 Recommend Songs"):
    if current_emotion is None:
        st.warning("Please capture an image first.")
    else:
        st.subheader(f"Recommended Songs for {current_emotion}")

        recommended = recommend_songs(music_df, current_emotion)

        if recommended.empty:
            st.info("No matching songs found.")
        else:
            for _, row in recommended.iterrows():
                yt = youtube_link(row["track_name"], row["artists"])
                sp = spotify_link(row["track_id"])

                st.markdown(
                    f"🎵 **{row['track_name']}** – {row['artists']}  \n"
                    f"[▶ Play on YouTube]({yt}) | [▶ Play on Spotify]({sp})"
                )
