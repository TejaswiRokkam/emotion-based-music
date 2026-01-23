import os

# Force headless mode (fixes libGL.so.1 error on Streamlit Cloud)
os.environ["MPLBACKEND"] = "Agg"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
import sqlite3
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import urllib.parse
from tensorflow.keras.models import load_model
from PIL import Image
from datetime import datetime

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "auth"   # auth | preferences | camera

if "user" not in st.session_state:
    st.session_state.user = None

# ---------------- DATABASE ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Emotion Based Music Recommender")

# =====================================================
# ================= AUTH PAGE =========================
# =====================================================
if st.session_state.page == "auth":
    col_l, col_c, col_r = st.columns([1, 4, 1])

    with col_c:
        st.markdown(
            "<h2 style='white-space: nowrap;'>🎵 Emotion Based Music Recommender</h2>",
            unsafe_allow_html=True
        )
        st.markdown("### Welcome")

        tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Continue as Guest"])

        # -------- LOGIN --------
        with tab1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login", use_container_width=True):
                c.execute(
                    "SELECT id FROM users WHERE username=? AND password=?",
                    (username, password)
                )
                row = c.fetchone()

                if row:
                    st.session_state.user = row[0]

                    # Check if preferences already exist
                    c.execute(
                        "SELECT 1 FROM preferences WHERE user_id=?",
                        (row[0],)
                    )
                    pref_exists = c.fetchone()

                    st.session_state.page = "camera" if pref_exists else "preferences"
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        # -------- SIGN UP --------
        with tab2:
            new_user = st.text_input("Choose Username")
            new_pass = st.text_input("Choose Password", type="password")

            if st.button("Create Account", use_container_width=True):
                c.execute("SELECT id FROM users WHERE username=?", (new_user,))
                if c.fetchone():
                    st.error("Username already exists")
                else:
                    c.execute(
                        "INSERT INTO users (username, password) VALUES (?, ?)",
                        (new_user, new_pass)
                    )
                    conn.commit()
                    st.success("Account created. Please login.")

        # -------- GUEST --------
        with tab3:
            st.info("Continue without an account. Preferences will not be saved.")
            if st.button("Continue as Guest", use_container_width=True):
                st.session_state.user = None
                st.session_state.page = "camera"
                st.rerun()

# =====================================================
# ================= PREFERENCES PAGE ==================
# =====================================================
elif st.session_state.page == "preferences":
    col_l, col_c, col_r = st.columns([1, 4, 1])

    with col_r:
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.page = "auth"
            st.rerun()

    with col_c:
        st.markdown("## 🎯 Tell us your preferences (one time)")

        sad_pref = st.selectbox(
            "When you feel sad, what kind of songs do you prefer?",
            ["sad", "uplifting", "upbeat"]
        )

        angry_pref = st.selectbox(
            "When you feel angry, what kind of songs do you prefer?",
            ["calm", "angry"]
        )

        if st.button("Save & Continue", use_container_width=True):
            c.execute(
                "INSERT INTO preferences (user_id, sad_style, angry_style) VALUES (?, ?, ?)",
                (st.session_state.user, sad_pref, angry_pref)
            )
            conn.commit()

            st.session_state.page = "camera"
            st.rerun()

# =====================================================
# ================= MAIN APP (FULL WIDTH) ==============
# =====================================================
else:
    # -------- TOP BAR --------
    top_left, top_right = st.columns([6, 1])

    with top_left:
        st.title("🎵 Emotion Based Music Recommender")

    with top_right:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.user:
            if st.button("Logout"):
                st.session_state.user = None
                st.session_state.page = "auth"
                st.rerun()

    # -------- LOAD DATA --------
    @st.cache_data
    def load_music_data():
        df = pd.read_csv("spotify_tracks1.csv")
        return df.dropna(
            subset=["track_name", "artist_name", "valence", "energy", "language"]
        )

    music_df = load_music_data()

    # -------- LOAD MODEL --------
    @st.cache_resource
    def load_emotion_model():
        return load_model("CNN_Model.h5")

    model = load_emotion_model()
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    # -------- LOAD SAVED PREFS --------
    sad_pref, angry_pref = "sad", "angry"
    if st.session_state.user:
        c.execute(
            "SELECT sad_style, angry_style FROM preferences WHERE user_id=?",
            (st.session_state.user,)
        )
        row = c.fetchone()
        if row:
            sad_pref, angry_pref = row

    # -------- OPTIONAL PERSONALIZATION --------
    with st.expander("🎯 Optional Personalization"):
        preferred_languages = st.multiselect(
            "Preferred languages",
            options=sorted(music_df["language"].unique())
        )

        artist_input = st.text_input("Preferred artists (comma separated)")
        preferred_artists = [
            a.strip().lower() for a in artist_input.split(",") if a.strip()
        ]

    # -------- EMOTION RANGES --------
    def get_ranges(emotion):
        if emotion == "Sad":
            return {
                "sad": (0.0, 0.3, 0.0, 0.4),
                "uplifting": (0.5, 0.8, 0.6, 1.0),
                "upbeat": (0.7, 1.0, 0.7, 1.0)
            }[sad_pref]

        if emotion == "Angry":
            return {
                "calm": (0.4, 0.6, 0.2, 0.4),
                "angry": (0.0, 0.4, 0.7, 1.0)
            }[angry_pref]

        return {
            "Happy": (0.7, 1.0, 0.6, 1.0),
            "Neutral": (0.4, 0.6, 0.4, 0.6),
            "Fear": (0.2, 0.4, 0.3, 0.5),
            "Surprise": (0.6, 0.8, 0.5, 0.7),
            "Disgust": (0.0, 0.3, 0.0, 0.3)
        }[emotion]

    # -------- FACE DETECTION (FULL WIDTH) --------
    mp_face = mp.solutions.face_detection.FaceDetection(0, 0.5)
    st.markdown("### 📷 Capture your face")
    img = st.camera_input("")
    current_emotion = None

    if img:
        frame = np.array(Image.open(img))
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

                face = frame[y:y+bh, x:x+bw]
                if face.size == 0:
                    continue

                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (48, 48)) / 255.0
                rgb_like = np.stack((resized,) * 3, axis=-1)
                input_face = rgb_like.reshape(1, 48, 48, 3)

                pred = model.predict(input_face, verbose=0)
                current_emotion = EMOTIONS[np.argmax(pred)]

                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                cv2.putText(
                    frame, current_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                )

            st.image(frame, caption=f"Detected Emotion: {current_emotion}", channels="BGR")

    # -------- RECOMMEND --------
    if st.button("🎧 Recommend Songs"):
        if not current_emotion:
            st.warning("Please capture your face first.")
        else:
            v1, v2, e1, e2 = get_ranges(current_emotion)

            songs = music_df[
                (music_df.valence.between(v1, v2)) &
                (music_df.energy.between(e1, e2))
            ]

            if preferred_languages:
                songs = songs[songs.language.isin(preferred_languages)]

            if preferred_artists:
                songs = songs[
                    songs.artist_name.str.lower()
                    .str.contains("|".join(preferred_artists), regex=True)
                ]

            if songs.empty:
                st.info(
                    "Sorry, no songs match your selected language or artist preferences. "
                    "Try changing the filters."
                )
            else:
                songs = songs.sample(min(6, len(songs)))
                for _, row in songs.iterrows():
                    yt = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(
                        f"{row.track_name} {row.artist_name}"
                    )
                    st.markdown(
                        f"🎵 **{row.track_name}** – {row.artist_name}\n"
                        f"[▶ YouTube]({yt}) | [▶ Spotify]({row.track_url})"
                    )

                    if st.session_state.user:
                        c.execute(
                            "INSERT INTO history VALUES (?, ?, ?, ?)",
                            (st.session_state.user, row.track_name, current_emotion, datetime.now())
                        )
                        conn.commit()



