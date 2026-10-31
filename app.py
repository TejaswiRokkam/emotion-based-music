import streamlit as st
import sqlite3
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import urllib.parse
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Emotion Adaptive Music Recommender")

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "auth"

if "user" not in st.session_state:
    st.session_state.user = None

if "recommended_songs" not in st.session_state:
    st.session_state.recommended_songs = None

if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = None

if "shown_songs" not in st.session_state:
    st.session_state.shown_songs = set()

# ---------------- DATABASE CONNECTION ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

# =====================================================
# AUTH PAGE
# =====================================================
if st.session_state.page == "auth":

    st.title("🎵 Emotion Adaptive Music Recommender")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            c.execute("SELECT id FROM users WHERE username=? AND password=?",
                      (username, password))
            row = c.fetchone()
            if row:
                st.session_state.user = row[0]
                st.session_state.page = "main"
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            try:
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                          (new_user, new_pass))
                conn.commit()
                st.success("Account created. Please login.")
            except:
                st.error("Username already exists.")

# =====================================================
# MAIN PAGE
# =====================================================
else:

    col1, col2 = st.columns([6, 1])

    with col1:
        st.title("🎵 Emotion Adaptive Music Recommender")

    with col2:
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.page = "auth"
            st.session_state.recommended_songs = None
            st.session_state.current_emotion = None
            st.session_state.shown_songs = set()
            st.rerun()

    # -------- LOAD MUSIC --------
    @st.cache_data
    def load_music():
        df = pd.read_csv("spotify_tracks1.csv")
        return df.dropna(subset=["track_name", "artist_name", "valence", "energy", "language"])

    music_df = load_music()

    # -------- SONG MOOD CLASSIFICATION --------
    def classify_song(row):
        if row["valence"] <= 0.35 and row["energy"] <= 0.45:
            return "Sad"
        elif 0.35 <= row["valence"] <= 0.6 and row["energy"] <= 0.35 and row.get("acousticness", 0) >= 0.5:
            return "Calm"
        elif row["valence"] >= 0.7 and 0.5 <= row["energy"] <= 0.85:
            return "Happy"
        elif row["valence"] >= 0.6 and row["energy"] >= 0.75 and row.get("danceability", 0) >= 0.7:
            return "Upbeat"
        elif 0.5 <= row["valence"] <= 0.8 and 0.6 <= row["energy"] <= 0.9:
            return "Motivational"
        elif row["energy"] >= 0.8 and row.get("loudness", -60) >= -6:
            return "Rage"
        elif row["valence"] <= 0.3 and 0.5 <= row["energy"] <= 0.8:
            return "Tense"
        else:
            return "Other"

    music_df["mood"] = music_df.apply(classify_song, axis=1)

    # -------- OPTIONAL PERSONALIZATION --------
    with st.expander("🎯 Optional Personalization"):
        preferred_languages = st.multiselect(
            "Preferred languages",
            options=sorted(music_df["language"].unique())
        )

        artist_input = st.text_input("Preferred artists (comma separated)")
        preferred_artists = [a.strip().lower() for a in artist_input.split(",") if a.strip()]

    filtered_base = music_df.copy()

    if preferred_languages:
        filtered_base = filtered_base[filtered_base["language"].isin(preferred_languages)]

    if preferred_artists:
        filtered_base = filtered_base[
            filtered_base["artist_name"].str.lower()
            .str.contains("|".join(preferred_artists), regex=True)
        ]

    # -------- LOAD MODEL --------
    @st.cache_resource
    def load_model_cached():
        return load_model("CNN_Model.h5")

    model = load_model_cached()
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    # -------- CAMERA --------
    mp_face = mp.solutions.face_detection.FaceDetection(0, 0.5)
    st.subheader("📷 Capture your face")
    img = st.camera_input("")

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
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (48, 48)) / 255.0
                rgb_like = np.stack((resized,) * 3, axis=-1)
                input_face = rgb_like.reshape(1, 48, 48, 3)

                pred = model.predict(input_face, verbose=0)
                emotion = EMOTIONS[np.argmax(pred)]

                if st.session_state.current_emotion != emotion:
                    st.session_state.recommended_songs = None
                    st.session_state.shown_songs = set()

                st.session_state.current_emotion = emotion

                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        st.image(frame, channels="BGR")

    # -------- SEARCH --------
    st.subheader("🔍 Search Songs")

    if not st.session_state.current_emotion:
        st.info("Capture emotion before searching.")
    else:
        search_query = st.text_input("Search by song or artist")

        if search_query:
            results = music_df[
                music_df["track_name"].str.contains(search_query, case=False, na=False) |
                music_df["artist_name"].str.contains(search_query, case=False, na=False)
            ].head(15)

            if not results.empty:
                selected = st.selectbox(
                    "Select a song",
                    results["track_name"] + " - " + results["artist_name"]
                )

                if selected:
                    row = results[
                        (results["track_name"] + " - " + results["artist_name"]) == selected
                    ].iloc[0]

                    st.markdown(f"### 🎵 {row['track_name']}")
                    st.write(row["artist_name"])

                    yt = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(
                        f"{row['track_name']} {row['artist_name']}"
                    )

                    st.markdown(f"[YouTube]({yt}) | [Spotify]({row['track_url']})")

                    if st.button("👍 Like", key="search_like"):

                        c.execute("""
                            INSERT INTO user_mood_preferences
                            VALUES (?,?,?,1)
                            ON CONFLICT(user_id,emotion,mood)
                            DO UPDATE SET weight = weight + 1
                        """, (st.session_state.user,
                              st.session_state.current_emotion,
                              row["mood"]))

                        c.execute("""
                            INSERT INTO liked_songs (user_id,emotion,track_name,artist_name,mood)
                            VALUES (?,?,?,?,?)
                        """, (st.session_state.user,
                              st.session_state.current_emotion,
                              row["track_name"],
                              row["artist_name"],
                              row["mood"]))

                        conn.commit()
                        st.success("Saved & learning updated")
            else:
                st.warning("No songs found.")
    
    # -------- RECOMMEND --------
    st.write("Detected Emotion:", st.session_state.current_emotion)

    if st.button("🎧 Recommend Songs"):

        emotion = st.session_state.current_emotion

        if emotion:

            # Fetch learned mood weights
            c.execute("""
                SELECT mood, SUM(weight)
                FROM user_mood_preferences
                WHERE user_id=? AND emotion=?
                GROUP BY mood
            """, (st.session_state.user, emotion))

            learned = c.fetchall()
            recommendations = []

            if learned:
                mood_weights = {row[0]: row[1] for row in learned}
                total_weight = sum(mood_weights.values())

                # Step 1: proportional allocation
                for mood, weight in mood_weights.items():

                    proportion = weight / total_weight
                    count = round(6 * proportion)

                    mood_songs = filtered_base[filtered_base["mood"] == mood]

                    # Remove skipped songs
                    c.execute("""
                        SELECT track_name, artist_name
                        FROM skipped_songs
                        WHERE user_id=? AND emotion=?
                    """, (st.session_state.user, emotion))

                    skipped = set(c.fetchall())

                    if skipped:
                        mood_songs = mood_songs[~mood_songs.apply(
                            lambda x: (x["track_name"], x["artist_name"]) in skipped,
                            axis=1
                        )]

                    # Remove already shown songs
                    if st.session_state.shown_songs:
                        mood_songs = mood_songs[~mood_songs.apply(
                            lambda x: (x["track_name"], x["artist_name"]) in st.session_state.shown_songs,
                            axis=1
                        )]

                    if not mood_songs.empty and count > 0:
                        selected = mood_songs.sample(min(count, len(mood_songs)))
                        recommendations.extend(selected.to_dict("records"))

                # Step 2: Fill remaining slots strictly from learned moods
                remaining = 6 - len(recommendations)

                if remaining > 0:
                    allowed_moods = list(mood_weights.keys())
                    extra_pool = filtered_base[filtered_base["mood"].isin(allowed_moods)]

                    # Remove skipped
                    c.execute("""
                        SELECT track_name, artist_name
                        FROM skipped_songs
                        WHERE user_id=? AND emotion=?
                    """, (st.session_state.user, emotion))

                    skipped = set(c.fetchall())

                    if skipped:
                        extra_pool = extra_pool[~extra_pool.apply(
                            lambda x: (x["track_name"], x["artist_name"]) in skipped,
                            axis=1
                        )]

                    # Remove shown
                    if st.session_state.shown_songs:
                        extra_pool = extra_pool[~extra_pool.apply(
                            lambda x: (x["track_name"], x["artist_name"]) in st.session_state.shown_songs,
                            axis=1
                        )]

                    if not extra_pool.empty:
                        extra_selected = extra_pool.sample(min(remaining, len(extra_pool)))
                        recommendations.extend(extra_selected.to_dict("records"))

            else:
                # If no learning yet, fallback to default neutral-safe behavior
                emotion_mood_map = {
                    "Happy": ["Happy", "Upbeat"],
                    "Sad": ["Sad", "Calm", "Motivational", "Upbeat"],
                    "Angry": ["Rage", "Motivational", "Calm"],
                    "Fear": ["Tense", "Calm"],
                    "Neutral": ["Calm", "Happy"],
                    "Surprise": ["Upbeat", "Happy"],
                    "Disgust": ["Tense", "Calm"]
                }

                allowed = emotion_mood_map.get(emotion, ["Other"])
                filtered = filtered_base[filtered_base["mood"].isin(allowed)]

                if not filtered.empty:
                    recommendations = filtered.sample(min(6, len(filtered))).to_dict("records")

            st.session_state.recommended_songs = recommendations

            for r in recommendations:
                st.session_state.shown_songs.add(
                    (r["track_name"], r["artist_name"])
                )

    # -------- DISPLAY --------
    if st.session_state.recommended_songs:
        st.subheader("🎶 Recommended Songs")

        for idx, row in enumerate(st.session_state.recommended_songs):

            st.markdown(f"### 🎵 {row['track_name']}")
            st.write(row["artist_name"])

            yt = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(
                f"{row['track_name']} {row['artist_name']}"
            )

            st.markdown(f"[YouTube]({yt}) | [Spotify]({row['track_url']})")

            col1, col2 = st.columns(2)

            if col1.button("👍", key=f"like_{idx}"):

                c.execute("""
                    INSERT INTO user_mood_preferences
                    VALUES (?,?,?,1)
                    ON CONFLICT(user_id,emotion,mood)
                    DO UPDATE SET weight = weight + 1
                """, (st.session_state.user,
                      st.session_state.current_emotion,
                      row["mood"]))

                c.execute("""
                    INSERT INTO liked_songs (user_id,emotion,track_name,artist_name,mood)
                    VALUES (?,?,?,?,?)
                """, (st.session_state.user,
                      st.session_state.current_emotion,
                      row["track_name"],
                      row["artist_name"],
                      row["mood"]))

                conn.commit()
                st.success("Saved")

            if col2.button("👎", key=f"skip_{idx}"):

                c.execute("""
                    INSERT INTO skipped_songs
                    (user_id, emotion, track_name, artist_name, mood)
                    VALUES (?,?,?,?,?)
                """, (st.session_state.user,
                      st.session_state.current_emotion,
                      row["track_name"],
                      row["artist_name"],
                      row["mood"]))

                conn.commit()
                st.success("Skipped")

            st.divider()

    # -------- BEHAVIOR ANALYSIS --------
    with st.expander("📊 View Your Emotional Listening Pattern"):

        c.execute("""
            SELECT emotion, mood, COUNT(*)
            FROM liked_songs
            WHERE user_id=?
            GROUP BY emotion, mood
            ORDER BY emotion
        """, (st.session_state.user,))

        rows = c.fetchall()

        if rows:
            df = pd.DataFrame(rows, columns=["Emotion", "Mood", "Count"])

            for emotion in df["Emotion"].unique():
                st.markdown(f"### When you are {emotion}:")

                emotion_df = df[df["Emotion"] == emotion]
                total = emotion_df["Count"].sum()

                for _, r in emotion_df.iterrows():
                    percent = round((r["Count"] / total) * 100)
                    st.write(f"- {r['Mood']} → {r['Count']} songs ({percent}%)")

                dominant = emotion_df.sort_values("Count", ascending=False).iloc[0]
                st.success(
                    f"You mostly prefer {dominant['Mood']} songs when you feel {emotion}."
                )
        else:
            st.info("Still learning your preferences.")

    # -------- RECENT LIKED SONGS --------
    with st.expander("🎵 Recently Liked Song"):

        c.execute("""
            SELECT emotion, track_name, artist_name, mood, liked_at
            FROM liked_songs
            WHERE user_id=?
            ORDER BY liked_at DESC
            LIMIT 15
        """, (st.session_state.user,))

        history = c.fetchall()

        if history:
            for row in history:
                st.write(f"{row[4]} | {row[0]} → {row[1]} - {row[2]} ({row[3]})")
        else:
            st.info("No liked songs yet.")
