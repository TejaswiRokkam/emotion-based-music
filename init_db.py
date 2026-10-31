import sqlite3

conn = sqlite3.connect("users.db")
c = conn.cursor()

# ---------------- USERS TABLE ----------------
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

# ---------------- MOOD PREFERENCES ----------------
c.execute("""
CREATE TABLE IF NOT EXISTS user_mood_preferences (
    user_id INTEGER,
    emotion TEXT,
    mood TEXT,
    weight INTEGER DEFAULT 1,
    PRIMARY KEY (user_id, emotion, mood)
)
""")

# ---------------- LIKED SONGS ----------------
c.execute("""
CREATE TABLE IF NOT EXISTS liked_songs (
    user_id INTEGER,
    emotion TEXT,
    track_name TEXT,
    artist_name TEXT,
    mood TEXT,
    liked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# ---------------- SKIPPED SONGS ----------------
c.execute("""
CREATE TABLE IF NOT EXISTS skipped_songs (
    user_id INTEGER,
    emotion TEXT,
    track_name TEXT,
    artist_name TEXT,
    mood TEXT,
    skipped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("Database initialized successfully.")
