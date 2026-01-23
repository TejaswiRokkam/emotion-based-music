import sqlite3

conn = sqlite3.connect("users.db")
c = conn.cursor()

# Users table
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

# Preferences table (behavior-based only)
c.execute("""
CREATE TABLE IF NOT EXISTS preferences (
    user_id INTEGER,
    sad_style TEXT,
    angry_style TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
""")

# History table
c.execute("""
CREATE TABLE IF NOT EXISTS history (
    user_id INTEGER,
    track_name TEXT,
    emotion TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()
