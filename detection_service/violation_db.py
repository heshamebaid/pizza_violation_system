import sqlite3
from typing import List, Tuple
import os
from datetime import datetime

def get_db_path():
    db_dir = os.path.join(os.path.dirname(__file__), '..', 'shared')
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, 'violations.db')

def init_db():
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        frame_path TEXT,
        timestamp TEXT,
        bboxes TEXT,
        labels TEXT
    )''')
    conn.commit()
    conn.close()

def save_violation(frame_path: str, bboxes: List[Tuple[int,int,int,int]], labels: List[str], timestamp: str = None):
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute('''INSERT INTO violations (frame_path, timestamp, bboxes, labels) VALUES (?, ?, ?, ?)''',
              (frame_path, timestamp, str(bboxes), str(labels)))
    conn.commit()
    conn.close()

# Call init_db() on import
init_db() 