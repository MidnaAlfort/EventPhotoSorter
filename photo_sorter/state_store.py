"""Incremental checkpoints for event runs with tens of thousands of photos."""
import json
import sqlite3
from contextlib import closing


def load_records(path):
    if not path.is_file():
        return {}
    try:
        with closing(sqlite3.connect(path)) as db:
            return {key: json.loads(payload) for key, payload in db.execute('SELECT source, payload FROM records')}
    except (sqlite3.Error, ValueError):
        return {}


def save_record(path, source, record):
    with closing(sqlite3.connect(path, timeout=10)) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS records (source TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        db.execute('INSERT OR REPLACE INTO records VALUES (?, ?)', (source, json.dumps(record, ensure_ascii=False)))
