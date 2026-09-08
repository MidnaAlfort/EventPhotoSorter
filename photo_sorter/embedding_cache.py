from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any


class ArtifactCache:
    """Disposable, content-addressed cache. Corruption never prevents sorting."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def get(self, namespace: str, key: str) -> Any | None:
        try:
            with self._lock, closing(sqlite3.connect(self.path, timeout=10)) as db:
                row = db.execute(
                    "SELECT payload FROM artifacts WHERE namespace=? AND key=?", (namespace, key)
                ).fetchone()
            return json.loads(row[0]) if row else None
        except (OSError, sqlite3.Error, ValueError):
            return None

    def put(self, namespace: str, key: str, payload: Any) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, closing(sqlite3.connect(self.path, timeout=10)) as db, db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS artifacts "
                    "(namespace TEXT, key TEXT, payload TEXT, PRIMARY KEY(namespace, key))"
                )
                db.execute(
                    "INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?)",
                    (namespace, key, json.dumps(payload)),
                )
        except (OSError, sqlite3.Error):
            pass
