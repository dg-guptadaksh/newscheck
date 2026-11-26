import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Database:
    def __init__(self, path):
        self.path = path

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self, claims_path):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript("""
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS claims (
                    id INTEGER PRIMARY KEY, claim TEXT NOT NULL, publisher TEXT NOT NULL,
                    rating TEXT NOT NULL, url TEXT NOT NULL,
                    UNIQUE(claim, publisher, url)
                );
                CREATE TABLE IF NOT EXISTS screenings (
                    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, created_at TEXT NOT NULL
                        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    headline TEXT NOT NULL, result TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS screenings_session ON screenings(session_id, id DESC);
            """)
            if claims_path.is_file() and not connection.execute("SELECT 1 FROM claims LIMIT 1").fetchone():
                rows = json.loads(claims_path.read_text(encoding="utf-8"))
                connection.executemany("INSERT OR IGNORE INTO claims(claim,publisher,rating,url) VALUES (?,?,?,?)",
                                       ((r["claim"], r["publisher"], r["rating"], r["url"]) for r in rows))

    def claims(self):
        with self.connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM claims ORDER BY id")]

    def save(self, session, headline, result):
        with self.connect() as connection:
            cursor = connection.execute("INSERT INTO screenings(session_id, headline, result) VALUES (?,?,?)",
                                        (session, headline, json.dumps(result, ensure_ascii=False)))
            return cursor.lastrowid

    def history(self, session):
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, created_at, headline, result FROM screenings WHERE session_id=? ORDER BY id DESC LIMIT 50",
                (session,))
            return [{"id": row["id"], "created_at": row["created_at"], "headline": row["headline"],
                     "result": json.loads(row["result"])} for row in rows]

    def clear(self, session):
        with self.connect() as connection:
            connection.execute("DELETE FROM screenings WHERE session_id=?", (session,))
