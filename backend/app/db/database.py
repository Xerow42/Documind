"""
SQLite connection + schema management, stdlib `sqlite3` only.

Design decision: SQLAlchemy was dropped in favor of raw sqlite3 for
this project. Two reasons, documented honestly:
  1. The scope here (4 small tables, no cross-database portability
     need) doesn't justify an ORM — plain SQL is simpler to explain
     in an interview and has zero extra dependencies.
  2. SQLAlchemy could not be installed in the build/verification
     sandbox this project was developed in (no outbound network
     access) — see README "Known Limitations". Reason (1) is why
     it's kept this way even though it could now be installed on
     your own machine.
"""
import os
import sqlite3
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "documind.db"


def get_db_path() -> str:
    return os.getenv("DATABASE_PATH", str(DEFAULT_DB_PATH))


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    file_type       TEXT NOT NULL CHECK (file_type IN ('pdf', 'txt')),
    raw_text        TEXT NOT NULL,
    char_count      INTEGER NOT NULL DEFAULT 0,
    word_count      INTEGER NOT NULL DEFAULT 0,
    sentence_count  INTEGER NOT NULL DEFAULT 0,
    uploaded_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    status          TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed')),
    error_message   TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at    TEXT
);

CREATE TABLE IF NOT EXISTS classification_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id         INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    predicted_category  TEXT NOT NULL,
    confidence          REAL NOT NULL,
    model_version       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS keywords (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id     INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    term            TEXT NOT NULL,
    score           REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analyses_document_id ON analyses(document_id);
CREATE INDEX IF NOT EXISTS idx_classification_results_analysis_id ON classification_results(analysis_id);
CREATE INDEX IF NOT EXISTS idx_keywords_analysis_id ON keywords(analysis_id);
"""


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
