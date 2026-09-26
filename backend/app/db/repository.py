"""
Data-access functions for documents/analyses/classification_results/keywords.

Kept as plain functions taking a sqlite3.Connection rather than a class
with implicit global state — makes every function trivially testable
with an in-memory or temp-file database, no framework required.
"""
import sqlite3
from typing import Optional


# ---------- documents ----------

def insert_document(
    conn: sqlite3.Connection,
    filename: str,
    file_type: str,
    raw_text: str,
    char_count: int,
    word_count: int,
    sentence_count: int,
) -> int:
    cur = conn.execute(
        """INSERT INTO documents
           (filename, file_type, raw_text, char_count, word_count, sentence_count)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (filename, file_type, raw_text, char_count, word_count, sentence_count),
    )
    conn.commit()
    return cur.lastrowid


def get_document(conn: sqlite3.Connection, document_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM documents WHERE id = ?", (document_id,)
    ).fetchone()


def list_documents(conn: sqlite3.Connection, limit: int = 50, offset: int = 0):
    return conn.execute(
        """SELECT * FROM documents ORDER BY uploaded_at DESC LIMIT ? OFFSET ?""",
        (limit, offset),
    ).fetchall()


def delete_document(conn: sqlite3.Connection, document_id: int) -> bool:
    cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.commit()
    return cur.rowcount > 0


# ---------- analyses ----------

def create_analysis(conn: sqlite3.Connection, document_id: int) -> int:
    cur = conn.execute(
        "INSERT INTO analyses (document_id, status) VALUES (?, 'pending')",
        (document_id,),
    )
    conn.commit()
    return cur.lastrowid


def complete_analysis(conn: sqlite3.Connection, analysis_id: int) -> None:
    conn.execute(
        "UPDATE analyses SET status = 'completed', completed_at = datetime('now') WHERE id = ?",
        (analysis_id,),
    )
    conn.commit()


def fail_analysis(conn: sqlite3.Connection, analysis_id: int, error_message: str) -> None:
    conn.execute(
        "UPDATE analyses SET status = 'failed', error_message = ?, completed_at = datetime('now') WHERE id = ?",
        (error_message, analysis_id),
    )
    conn.commit()


def get_latest_analysis(conn: sqlite3.Connection, document_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM analyses WHERE document_id = ?
           ORDER BY created_at DESC LIMIT 1""",
        (document_id,),
    ).fetchone()


# ---------- classification_results ----------

def save_classification_result(
    conn: sqlite3.Connection,
    analysis_id: int,
    predicted_category: str,
    confidence: float,
    model_version: str,
) -> int:
    cur = conn.execute(
        """INSERT INTO classification_results
           (analysis_id, predicted_category, confidence, model_version)
           VALUES (?, ?, ?, ?)""",
        (analysis_id, predicted_category, confidence, model_version),
    )
    conn.commit()
    return cur.lastrowid


def get_classification_result(conn: sqlite3.Connection, analysis_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM classification_results WHERE analysis_id = ?", (analysis_id,)
    ).fetchone()


# ---------- keywords ----------

def save_keywords(conn: sqlite3.Connection, analysis_id: int, keywords: list[tuple[str, float]]) -> None:
    conn.executemany(
        "INSERT INTO keywords (analysis_id, term, score) VALUES (?, ?, ?)",
        [(analysis_id, term, score) for term, score in keywords],
    )
    conn.commit()


def get_keywords(conn: sqlite3.Connection, analysis_id: int):
    return conn.execute(
        "SELECT term, score FROM keywords WHERE analysis_id = ? ORDER BY score DESC",
        (analysis_id,),
    ).fetchall()


# ---------- composite reads (for the API layer) ----------

def get_document_detail(conn: sqlite3.Connection, document_id: int) -> Optional[dict]:
    """Everything the 'Document Detail' / 'Analysis Result' page needs
    in one call: document + latest analysis + classification + keywords."""
    doc = get_document(conn, document_id)
    if doc is None:
        return None

    result = dict(doc)
    analysis = get_latest_analysis(conn, document_id)
    if analysis is None:
        result["analysis"] = None
        return result

    analysis_dict = dict(analysis)
    classification = get_classification_result(conn, analysis["id"])
    analysis_dict["classification"] = dict(classification) if classification else None
    analysis_dict["keywords"] = [dict(k) for k in get_keywords(conn, analysis["id"])]
    result["analysis"] = analysis_dict
    return result
