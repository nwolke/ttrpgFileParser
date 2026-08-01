"""SQLite database helpers – schema creation and CRUD helpers."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path


# --------------------------------------------------------------------------- #
# Schema                                                                        #
# --------------------------------------------------------------------------- #

_DDL = """
CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL UNIQUE,
    file_type   TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    indexed_at  TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    error_msg   TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id    INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL
);
"""


def init_db(db_path: Path) -> None:
    """Create the database file and tables if they do not exist yet."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_DDL)
        conn.commit()


@contextmanager
def get_connection(db_path: Path):
    """Yield a SQLite connection with foreign-key enforcement enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# File CRUD                                                                     #
# --------------------------------------------------------------------------- #

def upsert_file(conn: sqlite3.Connection, name: str, path: str, file_type: str, size_bytes: int) -> int:
    """Insert a new file record or reset an existing one for re-indexing.

    Returns the row id.
    """
    cur = conn.execute(
        "SELECT id FROM files WHERE path = ?",
        (path,),
    )
    row = cur.fetchone()
    if row:
        file_id = row["id"]
        conn.execute(
            "UPDATE files SET name=?, file_type=?, size_bytes=?, status='pending', "
            "error_msg=NULL, indexed_at=NULL WHERE id=?",
            (name, file_type, size_bytes, file_id),
        )
        # Remove old chunks (cascade removes embeddings too).
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
    else:
        cur = conn.execute(
            "INSERT INTO files (name, path, file_type, size_bytes, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            (name, path, file_type, size_bytes),
        )
        file_id = cur.lastrowid
    return file_id


def mark_file_indexed(conn: sqlite3.Connection, file_id: int) -> None:
    conn.execute(
        "UPDATE files SET status='indexed', indexed_at=datetime('now') WHERE id=?",
        (file_id,),
    )


def mark_file_error(conn: sqlite3.Connection, file_id: int, error: str) -> None:
    conn.execute(
        "UPDATE files SET status='error', error_msg=? WHERE id=?",
        (error, file_id),
    )


def get_file(conn: sqlite3.Connection, file_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()


def list_files(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM files ORDER BY name").fetchall()


def delete_file(conn: sqlite3.Connection, file_id: int) -> bool:
    cur = conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
    return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# Chunk CRUD                                                                    #
# --------------------------------------------------------------------------- #

def insert_chunk(conn: sqlite3.Connection, file_id: int, chunk_index: int, text: str) -> int:
    cur = conn.execute(
        "INSERT INTO chunks (file_id, chunk_index, text) VALUES (?, ?, ?)",
        (file_id, chunk_index, text),
    )
    return cur.lastrowid


def insert_embedding(conn: sqlite3.Connection, chunk_id: int, vector_bytes: bytes) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (chunk_id, vector) VALUES (?, ?)",
        (chunk_id, vector_bytes),
    )


def load_all_embeddings(conn: sqlite3.Connection) -> list[dict]:
    """Return every chunk that has an embedding, with file metadata."""
    rows = conn.execute(
        """
        SELECT e.chunk_id, e.vector, c.file_id, c.text
        FROM   embeddings e
        JOIN   chunks c ON c.id = e.chunk_id
        JOIN   files  f ON f.id = c.file_id
        WHERE  f.status = 'indexed'
        ORDER  BY e.chunk_id
        """
    ).fetchall()
    return [dict(r) for r in rows]
