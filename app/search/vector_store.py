"""In-memory vector store backed by SQLite for persistence.

Embeddings are stored as raw bytes in the ``embeddings`` table.  On each
search call the store loads all vectors from the database, stacks them into a
NumPy matrix and computes cosine similarity against the query vector.

For TTRPG collections this is perfectly fast: even a library of 500 PDF pages
split into ~5 chunks each (2 500 chunk vectors × 384 dims × 4 bytes) fits well
within RAM.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app import database as db

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    chunk_id: int
    file_id: int
    score: float
    matched_text: str


def _to_bytes(vector: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, vector)
    return buf.getvalue()


def _from_bytes(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data))


class VectorStore:
    """Facade for storing and querying document-chunk embeddings."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    # ------------------------------------------------------------------ #
    # Indexing                                                              #
    # ------------------------------------------------------------------ #

    def index_chunks(
        self,
        file_id: int,
        chunks: list[str],
        embeddings: np.ndarray,
    ) -> None:
        """Persist *chunks* and their *embeddings* for *file_id*.

        Existing chunks for this file must have been deleted before calling
        this (``database.upsert_file`` handles that).
        """
        with db.get_connection(self._db_path) as conn:
            for i, (text, vec) in enumerate(zip(chunks, embeddings)):
                chunk_id = db.insert_chunk(conn, file_id, i, text)
                db.insert_embedding(conn, chunk_id, _to_bytes(vec))
            conn.commit()

    # ------------------------------------------------------------------ #
    # Searching                                                             #
    # ------------------------------------------------------------------ #

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[ChunkResult]:
        """Return up to *top_k* chunks most similar to *query_vector*.

        Results are deduplicated by file: only the best-scoring chunk per
        file is kept so that the caller gets *files*, not individual chunks.
        """
        with db.get_connection(self._db_path) as conn:
            rows = db.load_all_embeddings(conn)

        if not rows:
            return []

        # Build matrix and metadata lists in one pass.
        vectors: list[np.ndarray] = []
        meta: list[dict] = []
        for row in rows:
            vectors.append(_from_bytes(row["vector"]))
            meta.append({"chunk_id": row["chunk_id"], "file_id": row["file_id"], "text": row["text"]})

        matrix = np.stack(vectors, axis=0)  # (N, D)

        # Cosine similarity – vectors are already L2-normalised.
        q = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        scores = matrix @ q  # (N,)

        # Keep best chunk per file.
        best: dict[int, tuple[float, int, str]] = {}  # file_id -> (score, chunk_id, text)
        for score, m in zip(scores.tolist(), meta):
            fid = m["file_id"]
            if fid not in best or score > best[fid][0]:
                best[fid] = (score, m["chunk_id"], m["text"])

        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]

        return [
            ChunkResult(
                chunk_id=chunk_id,
                file_id=fid,
                score=round(score, 4),
                matched_text=text,
            )
            for fid, (score, chunk_id, text) in ranked
        ]
