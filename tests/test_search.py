"""Tests for the vector store and search logic."""

import io
from pathlib import Path

import numpy as np
import pytest

from app.search.vector_store import VectorStore, _to_bytes, _from_bytes


# --------------------------------------------------------------------------- #
# Serialisation helpers                                                         #
# --------------------------------------------------------------------------- #

def test_round_trip_bytes():
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    restored = _from_bytes(_to_bytes(vec))
    np.testing.assert_array_almost_equal(vec, restored)


# --------------------------------------------------------------------------- #
# VectorStore – end-to-end with a temp database                                 #
# --------------------------------------------------------------------------- #

@pytest.fixture
def store(tmp_path):
    from app.database import init_db
    db_path = tmp_path / "test.db"
    init_db(db_path)
    return VectorStore(db_path), db_path


def _add_file(db_path: Path, name: str) -> int:
    from app import database as db
    with db.get_connection(db_path) as conn:
        file_id = db.upsert_file(conn, name=name, path=f"/fake/{name}", file_type="txt", size_bytes=100)
        db.mark_file_indexed(conn, file_id)
        conn.commit()
    return file_id


def test_search_empty_store(store):
    vs, db_path = store
    query = np.zeros(4, dtype=np.float32)
    results = vs.search(query, top_k=5)
    assert results == []


def test_search_returns_closest_file(store):
    vs, db_path = store

    # Create two files with orthogonal embeddings.
    file_a = _add_file(db_path, "crafting_rules.txt")
    file_b = _add_file(db_path, "encounter_tables.txt")

    emb_a = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    emb_b = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    vs.index_chunks(file_a, ["Crafting rules text"], emb_a)
    vs.index_chunks(file_b, ["Encounter table text"], emb_b)

    # Query aligned with file_a.
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = vs.search(query, top_k=2)

    assert len(results) == 2
    assert results[0].file_id == file_a
    assert results[0].score > results[1].score


def test_search_top_k_limits_results(store):
    vs, db_path = store

    for i in range(5):
        fid = _add_file(db_path, f"file_{i}.txt")
        vec = np.random.rand(1, 4).astype(np.float32)
        vec /= np.linalg.norm(vec)
        vs.index_chunks(fid, [f"text {i}"], vec)

    query = np.random.rand(4).astype(np.float32)
    results = vs.search(query, top_k=3)
    assert len(results) <= 3


def test_search_deduplicates_by_file(store):
    """Multiple chunks from the same file should yield only one result entry."""
    vs, db_path = store

    fid = _add_file(db_path, "big_book.txt")
    # Add three chunks for the same file.
    chunks = ["Chapter 1", "Chapter 2", "Chapter 3"]
    embeddings = np.random.rand(3, 4).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    vs.index_chunks(fid, chunks, embeddings)

    query = np.random.rand(4).astype(np.float32)
    results = vs.search(query, top_k=10)

    file_ids = [r.file_id for r in results]
    assert file_ids.count(fid) == 1, "Duplicate file entries in search results"
