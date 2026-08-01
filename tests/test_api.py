"""Integration tests for the FastAPI endpoints."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Patch the database path to a temp location before importing the app.
@pytest.fixture(autouse=True)
def use_temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr("app.config.settings.db_path", db_file)
    monkeypatch.setattr("app.config.settings.data_dir", tmp_path)

    # Re-init the database so the app lifespan finds the new path.
    from app import database as db
    db.init_db(db_file)
    yield


@pytest.fixture
def client(use_temp_db):
    from app.main import app
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# /files                                                                        #
# --------------------------------------------------------------------------- #

def test_list_files_empty(client):
    response = client.get("/files")
    assert response.status_code == 200
    assert response.json() == []


def test_get_file_not_found(client):
    response = client.get("/files/9999")
    assert response.status_code == 404


def test_delete_file_not_found(client):
    response = client.delete("/files/9999")
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# /catalogue                                                                    #
# --------------------------------------------------------------------------- #

def test_catalogue_invalid_directory(client):
    response = client.post("/catalogue", json={"directory": "/nonexistent/path/xyz"})
    assert response.status_code == 400


def test_catalogue_not_a_directory(client, tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hello", encoding="utf-8")
    response = client.post("/catalogue", json={"directory": str(f)})
    assert response.status_code == 400


def test_catalogue_empty_directory(client, tmp_path):
    empty_dir = tmp_path / "to_catalogue"
    empty_dir.mkdir()
    response = client.post("/catalogue", json={"directory": str(empty_dir)})
    assert response.status_code == 200
    body = response.json()
    assert body["total_files"] == 0
    assert body["indexed"] == 0


def test_catalogue_indexes_text_file(client, tmp_path):
    txt = tmp_path / "rules.txt"
    txt.write_text(
        "Crafting rules: to craft an item you need materials and a forge. "
        "Combine components using the crafting table.",
        encoding="utf-8",
    )

    # Mock the embedding model so tests run without a real model download.
    mock_embed = MagicMock(return_value=np.random.rand(1, 384).astype(np.float32))
    with patch("app.cataloguer.get_embedding_model") as mock_get_model:
        mock_get_model.return_value.embed = mock_embed
        mock_get_model.return_value.dim = 384

        response = client.post("/catalogue", json={"directory": str(tmp_path)})

    assert response.status_code == 200
    body = response.json()
    assert body["indexed"] == 1
    assert body["failed"] == 0


# --------------------------------------------------------------------------- #
# /search                                                                       #
# --------------------------------------------------------------------------- #

def test_search_empty_query(client):
    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 400


def test_search_no_results_when_empty_index(client):
    mock_embed = MagicMock(return_value=np.random.rand(1, 384).astype(np.float32))
    with patch("app.main.get_embedding_model") as mock_get_model:
        mock_get_model.return_value.embed = mock_embed

        response = client.post("/search", json={"query": "crafting rules"})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "crafting rules"
    assert body["results"] == []


def test_search_returns_indexed_file(client, tmp_path):
    txt = tmp_path / "crafting_guide.txt"
    txt.write_text(
        "The crafting guide explains how to forge swords. "
        "You need iron ingots, a hammer, and an anvil.",
        encoding="utf-8",
    )

    fake_embedding = np.array([[0.5, 0.5] + [0.0] * 382], dtype=np.float32)
    fake_query_vec = np.array([0.5, 0.5] + [0.0] * 382, dtype=np.float32)

    with patch("app.cataloguer.get_embedding_model") as mock_catalogue_model, \
         patch("app.main.get_embedding_model") as mock_search_model:

        mock_catalogue_model.return_value.embed = MagicMock(return_value=fake_embedding)
        mock_catalogue_model.return_value.dim = 384
        mock_search_model.return_value.embed = MagicMock(return_value=fake_embedding)

        client.post("/catalogue", json={"directory": str(tmp_path)})

        # Now search
        mock_search_model.return_value.embed = MagicMock(
            return_value=np.array([fake_query_vec])
        )
        response = client.post("/search", json={"query": "crafting"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) >= 1
    names = [r["name"] for r in body["results"]]
    assert "crafting_guide.txt" in names


# --------------------------------------------------------------------------- #
# /files – CRUD after cataloguing                                               #
# --------------------------------------------------------------------------- #

def test_list_and_get_file_after_catalogue(client, tmp_path):
    txt = tmp_path / "spells.txt"
    txt.write_text("A list of wizard spells for the campaign.", encoding="utf-8")

    mock_embed = MagicMock(return_value=np.random.rand(1, 384).astype(np.float32))
    with patch("app.cataloguer.get_embedding_model") as mock_get_model:
        mock_get_model.return_value.embed = mock_embed
        mock_get_model.return_value.dim = 384
        client.post("/catalogue", json={"directory": str(tmp_path)})

    files = client.get("/files").json()
    assert len(files) == 1
    file_id = files[0]["id"]
    assert files[0]["name"] == "spells.txt"

    single = client.get(f"/files/{file_id}").json()
    assert single["id"] == file_id

    # Delete it.
    resp = client.delete(f"/files/{file_id}")
    assert resp.status_code == 204

    assert client.get(f"/files/{file_id}").status_code == 404
