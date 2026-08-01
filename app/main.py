"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, status

from app import database as db
from app.cataloguer import catalogue_directory
from app.config import settings
from app.models import (
    CatalogueRequest,
    CatalogueResponse,
    FileRecord,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.search.embeddings import get_embedding_model
from app.search.vector_store import VectorStore

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Application lifecycle                                                         #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure the database is initialised before accepting requests.
    db.init_db(settings.db_path)
    yield


app = FastAPI(
    title="TTRPG File Parser",
    description=(
        "Catalogue, parse, and semantically search through TTRPG files "
        "(PDFs, images, text) using AI-powered vector embeddings."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Catalogue endpoints                                                           #
# --------------------------------------------------------------------------- #

@app.post(
    "/catalogue",
    response_model=CatalogueResponse,
    summary="Catalogue a directory",
    description=(
        "Scan a directory for supported files (PDF, images, text), extract "
        "their text content, generate vector embeddings, and store everything "
        "in the local database.  This is a synchronous operation; for large "
        "directories it may take several minutes."
    ),
)
def catalogue(request: CatalogueRequest) -> CatalogueResponse:
    try:
        stats = catalogue_directory(request.directory, recursive=request.recursive)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during cataloguing")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return CatalogueResponse(
        total_files=stats.total,
        indexed=stats.indexed,
        failed=stats.failed,
        skipped=stats.skipped,
        errors=stats.errors,
    )


# --------------------------------------------------------------------------- #
# File endpoints                                                                #
# --------------------------------------------------------------------------- #

@app.get(
    "/files",
    response_model=list[FileRecord],
    summary="List catalogued files",
)
def list_files() -> list[FileRecord]:
    with db.get_connection(settings.db_path) as conn:
        rows = db.list_files(conn)
    return [FileRecord(**dict(row)) for row in rows]


@app.get(
    "/files/{file_id}",
    response_model=FileRecord,
    summary="Get a single file record",
)
def get_file(file_id: int) -> FileRecord:
    with db.get_connection(settings.db_path) as conn:
        row = db.get_file(conn, file_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    return FileRecord(**dict(row))


@app.delete(
    "/files/{file_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a file from the catalogue",
)
def delete_file(file_id: int) -> None:
    with db.get_connection(settings.db_path) as conn:
        deleted = db.delete_file(conn, file_id)
        conn.commit()
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")


# --------------------------------------------------------------------------- #
# Search endpoint                                                               #
# --------------------------------------------------------------------------- #

@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search",
    description=(
        "Search for files matching a natural-language query.  "
        "For example: 'give me all PDFs that have crafting rules'.  "
        "Returns files ranked by semantic similarity."
    ),
)
def search(request: SearchRequest) -> SearchResponse:
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty.",
        )

    top_k = request.top_k or settings.top_k_results
    model = get_embedding_model(settings.embedding_model)
    store = VectorStore(settings.db_path)

    query_vec = model.embed([request.query])[0]
    chunk_results = store.search(query_vec, top_k=top_k)

    results: list[SearchResult] = []
    with db.get_connection(settings.db_path) as conn:
        for cr in chunk_results:
            row = db.get_file(conn, cr.file_id)
            if row is None:
                continue
            results.append(
                SearchResult(
                    file_id=row["id"],
                    name=row["name"],
                    path=row["path"],
                    file_type=row["file_type"],
                    score=cr.score,
                    matched_text=cr.matched_text,
                )
            )

    return SearchResponse(query=request.query, results=results)
