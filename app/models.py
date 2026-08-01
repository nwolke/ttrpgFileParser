"""Pydantic models for API request and response payloads."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CatalogueRequest(BaseModel):
    """Request body for the catalogue endpoint."""

    directory: str
    recursive: bool = True


class CatalogueResponse(BaseModel):
    """Summary returned after a catalogue operation completes."""

    total_files: int
    indexed: int
    failed: int
    skipped: int
    errors: list[str]


class FileRecord(BaseModel):
    """Metadata for a single catalogued file."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    path: str
    file_type: str
    size_bytes: int
    indexed_at: Optional[str] = None
    status: str
    error_msg: Optional[str] = None


class SearchRequest(BaseModel):
    """Request body for the search endpoint."""

    query: str
    top_k: Optional[int] = None


class SearchResult(BaseModel):
    """A single file returned from a search query."""

    file_id: int
    name: str
    path: str
    file_type: str
    score: float
    matched_text: str


class SearchResponse(BaseModel):
    """Response envelope for a search query."""

    query: str
    results: list[SearchResult]
