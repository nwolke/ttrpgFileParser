"""Cataloguing service – scans a directory, parses files, builds the index."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app import database as db
from app.chunker import chunk_text
from app.config import settings
from app.parsers import get_parser, SUPPORTED_EXTENSIONS
from app.search.embeddings import get_embedding_model
from app.search.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class CatalogueStats:
    total: int = 0
    indexed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def _validate_directory(directory: str) -> Path:
    """Resolve and validate the target directory.

    Raises:
        FileNotFoundError: If the path does not exist.
        NotADirectoryError: If the path is not a directory.
        PermissionError: If the path is outside ``settings.base_dir``.
    """
    root = Path(directory).resolve()
    allowed = settings.base_dir.resolve()

    # Reject any path that escapes the configured base directory.
    try:
        root.relative_to(allowed)
    except ValueError:
        raise PermissionError(
            f"Directory {directory!r} is outside the allowed base path "
            f"({allowed}).  Set TTRPG_BASE_DIR to change the restriction."
        )

    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {directory!r}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory!r}")

    return root


def catalogue_directory(directory: str, recursive: bool = True) -> CatalogueStats:
    """Scan *directory*, parse every supported file, and update the index.

    Args:
        directory: Absolute or relative path to scan.
        recursive: When ``True`` (default) descend into sub-directories.

    Returns:
        A :class:`CatalogueStats` summary of what was processed.
    """
    root = _validate_directory(directory)

    stats = CatalogueStats()
    store = VectorStore(settings.db_path)
    model = None  # loaded lazily on the first file that needs embedding

    pattern = "**/*" if recursive else "*"
    paths = [p for p in root.glob(pattern) if p.is_file()]
    stats.total = len(paths)

    for path in paths:
        parser = get_parser(path)
        if parser is None:
            logger.debug("Skipping unsupported file: %s", path)
            stats.skipped += 1
            continue

        with db.get_connection(settings.db_path) as conn:
            file_id = db.upsert_file(
                conn,
                name=path.name,
                path=str(path),
                file_type=path.suffix.lower().lstrip("."),
                size_bytes=path.stat().st_size,
            )
            conn.commit()

        try:
            text = parser.extract_text(path)
        except Exception as exc:
            logger.warning("Failed to parse '%s': %s", path, exc)
            with db.get_connection(settings.db_path) as conn:
                db.mark_file_error(conn, file_id, str(exc))
                conn.commit()
            stats.failed += 1
            stats.errors.append(f"{path.name}: {exc}")
            continue

        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            with db.get_connection(settings.db_path) as conn:
                db.mark_file_error(conn, file_id, "No text chunks produced after parsing.")
                conn.commit()
            stats.failed += 1
            stats.errors.append(f"{path.name}: no text extracted")
            continue

        if model is None:
            model = get_embedding_model(settings.embedding_model)

        embeddings = model.embed(chunks)
        store.index_chunks(file_id, chunks, embeddings)

        with db.get_connection(settings.db_path) as conn:
            db.mark_file_indexed(conn, file_id)
            conn.commit()

        logger.info("Indexed '%s' (%d chunks)", path.name, len(chunks))
        stats.indexed += 1

    return stats
