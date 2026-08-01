"""Application configuration via environment variables or defaults."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Directory used to persist the SQLite database and any cached data.
    data_dir: Path = Path("./data")

    # Full path of the SQLite database file.
    db_path: Path = Path("./data/ttrpg.db")

    # Sentence-transformers model used to generate embeddings.
    embedding_model: str = "all-MiniLM-L6-v2"

    # Number of characters per text chunk when indexing a document.
    chunk_size: int = 500

    # Character overlap between consecutive chunks.
    chunk_overlap: int = 50

    # Maximum number of matching files returned by a search.
    top_k_results: int = 10

    model_config = {"env_prefix": "TTRPG_"}


settings = Settings()
