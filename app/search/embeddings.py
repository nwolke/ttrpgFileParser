"""Thin wrapper around sentence-transformers for embedding generation."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Loads a sentence-transformers model and produces L2-normalised embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        logger.info("Loading embedding model '%s' …", model_name)
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 array of L2-normalised embeddings.

        Args:
            texts: List of strings to embed.

        Returns:
            A 2-D numpy array with shape ``(len(texts), embedding_dim)``.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """Return a cached singleton instance of :class:`EmbeddingModel`."""
    return EmbeddingModel(model_name)
